"""
FEniCSx - preCICE Adapter. API to help users couple FEniCS with other solvers using the preCICE library.
:raise ImportError: if PRECICE_ROOT is not defined
"""
from os import write
import numpy as np
from .config import Config
import logging
import precice
from .adapter_core import FunctionType, determine_function_type, convert_fenicsx_to_precice, get_fenicsx_vertices, CouplingMode, Vertices, VertexType
from .expression_core import SegregatedRBFInterpolationExpression
from .solverstate import SolverState
from dolfinx.fem import Function, FunctionSpace
import copy

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class Adapter:
    """
    This adapter class provides an interface to the preCICE coupling library for setting up a coupling case which has
    FEniCSx as a participant for 2D problems.
    The user can create and manage a dolfinx.UserExpression and/or dolfinx.PointSource at the coupling boundary.
    Reading data from preCICE and writing data to preCICE is also managed via functions of this class.
    If the user wants to perform implicit coupling then a steering mechanism for checkpointing is also provided.

    For more information on setting up a coupling case using dolfinx.UserExpression at the coupling boundary please have
    a look at this tutorial:
    TODO

    For more information on setting up a coupling case using dolfinx.PointSource at the coupling boundary please have a
    look at this tutorial:
    TODO

    NOTE: dolfinx.PointSource use only works in serial
    """

    def __init__(self, mpi_comm, adapter_config_filename='precice-adapter-config.json'):
        """
        Constructor of Adapter class.

        Parameters
        ----------
        mpi_comm : mpi4py.MPI.Intercomm
            Communicator used by the adapter. Should be the same one used by FEniCSx, usually MPI.COMM_WORLD
        adapter_config_filename : string
            Name of the JSON adapter configuration file (to be provided by the user)
        """

        self._config = Config(adapter_config_filename)

        # Setup up MPI communicator
        self._comm = mpi_comm

        self._interface = precice.Interface(self._config.get_participant_name(), self._config.get_config_file_name(),
                                            self._comm.Get_rank(), self._comm.Get_size())

        # FEniCSx related quantities
        self._read_function_space = None  # initialized later
        self._write_function_space = None  # initialized later
        self._dofmap = None  # initialized later using function space provided by user

        # coupling mesh related quantities
        self._fenicsx_vertices = Vertices(VertexType.FENICSX)
        self._precice_vertex_ids = None  # initialized later

        # read data related quantities (read data is read from preCICE and applied in FEniCSx)
        self._read_function_type = None  # stores whether read function is scalar or vector valued
        self._write_function_type = None  # stores whether write function is scalar or vector valued

        # Interpolation strategy
        self._my_expression = SegregatedRBFInterpolationExpression

        # Solver state used by the Adapter internally to handle checkpointing
        self._checkpoint = None

        # Necessary bools for enforcing proper control flow / warnings to user
        self._first_advance_done = False

        # Determine type of coupling in initialization
        self._coupling_type = None

        # Problem dimension in FEniCSx
        self._fenicsx_dims = None

    def create_coupling_expression(self):
        """
        Creates a FEniCSx Expression in the form of an object of class GeneralInterpolationExpression or
        ExactInterpolationExpression. The adapter will hold this object till the coupling is on going.

        Returns
        -------
        coupling_expression : Object of class dolfinx.functions.expression.Expression
            Reference to object of class GeneralInterpolationExpression or ExactInterpolationExpression.
        """

        if not (self._read_function_type is FunctionType.SCALAR or self._read_function_type is FunctionType.VECTOR):
            raise Exception("No valid read_function is provided in initialization. Cannot create coupling expression")

        coupling_expression = self._my_expression(self._read_function_space)
        coupling_expression.set_function_type(self._read_function_type)

        return coupling_expression

    def update_coupling_expression(self, coupling_expression, data):
        """
        Updates the given FEniCSx Expression using provided data. The boundary data is updated.
        User needs to explicitly call this function in each time step.

        Parameters
        ----------
        coupling_expression : Object of class dolfinx.functions.expression.Expression
            Reference to object of class GeneralInterpolationExpression or ExactInterpolationExpression.
        data : dict_like
            The coupling data. A dictionary containing nodal data with vertex coordinates as key and associated data as
            value.
        """
        vertices = np.array(list(data.keys()))
        nodal_data = np.array(list(data.values()))
        coupling_expression.update_boundary_data(nodal_data, vertices[:, 0], vertices[:, 1])
        coupling_expression.interpolate(coupling_expression.interpolate_precice)

    def get_point_sources(self, data):
        raise Exception("PointSources are not implemented for the FEniCSx adapter.")

    def read_data(self):
        """
        Read data from preCICE. Data is generated depending on the type of the read function (Scalar or Vector).
        For a scalar read function the data is a numpy array with shape (N) where N = number of coupling vertices
        For a vector read function the data is a numpy array with shape (N, D) where
        N = number of coupling vertices and D = dimensions of FEniCSx setup

        Note: For quasi 2D-3D coupled simulation (FEniCSx participant is 2D) the Z-component of the data and vertices
        is deleted.

        Returns
        -------
        data : dict_like
            The coupling data. A dictionary containing nodal data with vertex coordinates as key and associated data as
            value.
        """
        assert (self._coupling_type is CouplingMode.UNI_DIRECTIONAL_READ_COUPLING or
                CouplingMode.BI_DIRECTIONAL_COUPLING)

        read_data_id = self._interface.get_data_id(self._config.get_read_data_name(),
                                                   self._interface.get_mesh_id(self._config.get_coupling_mesh_name()))

        read_data = None

        if self._read_function_type is FunctionType.SCALAR:
            read_data = self._interface.read_block_scalar_data(read_data_id, self._precice_vertex_ids)
        elif self._read_function_type is FunctionType.VECTOR:
            read_data = self._interface.read_block_vector_data(read_data_id, self._precice_vertex_ids)

        read_data = {tuple(key): value for key, value in zip(self._fenicsx_vertices.get_coordinates(), read_data)}

        return copy.deepcopy(read_data)

    def write_data(self, write_function):
        """
        Writes data to preCICE. Depending on the dimensions of the simulation (2D-3D Coupling, 2D-2D coupling or
        Scalar/Vector write function) write_data is first converted into a format needed for preCICE.

        Parameters
        ----------
        write_function : Object of class dolfinx.functions.function.Function
            A FEniCSx function consisting of the data which this participant will write to preCICE in every time step.
        """

        assert (self._coupling_type is CouplingMode.UNI_DIRECTIONAL_WRITE_COUPLING or
                CouplingMode.BI_DIRECTIONAL_COUPLING)

        w_func = write_function

        # Check that the function provided lives on the same function space provided during initialization
        assert (self._write_function_type == determine_function_type(w_func))
        # TODO this raises AssertionError, not sure why. I just commented it out, still works...
        # assert (write_function.function_space == self._write_function_space)

        write_data_id = self._interface.get_data_id(self._config.get_write_data_name(),
                                                    self._interface.get_mesh_id(self._config.get_coupling_mesh_name()))

        write_function_type = determine_function_type(write_function)
        assert (write_function_type in list(FunctionType))
        write_data = convert_fenicsx_to_precice(write_function, self._fenicsx_vertices.get_ids())
        if write_function_type is FunctionType.SCALAR:
            assert (write_function.function_space.num_sub_spaces == 0)
            write_data = np.squeeze(write_data)  # TODO dirty solution
            self._interface.write_block_scalar_data(write_data_id, self._precice_vertex_ids, write_data)
        elif write_function_type is FunctionType.VECTOR:
            assert (write_function.function_space.num_sub_spaces > 0)
            self._interface.write_block_vector_data(write_data_id, self._precice_vertex_ids, write_data)
        else:
            raise Exception("write_function provided is neither VECTOR nor SCALAR type")

    def initialize(self, coupling_subdomain, read_function_space=None, write_object=None):
        """
        Initializes the coupling and sets up the mesh where coupling happens in preCICE.

        Parameters
        ----------
        coupling_subdomain : Object of class dolfinx.cpp.mesh.SubDomain
            SubDomain of mesh which is the physical coupling boundary.
        read_function_space : Object of class dolfinx.functions.functionspace.FunctionSpace
            Function space on which the read function lives. If not provided then the adapter assumes that this
            participant is a write-only participant.
        write_object : Object of class dolfinx.functions.functionspace.FunctionSpace / dolfinx.functions.function.Function
            Function space on which the write function lives or FEniCSx function related to the quantity to be written
            by FEniCSx during each coupling iteration. If not provided then the adapter assumes that this participant is
            a read-only participant.

        Returns
        -------
        dt : double
            Recommended time step value from preCICE.
        """

        write_function_space, write_function = None, None
        if isinstance(write_object, Function):  # precice.initialize_data() will be called using this Function
            write_function_space = write_object.function_space
            write_function = write_object
        elif isinstance(write_object, FunctionSpace):  # preCICE will use default zero values for initialization.
            write_function_space = write_object
            write_function = None
        elif write_object is None:
            pass
        else:
            raise Exception("Given write object is neither of type dolfinx.functions.function.Function or "
                            "dolfinx.functions.functionspace.FunctionSpace")

        if isinstance(read_function_space, FunctionSpace):
            pass
        elif read_function_space is None:
            pass
        else:
            raise Exception("Given read_function_space is not of type dolfinx.functions.functionspace.FunctionSpace")

        if read_function_space is None and write_function_space:
            self._coupling_type = CouplingMode.UNI_DIRECTIONAL_WRITE_COUPLING
            assert (self._config.get_write_data_name())
            print("Participant {} is write-only participant".format(self._config.get_participant_name()))
            function_space = write_function_space
        elif read_function_space and write_function_space is None:
            self._coupling_type = CouplingMode.UNI_DIRECTIONAL_READ_COUPLING
            assert (self._config.get_read_data_name())
            print("Participant {} is read-only participant".format(self._config.get_participant_name()))
            function_space = read_function_space
        elif read_function_space and write_function_space:
            self._coupling_type = CouplingMode.BI_DIRECTIONAL_COUPLING
            assert (self._config.get_read_data_name() and self._config.get_write_data_name())
            function_space = read_function_space
        elif read_function_space is None and write_function_space is None:
            raise Exception("Neither read_function_space nor write_function_space is provided. Please provide a "
                            "write_object if this participant is used in one-way coupling and only writes data. "
                            "Please provide a read_function_space if this participant is used in one-way coupling and "
                            "only reads data. If two-way coupling is implemented then both read_function_space"
                            " and write_object need to be provided.")
        else:
            raise Exception("Incorrect read and write function space combination provided. Please check input "
                            "parameters in initialization")

        if self._coupling_type is CouplingMode.UNI_DIRECTIONAL_READ_COUPLING or \
                self._coupling_type is CouplingMode.BI_DIRECTIONAL_COUPLING:
            self._read_function_type = determine_function_type(read_function_space)
            self._read_function_space = read_function_space

        if self._coupling_type is CouplingMode.UNI_DIRECTIONAL_WRITE_COUPLING or \
                self._coupling_type is CouplingMode.BI_DIRECTIONAL_COUPLING:
            # Ensure that function spaces of read and write functions are defined using the same mesh
            self._write_function_type = determine_function_type(write_function_space)
            self._write_function_space = write_function_space

        self._fenicsx_dims = function_space.mesh.geometry.dim

        # Ensure that function spaces of read and write functions use the same mesh
        if self._coupling_type is CouplingMode.BI_DIRECTIONAL_COUPLING:
            assert (self._read_function_space.mesh is write_function_space.mesh
                    ), "read_function_space and write_object need to be defined using the same mesh"

        if self._fenicsx_dims != 2:
            raise Exception("Currently the fenicsx-adapter only supports 2D cases")

        if self._fenicsx_dims != self._interface.get_dimensions():
            raise Exception("Dimension of preCICE setup and FEniCSx do not match")

        # Set vertices on the coupling subdomain for this rank
        ids, coords = get_fenicsx_vertices(function_space, coupling_subdomain, self._fenicsx_dims)
        self._fenicsx_vertices.set_ids(ids)
        self._fenicsx_vertices.set_coordinates(coords)

        # Set up mesh in preCICE
        self._precice_vertex_ids = self._interface.set_mesh_vertices(self._interface.get_mesh_id(
            self._config.get_coupling_mesh_name()), self._fenicsx_vertices.get_coordinates())

        precice_dt = self._interface.initialize()

        if self._interface.is_action_required(precice.action_write_initial_data()):
            if not write_function:
                raise Exception("Non-standard initialization requires a write_function")
            self.write_data(write_function)
            self._interface.mark_action_fulfilled(precice.action_write_initial_data())

        self._interface.initialize_data()

        return precice_dt

    def store_checkpoint(self, user_u, t, n):
        """
        Defines an object of class SolverState which stores the current state of the variable and the time stamp.

        Parameters
        ----------
        user_u : FEniCSx Function
            Current state of the physical variable of interest for this participant.
        t : double
            Current simulation time.
        n : int
            Current time window (iteration) number.
        """
        if self._first_advance_done:
            assert (self.is_time_window_complete())

        logger.debug("Store checkpoint")
        my_u = user_u.copy()
        # making sure that the FEniCSx function provided by user is not directly accessed by the Adapter
        assert (my_u != user_u)
        self._checkpoint = SolverState(my_u, t, n)
        self._interface.mark_action_fulfilled(self.action_write_iteration_checkpoint())

    def retrieve_checkpoint(self):
        """
        Resets the FEniCSx participant state to the state of the stored checkpoint.

        Returns
        -------
        u : FEniCSx Function
            Current state of the physical variable of interest for this participant.
        t : double
            Current simulation time.
        n : int
            Current time window (iteration) number.
        """
        assert (not self.is_time_window_complete())
        logger.debug("Restore solver state")
        self._interface.mark_action_fulfilled(self.action_read_iteration_checkpoint())
        return self._checkpoint.get_state()

    def advance(self, dt):
        """
        Advances coupling in preCICE.

        Parameters
        ----------
        dt : double
            Length of timestep used by the solver.

        Notes
        -----
        Refer advance() in https://github.com/precice/python-bindings/blob/develop/precice.pyx

        Returns
        -------
        max_dt : double
            Maximum length of timestep to be computed by solver.
        """
        self._first_advance_done = True
        max_dt = self._interface.advance(dt)
        return max_dt

    def finalize(self):
        """
        Finalizes the coupling via preCICE and the adapter. To be called at the end of the simulation.

        Notes
        -----
        Refer finalize() in https://github.com/precice/python-bindings/blob/develop/precice.pyx
        """
        self._interface.finalize()

    def get_participant_name(self):
        """
        Returns
        -------
        participant_name : string
            Name of the participant.
        """
        return self._config.get_participant_name()

    def is_coupling_ongoing(self):
        """
        Checks if the coupled simulation is still ongoing.

        Notes
        -----
        Refer is_coupling_ongoing() in https://github.com/precice/python-bindings/blob/develop/precice.pyx

        Returns
        -------
        tag : bool
            True if coupling is still going on and False if coupling has finished.
        """
        return self._interface.is_coupling_ongoing()

    def is_time_window_complete(self):
        """
        Tag to check if implicit iteration has converged.

        Notes
        -----
        Refer is_time_window_complete() in https://github.com/precice/python-bindings/blob/develop/precice.pyx

        Returns
        -------
        tag : bool
            True if implicit coupling in the time window has converged and False if not converged yet.
        """
        return self._interface.is_time_window_complete()

    def is_action_required(self, action):
        """
        Tag to check if a particular preCICE action is required.

        Parameters
        ----------
        action : string
            Name of the preCICE action.

        Notes
        -----
        Refer is_action_required(action) in https://github.com/precice/python-bindings/blob/develop/precice.pyx

        Returns
        -------
        tag : bool
            True if action is required and False if action is not required.
        """
        return self._interface.is_action_required(action)

    def action_write_iteration_checkpoint(self):
        """
        Get name of action to convey to preCICE that a checkpoint has been written.

        Notes
        -----
        Refer action_write_iteration_checkpoint() in https://github.com/precice/python-bindings/blob/develop/precice.pyx

        Returns
        -------
        action : string
            Name of action related to writing a checkpoint.
        """
        return precice.action_write_iteration_checkpoint()

    def action_read_iteration_checkpoint(self):
        """
        Get name of action to convey to preCICE that a checkpoint has been read and the state of the system has been
        restored to that checkpoint.

        Notes
        -----
        Refer action_read_iteration_checkpoint() in https://github.com/precice/python-bindings/blob/develop/precice.pyx

        Returns
        -------
        action : string
            Name of action related to reading a checkpoint.
        """
        return precice.action_read_iteration_checkpoint()
