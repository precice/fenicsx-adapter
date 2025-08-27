"""
FEniCSx - preCICE Adapter. API to help users couple FEniCS with other solvers using the preCICE library.
:raise ImportError: if PRECICE_ROOT is not defined
"""
import numpy as np
from .config import Config
import logging
import precice
from .adapter_core import FunctionType, determine_function_type, get_fenicsx_vertices, CouplingMode, Vertices, convert_fenicsx_to_precice
from .expression_core import SegregatedRBFInterpolationExpression
from .solverstate import SolverState
from .coupling_mesh import CouplingMesh
from dolfinx import fem
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

        self._participant = precice.Participant(
            self._config.get_participant_name(),
            self._config.get_config_file_name(),
            self._comm.Get_rank(),
            self._comm.Get_size()
        )

        # FEniCSx related quantities
        self._read_function_spaces = {}  # initialized later
        self._write_function_spaces = {}  # initialized later
        self._dofmap = None  # initialized later using function space provided by user

        # coupling mesh related quantities
        self._fenicsx_vertices = {}
        self._precice_vertex_ids = {}  # initialized later

        # read data related quantities (read data is read from preCICE and applied in FEniCSx)
        self._read_function_types = {}  # stores whether read function is scalar or vector valued
        self._write_function_types = {}  # stores whether write function is scalar or vector valued

        # Interpolation strategy
        self._my_expression = SegregatedRBFInterpolationExpression

        # Solver state used by the Adapter internally to handle checkpointing
        self._checkpoint = None

        # Necessary bools for enforcing proper control flow / warnings to user
        self._first_advance_done = False

        # Determine type of coupling in initialization
        self._coupling_types = {}

        # Problem dimension in FEniCSx
        self._fenicsx_dims = None
        self._empty_rank = True

    def create_coupling_expression(self, mesh_name):
        """
        Creates a FEniCSx Expression in the form of an object of class GeneralInterpolationExpression or
        ExactInterpolationExpression. The adapter will hold this object till the coupling is on going.

        Parameters
        ----------
        mesh_name:
            Specifies for which field a coupling expression shall be created

        Returns
        -------
        coupling_expression : Object of class dolfinx.functions.expression.Expression
            Reference to object of class GeneralInterpolationExpression or ExactInterpolationExpression.
        """

        if not (self._read_function_types[mesh_name]
                is FunctionType.SCALAR or self._read_function_types[mesh_name] is FunctionType.VECTOR):
            raise Exception("No valid read_function is provided in initialization. Cannot create coupling expression")

        coupling_expression = self._my_expression(
            self._read_function_spaces[mesh_name],
            self._read_function_types[mesh_name])

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
            The coupling data. A dictionary containing the values of the vertex coordinates as key and associated data as
            value.
        """
        vertices = np.array(list(data.keys()))
        nodal_data = np.array(list(data.values()))
        coupling_expression.update_boundary_data(nodal_data, vertices[:, 0], vertices[:, 1])

    def get_point_sources(self, data):
        raise Exception("PointSources are not implemented for the FEniCSx adapter.")

    def read_data(self, mesh_name, read_data_name, dt):
        """
        Read data from preCICE. Data is generated depending on the type of the read function (Scalar or Vector).
        For a scalar read function the data is a numpy array with shape (N) where N = number of coupling vertices
        For a vector read function the data is a numpy array with shape (N, D) where
        N = number of coupling vertices and D = dimensions of FEniCSx setup

        Note: For quasi 2D-3D coupled simulation (FEniCSx participant is 2D) the Z-component of the data and vertices
        is deleted.

        Parameters
        ----------
        dt : offset within time window
        mesh_name:
            Specifies for which mesh the data shall be read

        Returns
        -------
        data : dict_like
            The coupling data. A dictionary containing nodal data with vertex coordinates as key and associated data as
            value.
        """
        assert (self._coupling_types[mesh_name] is CouplingMode.UNI_DIRECTIONAL_READ_COUPLING or
                CouplingMode.BI_DIRECTIONAL_COUPLING)

        read_data = None

        if not self._empty_rank:
            read_data = self._participant.read_data(
                mesh_name,
                read_data_name,
                self._precice_vertex_ids[mesh_name],
                dt
            )
            read_data = {
                tuple(key): value for key,
                value in zip(
                    self._fenicsx_vertices[mesh_name].get_coordinates(),
                    read_data)}

        else:
            pass

        return copy.deepcopy(read_data)

    def read_data_at_coordinates(self, mesh_name, read_data_name, coordinates, dt):
        """
        Read data from preCICE at specified coordinates. This function uses the just-in-time mapping of preCICE.

        Parameters
        ----------
            mesh_name: Specifies for which mesh the data is read
            read_data_name: Specifies the data name of the mesh
            coordinates: List of coordinates where preCICE reads the data
            dt: offset within time window

        Returns
        -------
            dict: Returns a dict of coordinates as key and read data as value
        """
        read_data = None

        if not self._empty_rank:
            read_data = self._participant.map_and_read_data(
                mesh_name,
                read_data_name,
                coordinates,
                dt
            )

            read_data = {
                tuple(key): value for key, value in zip(coordinates, read_data)
            }

        else:
            pass

        return copy.deepcopy(read_data)

    def write_data(self, mesh_name, write_data_name, write_function):
        """
        Writes data to preCICE. Depending on the dimensions of the simulation (2D-3D Coupling, 2D-2D coupling or
        Scalar/Vector write function) write_data is first converted into a format needed for preCICE.

        Parameters
        ----------
        write_function : Object of class dolfinx.functions.function.Function
            A FEniCSx function consisting of the data which this participant will write to preCICE in every time step.
        mesh_name :
            Specifies the mesh from which the data is written
        """

        assert (self._coupling_types[mesh_name] is CouplingMode.UNI_DIRECTIONAL_WRITE_COUPLING or
                CouplingMode.BI_DIRECTIONAL_COUPLING)

        w_func = write_function.copy()

        # Check that the function provided lives on the same function space provided during initialization
        assert self._write_function_types[mesh_name] == determine_function_type(w_func)
        assert write_function.function_space == self._write_function_spaces[mesh_name]

        write_function_type = determine_function_type(write_function)
        assert write_function_type in list(FunctionType)
        write_data = convert_fenicsx_to_precice(write_function, self._fenicsx_vertices[mesh_name].get_coordinates())
        self._participant.write_data(
            mesh_name,
            write_data_name,
            self._precice_vertex_ids[mesh_name],
            write_data
        )

    def write_data_at_coordinates(self, mesh_name, write_data_name, coordinates, write_function):
        """
        Writes data to preCICE at the given coordinates with the just-in-time mapping of preCICE.

        Parameters
        ----------
            mesh_name: Name of the mesh that is written to
            write_data_name: Name of the data field that is written
            coordinates: A list of coordinates that defines where write_function is evaluated
            write_function: The function whose values at the given coordinates will be written
        """
        write_function_type = determine_function_type(write_function)
        assert write_function_type in list(FunctionType)
        write_data = convert_fenicsx_to_precice(write_function, coordinates)
        self._participant.write_and_map_data(mesh_name, write_data_name, write_data)

    def validate_function_space(self, function_objects, mesh_name, checkIfFunction):
        """_summary_

        Args:
            functions: a dict of Function and FunctionSpace
        ------
        Returns: the function space that is equal for all function_objects or raises an exception
        """

        # get first and extract function space
        function_space = None
        if function_objects is None:
            return None
        else:
            f = list(function_objects.values())[0]
            if checkIfFunction and isinstance(f, fem.Function):
                function_space = f.function_space
            # preCICE will use default zero values for initialization.
            elif isinstance(f, fem.FunctionSpace):
                function_space = f
            elif f is None:
                pass
            else:
                if checkIfFunction:
                    raise Exception("A given object in {} is neither of type dolfinx.functions.function.Function or "
                                    "dolfinx.functions.functionspace.FunctionSpace".format(mesh_name))
                else:
                    raise Exception(
                        "A given object of {} is not of type dolfinx.functions.functionspace.FunctionSpace".format(mesh_name))

        if function_space is None and len(function_objects) > 1:
            raise Exception("Invalid argument provided: At least one write function space is defined as None,"
                            "but the number of given write functions was {}."
                            "If no write function want to be used on mesh {}, set"
                            "the write function object array to None instead!".format(len(function_objects, mesh_name)))

        for fun in function_objects:
            func = function_objects[fun]
            if checkIfFunction and isinstance(func, fem.Function):
                assert func.function_space == function_space
            elif isinstance(func, fem.FunctionSpace):
                assert func == function_space
            else:
                if checkIfFunction:
                    raise Exception("A given object in {} is neither of type dolfinx.functions.function.Function or "
                                    "dolfinx.functions.functionspace.FunctionSpace".format(mesh_name))
                else:
                    raise Exception(
                        "A given object of {} is not of type dolfinx.functions.functionspace.FunctionSpace".format(mesh_name))

        return function_space

    def initialize(self, coupling_meshes: list[CouplingMesh]):
        """
        Initializes the coupling and sets up the mesh where coupling happens in preCICE.

        Parameters
        ----------
        coupling_meshes: A list of coupling meshes of the class CouplingMesh.

        Returns
        -------
        dt : double
            Recommended time step value from preCICE.
        """

        for c_mesh in coupling_meshes:
            mesh_name = c_mesh.get_name()
            # check if all function spaces (read amd write are equal each) and get the function space
            write_function_space = self.validate_function_space(
                c_mesh.get_write_fields(), mesh_name, checkIfFunction=True)
            read_function_space = self.validate_function_space(
                c_mesh.get_read_fields(), mesh_name, checkIfFunction=False)

            if read_function_space is None and write_function_space:
                self._coupling_types[mesh_name] = CouplingMode.UNI_DIRECTIONAL_WRITE_COUPLING
                assert self._config.get_write_data_names(mesh_name)  # error if empty array or None
                print("Participant {} is write-only participant".format(self._config.get_participant_name()))
                function_space = write_function_space
            elif read_function_space and write_function_space is None:
                self._coupling_types[mesh_name] = CouplingMode.UNI_DIRECTIONAL_READ_COUPLING
                assert self._config.get_read_data_names(mesh_name)  # error if empty array or None
                print("Participant {} is read-only participant".format(self._config.get_participant_name()))
                function_space = read_function_space
            elif read_function_space and write_function_space:
                self._coupling_types[mesh_name] = CouplingMode.BI_DIRECTIONAL_COUPLING
                assert self._config.get_read_data_names(mesh_name) and self._config.get_write_data_names(mesh_name)
                function_space = read_function_space
            elif read_function_space is None and write_function_space is None:
                raise Exception(
                    "Neither read_function_space nor write_function_space for {} is provided."
                    "Please provide a write_object if this participant is used in one-way coupling"
                    "and only writes data. Please provide a read_function_space if this participant"
                    "is used in one-way coupling and only reads data. If two-way coupling is"
                    "implemented then both read_function_space and write_object need to be provided.".format(mesh_name))
            else:
                raise Exception(
                    "Incorrect read and write function space combination provided for {}. Please check input "
                    "parameters in initialization".format(mesh_name))

            coupling_type = self._coupling_types[mesh_name]
            if coupling_type is CouplingMode.UNI_DIRECTIONAL_READ_COUPLING or \
                    coupling_type is CouplingMode.BI_DIRECTIONAL_COUPLING:
                self._read_function_types[mesh_name] = determine_function_type(read_function_space)
                self._read_function_spaces[mesh_name] = read_function_space

            if coupling_type is CouplingMode.UNI_DIRECTIONAL_WRITE_COUPLING or \
                    coupling_type is CouplingMode.BI_DIRECTIONAL_COUPLING:
                # Ensure that function spaces of read and write functions are defined using the same mesh
                self._write_function_types[mesh_name] = determine_function_type(write_function_space)
                self._write_function_spaces[mesh_name] = write_function_space

            # Set vertices on the coupling subdomain for this rank (assumed to be
            # equal across all fields and meshes that will be coupled)
            self._fenicsx_dims = function_space.mesh.geometry.dim
            # returns 3d coordinates (necessary later for writing the data!)
            ids, coords = get_fenicsx_vertices(function_space, c_mesh.get_coupling_boundary(), self._fenicsx_dims)
            # this isnt a problem in update_coupling_expression, because in this function
            # , the two first dimensions are extracted. Exactly what we want!
            self._fenicsx_vertices[mesh_name] = Vertices()
            self._fenicsx_vertices[mesh_name].set_ids(ids)
            self._fenicsx_vertices[mesh_name].set_coordinates(coords)

            # Set up mesh in preCICE
            self._precice_vertex_ids[mesh_name] = self._participant.set_mesh_vertices(
                mesh_name, self._fenicsx_vertices[mesh_name].get_coordinates()[
                    :, :2])  # give preCICE only 2D coordinates

            if self._fenicsx_vertices[mesh_name].get_ids().size > 0:
                self._empty_rank = False
            else:
                print("Rank {} has no part of coupling boundary.".format(self._comm.Get_rank()))

            # Ensure that function spaces of read and write functions use the same mesh
            if coupling_type is CouplingMode.BI_DIRECTIONAL_COUPLING:
                assert (self._read_function_spaces[mesh_name].mesh is write_function_space.mesh
                        ), "read_function_space and write_object need to be defined using the same mesh"

            if self._fenicsx_dims != 2:
                raise Exception("Currently the fenicsx-adapter only supports 2D cases")

            if self._fenicsx_dims != self._participant.get_mesh_dimensions(mesh_name):
                raise Exception("Dimension of preCICE setup and FEniCSx do not match")

            if self._participant.requires_initial_data():
                for write_data_name in c_mesh.get_write_fields():
                    write_function = c_mesh.get_write_fields()[write_data_name]
                    if not isinstance(write_function, fem.Function):
                        raise Exception(
                            "preCICE requires you to write initial data. Please provide a write_function to initialize(...)")
                    self.write_data(mesh_name, write_data_name, write_function)

        self._participant.initialize()

    def store_checkpoint(self, payload, t, n):
        """
        Defines an object of class SolverState which stores the current state of the variable and the time stamp.

        Parameters
        ----------
        payload : FEniCSx Function
            Current state of the physical variable of interest for this participant.
        t : double
            Current simulation time.
        n : int
            Current time window (iteration) number.
        """
        if self._first_advance_done:
            assert (self.is_time_window_complete())

        logger.debug("Store checkpoint")
        my_u = payload.copy()
        # making sure that the FEniCSx function provided by user is not directly accessed by the Adapter
        assert (my_u != payload)
        self._checkpoint = SolverState(my_u, t, n)

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
        max_dt = self._participant.advance(dt)
        return max_dt

    def finalize(self):
        """
        Finalizes the coupling via preCICE and the adapter. To be called at the end of the simulation.

        Notes
        -----
        Refer finalize() in https://github.com/precice/python-bindings/blob/develop/precice.pyx
        """
        self._participant.finalize()

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
        return self._participant.is_coupling_ongoing()

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
        return self._participant.is_time_window_complete()

    def get_max_time_step_size(self):
        return self._participant.get_max_time_step_size()

    def requires_writing_checkpoint(self):
        return self._participant.requires_writing_checkpoint()

    def requires_reading_checkpoint(self):
        return self._participant.requires_reading_checkpoint()

    def set_mesh_access_region(self, mesh_name, access_region):
        """
        Defines the access region for the just-in-time mapping

        Parameters
        ----------
            mesh_name: The name of the mesh for which the access region is defined
            access_region: A list of tuples defining the access region. Expects a list of 2 points.
        """
        if len(access_region) != 2:  # should also be fine for 3d cases
            raise Exception(
                "Two points to define the access region were expected but {} were given.".format(
                    len(access_region)))
        if len(access_region[0]) != len(access_region[1]):
            raise Exception("The dimensions of the given points do not coincide.".format(len(access_region)))
        ar = [0] * (len(access_region[0]) * 2)
        for idx, v in enumerate(access_region[0]):
            ar[idx * 2] = v
        for idx, v in enumerate(access_region[1]):
            ar[1 + idx * 2] = v

        self._participant.set_mesh_access_region(mesh_name, ar)
