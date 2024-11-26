import copy
import logging
from typing import List, NamedTuple

import dolfinx as dfx
import numpy as np
import precice

from .config import Config
from .solverstate import SolverState

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class Dimensions(NamedTuple):
    interface: int
    mesh: int


class Adapter:
    """This adapter class provides an interface to the preCICE v3 coupling library for setting up a
    coupling case with FEniCSx as a participant in 2D and 3D problems.

    The coupling is achieved using the FunctionSpace degrees of freedom (DOFs) on the interface.
    Data interchange between participants (read/write) is performed by accessing the values on
    the interface DOFs. This approach allows us to leverage the full range of FEniCSx functionalities
    while ensuring seamless communication with other participants in the coupling process.

    """

    def __init__(self, mpi_comm, adapter_config_filename="precice-adapter-config.json"):
        """Constructor of Adapter class.

        Parameters
        ----------
        mpi_comm : mpi4py.MPI.Intercomm
            Communicator used by the adapter. Should be the same one used by FEniCSx, usually MPI.COMM_WORLD
        adapter_config_filename : string
            Name of the JSON adapter configuration file (to be provided by the user)
        """
        self._comm = mpi_comm
        self._config = Config(adapter_config_filename)

        self._interface = precice.Participant(
            self._config.get_participant_name(),
            self._config.get_config_file_name(),
            self._comm.Get_rank(),
            self._comm.Get_size(),
        )

        # coupling mesh related quantities
        self._fenicsx_vertices = None
        self._precice_vertex_ids = None

        # problem Function Space
        self._function_space = None

        # Solver state used by the Adapter internally to handle checkpointing
        self._checkpoint = None

        # Necessary bools for enforcing proper control flow / warnings to user
        self._first_advance_done = False

        # Determine type of coupling in initialization
        self._coupling_type = None

        # Problem dimension in FEniCSx
        self._fenicsx_dims = None

    def interpolation_points_in_vector_space(self):
        """Determine interpolation points in the vector space.

        Returns
        -------
        tuple
            Unrolled degrees of freedom and their coordinates.
        """
        V = self._function_space
        bs = V.dofmap.bs

        fs_coords = V.tabulate_dof_coordinates()
        fs_coords = fs_coords[:, : self.dimensions.mesh]
        boundary_dofs = dfx.fem.locate_dofs_topological(V, self.dimensions.mesh - 1, self._tags)
        unrolled_dofs = np.empty(len(boundary_dofs) * bs, dtype=np.int32)

        for i, dof in enumerate(boundary_dofs):
            for b in range(bs):
                unrolled_dofs[i * bs + b] = dof * bs + b

        return unrolled_dofs.reshape(-1, bs), fs_coords[boundary_dofs]

    def read_data(self):
        """Read data from preCICE.

        Incoming data is a ndarray where the shape of the array depends on the dimensions of the problem.
        For scalar problems, this will be a 1D array (vector), while for vector problems,
        it will be an Mx2 array (in 2D) or an Mx3 array (in 3D), where M is the number of interface nodes.

        Returns
        -------
        np.ndarray
            The incoming data containing nodal data ordered according to _fenicsx_vertices
        """
        mesh_name = self._config.get_coupling_mesh_name()
        data_name = self._config.get_read_data_name()
        # For more information about readDara see and and time start here:
        # https://precice.org/couple-your-code-porting-v2-3.html#add-relativereadtime-for-all-read-data-calls
        read_data = self._interface.read_data(
            mesh_name, data_name, self._precice_vertex_ids, self.dt
        )
        return copy.deepcopy(read_data)

    def write_data(self, write_function):
        """Writes data to preCICE. Depending on the dimensions of the simulation.
        For scalar problems, this will be a 1D array (vector), while for vector problems,
        it will be an Mx2 array (in 2D) or an Mx3 array (in 3D), where M is the number of interface nodes.


        Parameters
        ----------
        write_function : dolfinx.fem.Function
            A FEniCSx function consisting of the data which this participant will write to preCICE
            in every time step.
        """
        mesh_name = self._config.get_coupling_mesh_name()
        write_data_name = self._config.get_write_data_name()
        write_data = write_function.x.array[self.interface_dof]
        self._interface.write_data(mesh_name, write_data_name, self._precice_vertex_ids, write_data)

    def initialize(self, coupling_subdomain, read_function_space=None, write_object=None):
        """Initializes the coupling and sets up the mesh where coupling happens in preCICE.

        Parameters
        ----------
        coupling_subdomain : List
            Indices of entities representing the coupling interface normally face sets tags.
        read_function_space : dolfinx.fem.FunctionSpace
            Function space on which the read function lives. If not provided then the adapter assumes that this
            participant is a write-only participant.
        write_object : dolfinx.fem.Function
            FEniCSx function related to the quantity to be written
            by FEniCSx during each coupling iteration. If not provided then the adapter assumes that this participant is
            a read-only participant.

        Returns
        -------
        dt : double
            Recommended time step value from preCICE.
        """
        self._domain = read_function_space.mesh
        self._function_space = read_function_space
        self._tags = coupling_subdomain

        self._interface_dof, self._interface_dof_coords = (
            self.interpolation_points_in_vector_space()
        )
        self._fenicsx_vertices = list(zip(self._interface_dof, self._interface_dof_coords))

        self._precice_vertex_ids = self._interface.set_mesh_vertices(
            self._config.get_coupling_mesh_name(), self._interface_dof_coords[:, : self.dimensions.interface]
        )

        if self._interface.requires_initial_data():
            self._interface.write_data(
                self._config.get_coupling_mesh_name(),
                self._config.get_write_data_name(),
                self._precice_vertex_ids,
                np.zeros(self._interface_dof_coords.shape),
            )
        self._interface.initialize()
        return self.dt

    def store_checkpoint(self, states: List):
        """Defines an object of class SolverState which stores the current states of the variable and the time stamp."""
        if self._first_advance_done:
            assert self.is_time_window_complete()
        logger.debug("Store checkpoint")
        self._checkpoint = SolverState(states)

    def retrieve_checkpoint(self):
        """
        Resets the FEniCSx participant state to the state of the stored checkpoint.

        Returns
        -------
        tuple
            The stored checkpoint state (u, v, a, t).
        """
        assert not self.is_time_window_complete()
        logger.debug("Restore solver state")
        return self._checkpoint.get_state()

    def advance(self, dt: float):
        """Advances coupling in preCICE.

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
        """Tag to check if implicit iteration has converged.

        Notes:
        -----
        Refer is_time_window_complete() in https://github.com/precice/python-bindings/blob/develop/precice.pyx

        Returns:
        -------
        tag : bool
            True if implicit coupling in the time window has converged and False if not converged yet.
        """
        return self._interface.is_time_window_complete()

    def requires_reading_checkpoint(self):
        """Checks if reading a checkpoint is required.

        Returns:
        -------
        bool
            True if reading a checkpoint is required, False otherwise.
        """
        return self._interface.requires_reading_checkpoint()

    def requires_writing_checkpoint(self):
        """Checks if writing a checkpoint is required.

        Returns:
        -------
        bool
            True if writing a checkpoint is required, False otherwise.
        """
        return self._interface.requires_writing_checkpoint()

    @property
    def interface_dof(self):
        """Returns the interface degrees of freedom."""
        return self._interface_dof

    @property
    def interface_coordinates(self):
        """Returns the interface coordinates."""
        return self._interface_dof_coords

    @property
    def precice(self):
        """Returns the preCICE interface object."""
        return self._interface

    @property
    def dimensions(self) -> Dimensions:
        """Returns the dimensions of the mesh and the interface."""
        return Dimensions(
            self._interface.get_mesh_dimensions(self._config.get_coupling_mesh_name()), self._domain.geometry.dim
        )

    @property
    def dt(self):
        """Returns the maximum time step size allowed by preCICE."""
        return self._interface.get_max_time_step_size()
