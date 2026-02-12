"""
This module consists of helper functions used in the Adapter class. Names of the functions are self explanatory
"""

from dolfinx import fem, geometry, mesh as msh
import numpy as np
from enum import Enum
import logging
import copy
from numbers import Number
from mpi4py import MPI

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def quantize_to_chunks(coords, digit_cutoff, chunk_digits=10):
    """
    Quantize coordinates to fixed-point with digit_cutoff decimals,
    represented as int64 chunks of size chunk_digits.

    Parameters
    ----------
    coords: numpy array
        Coordinate array to be rounded
    digit_cutoff: int
        Specifies at which decimal place the coordinates are rounded
    chunk_digits: int
        Chunk size

    Returns
    -------
    numpy array:
        ndarray of shape (N, dim, n_chunks) with dtype int64
    """
    N, dim = coords.shape

    base = 10 ** chunk_digits
    max_abs = np.max(np.abs(coords))
    max_int_digits = int(np.ceil(np.log10(max_abs + 1))) if int(max_abs) > 0 else 0
    total_digits = max_int_digits + digit_cutoff
    n_chunks = (total_digits + chunk_digits - 1) // chunk_digits

    # scale relevant region into int range (is still float)
    scale = 10.0 ** digit_cutoff
    q = np.floor(coords * scale + 0.5)

    # store chunks
    chunks = np.zeros((N, dim, n_chunks), dtype=np.int64)

    for k in range(n_chunks):
        chunks[..., k] = q % base
        q //= base
    # chunks[..., -1] = q

    return chunks


def unique_by_chunks(chunks):
    """
    Perform uniqueness over (dim × n_chunks) int64 keys.

    Parameters
    ----------
    chunks: numpy array
        ndarray of shape (N, dim, n_chunks) with dtype int64

    Returns
    -------
    numpy array:
        indices of unique keys
    """
    N, dim, n_chunks = chunks.shape
    flat = chunks.reshape(N, dim * n_chunks)

    dtype = np.dtype([
        (f'f{i}', np.int64) for i in range(flat.shape[1])
    ])
    structured = flat.view(dtype).reshape(N)
    _, idx = np.unique(structured, return_index=True)
    # is equiv to: idx = np.unique(flat, axis=0, return_index=True)[1]
    # should have better performance this way

    return idx


def round_unique_coordinates(coords, digit_cutoff):
    """
    round the coordinates (coords) to avoid close points (i.e. those that are equal
    until the 8 decimal place) to be able to use rbf mapping

    Parameters
    ----------
    coords: numpy array
        Coordinate array to be rounded
    digit_cutoff: int
        Specifies at which decimal place the coordinates are rounded


    Returns
    -------
    numpy array:
        array of rounded and unique coordinates
    """
    chunks = quantize_to_chunks(coords, digit_cutoff, chunk_digits=10)
    idx = unique_by_chunks(chunks)

    coords_unique = coords[idx]
    return coords_unique


class Vertices:
    """
    Vertices class provides a generic skeleton for vertices. A set of vertices has a set of IDs and
    coordinates as defined in FEniCSx.
    """

    def __init__(self):
        self._ids = None
        self._coordinates = None

    def set_ids(self, ids):
        self._ids = ids

    def set_coordinates(self, coords):
        self._coordinates = coords

    def get_ids(self):
        return self._ids

    def get_coordinates(self):
        return copy.deepcopy(self._coordinates)


class FunctionType(Enum):
    """
    Defines scalar- and vector-valued function.
    Used in assertions to check if a FEniCSx function is scalar or vector.
    """
    SCALAR = 0  # scalar valued function
    VECTOR = 1  # vector valued function


class CouplingMode(Enum):
    """
    Defines the type of coupling being used.
    Options are: Bi-directional coupling, Uni-directional Write Coupling, Uni-directional Read Coupling
    Used in assertions to check which type of coupling is done
    """
    BI_DIRECTIONAL_COUPLING = 4
    UNI_DIRECTIONAL_WRITE_COUPLING = 5
    UNI_DIRECTIONAL_READ_COUPLING = 6


class CouplingBoundaryInterpolation(Enum):
    ADAPTER = 1
    USER = 2


def determine_function_type(input_obj):
    """
    Determines if the function is scalar- or vector-valued based on rank evaluation.

    Parameters
    ----------
    input_obj :
        A FEniCSx function.

    Returns
    -------
    tag : bool
        0 if input_function is SCALAR and 1 if input_function is VECTOR.
    """
    if isinstance(input_obj, fem.FunctionSpace):  # scalar-valued functions have rank 0 is FEniCSx
        if input_obj.num_sub_spaces == 0:
            return FunctionType.SCALAR
        elif input_obj.num_sub_spaces >= 1:
            return FunctionType.VECTOR
    elif isinstance(input_obj, fem.Function):
        input_fspace = input_obj.function_space
        if input_fspace.num_sub_spaces == 0:
            return FunctionType.SCALAR
        elif input_fspace.num_sub_spaces >= 1:
            return FunctionType.VECTOR
        else:
            raise Exception("Error determining type of given dolfin Function")
    else:
        raise Exception("Error determining type of given dolfin FunctionSpace")


def convert_fenicsx_to_precice(fenicsx_function, local_coords, digit_cutoff):
    """
    Converts data of type dolfinx.Function into Numpy array for all x and y coordinates on the boundary.

    Parameters
    ----------
    fenicsx_function : FEniCSx function
        A FEniCSx function referring to a physical variable in the problem.
    local_coords: numpy array
        Array of local coordinates of vertices on the coupling interface and owned by this rank.

    Returns
    -------
    precice_data : array_like
        Array of FEniCSx function values at each point on the boundary.
    """

    if not isinstance(fenicsx_function, fem.Function):
        raise Exception("Cannot handle data type {}".format(type(fenicsx_function)))

    mesh = fenicsx_function.function_space.mesh

    # this evaluation is a bit annoying, see:
    # https://github.com/FEniCS/dolfinx/blob/main/python/test/unit/fem/test_function.py#L63

    # for fast function evaluation
    # TODO: as long as the domain didn't change, we could store that tree somewhere
    bb_tree = geometry.bb_tree(mesh, mesh.geometry.dim)

    cells = []
    points = []

    # Find cells whose bounding-box collide with the the points
    cell_candidates = geometry.compute_collisions_points(bb_tree, local_coords)
    # Choose one of the cells that contains the point
    colliding_cells = geometry.compute_colliding_cells(mesh, cell_candidates, local_coords)

    # generate the midpoint tree from the already computed cell candidates
    # (-> original cell owner is in the list anyways)
    all_cells = np.unique(colliding_cells.array)
    midpoint_tree = geometry.create_midpoint_tree(mesh, mesh.topology.dim, all_cells)

    for i, point in enumerate(local_coords):
        if len(colliding_cells.links(i)) > 0:
            points.append(point)
            cells.append(colliding_cells.links(i)[0])
        else:
            # point is outside domain, probably because of rounding
            closest_cell_idx = geometry.compute_closest_entity(bb_tree, midpoint_tree, mesh, point)[0]
            # change point such that it is in the function domain
            # -> find midpoint to closest cell
            closest_midpoint_cell = msh.compute_midpoints(mesh, mesh.topology.dim, np.array([closest_cell_idx]))[0]
            # in each direction, digit_cutoff defines on how much the point has been shifted
            # -> move it in the direction of midpoint cell by at max. this amount, so, in each direction maximal 1e-digit_cutoff
            direction = closest_midpoint_cell - point
            direction = np.sign(direction) * (10**(-digit_cutoff))
            moved_point = point + direction

            cells.append(closest_cell_idx)
            points.append(moved_point)

    precice_data = fenicsx_function.eval(points, cells)
    return np.array(precice_data)


def get_fenicsx_interpolation_points(
        function_space: fem.FunctionSpace,
        coupling_boundary,
        comm: MPI.Comm,
        digit_cutoff: int):
    """
    Determines the interpolation points FEniCSx needs to interpolate the coupling boundary and the coordinates for the preCICE mesh.

    Parameters
    ----------
    function_space:  fem.FunctionSpace
        The function space of the problem
    coupling_boundary:
        A callable function describing the coupling boundary
    comm: MPI.Comm
        The used MPI communicator
    digit_cutoff: int
        Specifies the decimal place at which the coordinates are rounded

    Returns
    -------
    (ndarray, list, dict):
        Returns a triplet of (interpolation coordinates, interpolation cells, function values to be sent to other MPI ranks)
    """
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()

    # domain of function space
    domain = function_space.mesh
    # function used to query interpolation coordinates
    query_function = fem.Function(function_space)

    # variables and function definition required for querying the coordinates FEniCSx needs
    interpolation_coordinates = []
    vec_len = function_space.dofmap.bs

    def query_coordinates(x):
        # to avoid UnboundLocalError, interpolation_coordinates is an array to which x is appended to
        interpolation_coordinates.append(np.transpose(copy.deepcopy(x)))
        # directly round and make each rounded coordinate unique to keep the code clean and to reduce memory consumption
        interpolation_coordinates[-1] = round_unique_coordinates(
            interpolation_coordinates[-1], digit_cutoff=digit_cutoff)
        return np.zeros((vec_len, x.shape[1]))

    # determine process local cells that are on the coupling boundary
    dofs_coupling = fem.locate_dofs_geometrical(function_space, coupling_boundary)
    dofs_coupling_coordinates = function_space.tabulate_dof_coordinates()[dofs_coupling]
    bb_tree = geometry.bb_tree(domain, domain.geometry.dim)
    cell_candidates_local = geometry.compute_collisions_points(bb_tree, dofs_coupling_coordinates)
    cell_candidates_local = geometry.compute_colliding_cells(domain, cell_candidates_local, dofs_coupling_coordinates)
    # the cell candidates with local cell ids
    cell_candidates_local = np.unique(cell_candidates_local.array)

    # determine process-owned cells
    index_map = domain.topology.index_map(domain.topology.dim)
    # range of owned cells
    owned_cell_ids = index_map.local_range
    owned_cell_ids = np.arange(owned_cell_ids[0], owned_cell_ids[1])
    # map local cell ids to global cell ids to determine ghost cells
    cell_candidates_global = index_map.local_to_global(cell_candidates_local)
    # cells to be interpolated over must be owned! Find intersection of cell candidates and owned cells
    owned_and_candidate_cells_global = np.intersect1d(owned_cell_ids, cell_candidates_global)
    # map back to local cell indexing
    owned_and_candidate_cells_local = index_map.global_to_local(owned_and_candidate_cells_global)

    # get all interpolation points of owned domain
    query_function.interpolate(query_coordinates, owned_and_candidate_cells_local)
    # convert the numpy array to a set of tuples to allow set operations
    interpolation_coordinates = set(map(tuple, interpolation_coordinates[0]))

    # to avoid creating an ill-posed mapping problem for preCICE (especially RBF mapping), equal coordinates used by multiple MPI ranks
    # must be determined and removed by all but one MPI rank.

    # rule: if rankA and rankB use coordinate x and rankA < rankB, rankA keeps
    # x and rankB discards the coordinate, else rankB keeps x

    # send interpolation points of complete boundary (interpolation_coordinates[0]) to higher ranks
    # OR receive interpolation points that potentially need to be filtered out from lower ranks

    # TODO add persistent communication for performance increase?

    duplicate_coordinates_per_rank = {}  # save the filtered coordinates per rank to reduce communication volume

    for source_rank in range(comm_rank):
        interpolation_coordinates_from_source = comm.recv(source=source_rank)  # recv a set of tuples
        # save duplicates
        duplicate_coordinates_per_rank[source_rank] = interpolation_coordinates & interpolation_coordinates_from_source
        # filter out duplicates
        interpolation_coordinates = interpolation_coordinates - interpolation_coordinates_from_source

    for dest_rank in range(comm_rank + 1, comm_size):
        comm.send(interpolation_coordinates, dest_rank)

    coordinates_to_send = {}
    # receive coordinates that need to be communicated
    for source_rank in range(comm_size - 1, comm_rank, -1):
        # TODO if there is no duplicate coordinate between two ranks, there is no need to send an empty array
        coordinates_to_send[source_rank] = comm.recv(source=source_rank)
    # send coordinates
    for dest_rank in range(comm_rank):
        comm.send(duplicate_coordinates_per_rank[dest_rank], dest_rank)

    # convert set of tuples to numpy array
    interpolation_coordinates = np.array(list(interpolation_coordinates))
    # no need to change coordinates_to_send as they need to be tuples anyways

    return interpolation_coordinates, owned_and_candidate_cells_local, coordinates_to_send


def interpolate_boundary_function(
        read_values: dict,
        function_type: FunctionType,
        values_to_send: dict,
        boundary_function: fem.Function,
        boundary_cells: list,
        comm: MPI.Comm,
        is_empty_rank: bool,
        digit_cutoff: int):
    """
    Interpolates the coupling boundary function at the specified cells.

    Parameters
    ----------
    read_values: dict
        A dict of (coordinates: function values) that were read from preCICE
    function_type: FunctionType
        Type of the function that needs to be interpolated
    values_to_send: dict
        A dict of (rank: coordinates) that defines which function values at which coordinates must be sent to the corresponding rank
    boundary_function: fem.Function
        The function that should be interpolated
    boundary_cells: list
        A list of cell indices to be interpolated
    comm: MPI.Comm
        The MPI communicator to be used
    is_empty_rank: bool
        Specifies if the rank has no relation to the coupling boundary
    digit_cutoff: int
        Specifies at which decimal place the coordinates are rounded
    """
    if is_empty_rank:
        # an empty rank does not need to do any interpolation
        for dest_rank in values_to_send.keys():
            comm.send(values_to_send[dest_rank], dest_rank)
    else:
        # check if it is vector or scalar valued
        vector_length = 0
        if function_type is FunctionType.SCALAR:
            # scalar valued function has vector length 1
            vector_length = 1
        else:
            # vector valued function
            # vector length is determined by getting one value of the values dict
            vector_length = len(read_values[next(iter(read_values))])

        # append filtered coordinates again to provide all necessary points for interpolation
        for source_rank in range(comm.Get_rank()):
            value = comm.recv(source=source_rank)
            read_values.update(value)

        # send the values that other ranks need for interpolation
        for dest_rank in values_to_send.keys():
            payload = {coord: read_values[coord] for coord in values_to_send[dest_rank]}
            comm.send(payload, dest_rank)

        # define the interpolation function
        def interpolation_function(x):
            # truncation to smaller dimension not necessary because fenicsx coordinates are always 3D
            coords = np.zeros_like(x)
            np.round(x, digit_cutoff, coords)
            coords = np.transpose(coords)
            npoints = len(coords)

            if vector_length == 1:
                # function is scalar valued
                return_value = np.zeros((npoints,))
                for idx, c in enumerate(coords):
                    return_value[idx] = read_values[tuple(c)]
            else:
                # function is vector valued
                return_value = np.zeros((vector_length, npoints))
                for idx, c in enumerate(coords):
                    return_value[:, idx] = read_values[tuple(c)]
            return return_value

        # do the actual interpolation
        boundary_function.interpolate(interpolation_function, boundary_cells)
