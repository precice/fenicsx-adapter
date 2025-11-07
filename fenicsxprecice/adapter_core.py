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

# TODO make it potentially variable?
COORDINATE_DIGITS = 8

def round_unique_coordinates(coords):
    # round the coordinates to avoid close points (i.e. those that are equal until the 8 decimal place) to be able to use rbf mapping
    tmp = np.zeros_like(coords)
    np.round(coords, COORDINATE_DIGITS, tmp)
    tmp = np.unique(tmp, axis=0)
    return tmp

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
    
class CouplingBoundaryProcessing(Enum):
    AUTOMATIC = 1
    MANUAL = 2


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


def convert_fenicsx_to_precice(fenicsx_function, local_coords):
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
    
    #generate from the already computed cell candidates the midpoint tree! (-> original cell owner is in the list anyways)
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
            # in each direction, COORDINATE_DIGITS defines on how much the point has been shifted
            # -> move it in the direction of midpoint cell by at max. this amount, so, in each direction maximal 1e-COORDINATE_DIGITS
            direction = closest_midpoint_cell-point
            direction = np.sign(direction)*(10**(-COORDINATE_DIGITS))
            new_point = point + direction
            
            cells.append(closest_cell_idx)
            points.append(new_point)

    precice_data = fenicsx_function.eval(points, cells)
    return np.array(precice_data)

def m_print(rank, msg):
    print(f"Rank {rank}: {msg}")

def get_fenicsx_interpolation_points(function_space : fem.FunctionSpace, coupling_boundary, comm: MPI.Comm):
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()
    
    # domain of function space
    domain = function_space.mesh
    # dummy function used to query interpolation coordinates
    query_function = fem.Function(function_space)
    
    # variables and function definition required for querying the coordinates FEniCSx needs
    interpolation_coordinates = []
    vec_len = function_space.dofmap.bs
    def query_coordinates(x):
        # to avoid UnboundLocalError, interpolation_coordinates is an array to which x is appended to
        interpolation_coordinates.append(np.transpose(copy.deepcopy(x)))
        # directly round and make each rounded coordinate unique to keep the code clean
        interpolation_coordinates[-1] = round_unique_coordinates(interpolation_coordinates[-1])
        
        return np.zeros((vec_len, x.shape[1]))
    
    # determine process local cells that are on the coupling boundary
    dofs_coupling = fem.locate_dofs_geometrical(function_space, coupling_boundary)
    dofs_coupling_coordinates = function_space.tabulate_dof_coordinates()[dofs_coupling]
    bb_tree = geometry.bb_tree(domain, domain.geometry.dim)
    cell_candidates_local = geometry.compute_collisions_points(bb_tree, dofs_coupling_coordinates)
    cell_candidates_local = geometry.compute_colliding_cells(domain, cell_candidates_local, dofs_coupling_coordinates)
    # the cell candidates with local cell ids
    cell_candidates_local = np.unique(cell_candidates_local.array)
    
    # determine process owned cells
    index_map = domain.topology.index_map(domain.topology.dim)
    # range of owned cells
    owned_cell_ids = index_map.local_range
    owned_cell_ids = np.arange(owned_cell_ids[0], owned_cell_ids[1])
    # map local cell ids to global cell ids to determine ghost cells
    cell_candidates_global = index_map.local_to_global(cell_candidates_local)
    # cells to be interpolated over must be owned! Find intersection of cell candidates and owned cells
    owned_and_candidate_cells_global = np.intersect1d(owned_cell_ids, cell_candidates_global)
    owned_and_candidate_cells_local = index_map.global_to_local(owned_and_candidate_cells_global)
    
    # get all interpolation points of owned domain
    query_function.interpolate(query_coordinates, owned_and_candidate_cells_local)
    # convert the numpy array to a set of tuples to make set operations
    interpolation_coordinates = set(map(tuple, interpolation_coordinates[0]))
    
    
    # to avoid creating an ill-posed mapping problem for preCICE, equal coordinates used that are used to define the preCICE mesh across multiple MPI ranks
    # must be determined and removed by all but one MPI rank.
    
    # send interpolation points of complete boundary (interpolation_coordinates[0]) to higher ranks 
    # OR receive interpolation points that potentially need to be filtered out from lower ranks
    
    # TODO add persistent communication for performance increase?
    
    duplicate_coordinates_per_rank = {} # save the filtered coordinates per rank to reduce communication volume
    
    for source_rank in range(comm_rank):
        interpolation_coordinates_from_source = comm.recv(source = source_rank) # recv a set of tuples
        # save duplicates
        duplicate_coordinates_per_rank[source_rank] = interpolation_coordinates & interpolation_coordinates_from_source
        # filter out duplicates
        interpolation_coordinates = interpolation_coordinates - interpolation_coordinates_from_source
    
    for receiver_rank in range(comm_rank + 1, comm_size):
        comm.send(interpolation_coordinates, receiver_rank)
    
    coordinates_to_send = {}
    # receive coordinates that need to be communicated
    for source_rank in range(comm_size-1, comm_rank, -1):
        #TODO if there is no duplicate coordinate between two ranks, there is no need to send an empty array
        coordinates_to_send[source_rank] = comm.recv(source = source_rank)
    # send coordinates
    for receiver_rank in range(comm_rank):
        comm.send(duplicate_coordinates_per_rank[receiver_rank], receiver_rank)
    
    # convert set of tuples to numpy array
    interpolation_coordinates = np.array(list(interpolation_coordinates))
    # no need to change coordinates_to_send as they need to be tuples anyways

    return interpolation_coordinates, owned_and_candidate_cells_local, coordinates_to_send


def interpolate_fenicsx(values, function_type, values_to_send, comm:MPI.Comm):
    # check if it is vector or scalar valued
    vector_length = 0
    if function_type is FunctionType.SCALAR:
        # scalar valued function has vector length 1
        vector_length = 1
    else:
        # vector valued function
        # vector length is determined by getting one value of the values dict
        vector_length = len(values[next(iter(values))])

    # TODO come up with something more efficient

    # append previously filtered out coordinates to values
    for source_rank in range(comm.Get_rank()):
        value = comm.recv(source = source_rank)
        values.update(value)
    
    # the values to append obvioulsy need to be send
    for dest_rank in values_to_send.keys():
        payload = {coord : values[coord] for coord in values_to_send[dest_rank]}
        comm.send(payload, dest_rank)
    
    def return_function(x):
        # truncation to smaller dimension not necessary because fenicsx coordinates are always 3D
        coords = np.zeros_like(x)
        np.round(x, COORDINATE_DIGITS, coords)
        coords = np.transpose(coords)
        npoints = len(coords)
        
        if vector_length == 1:
            # function is scalar valued
            return_value = np.zeros((npoints,))
            for idx, c in enumerate(coords):
                return_value[idx] = values[tuple(c)]
        else:
            # function is vector valued
            return_value = np.zeros((vector_length, npoints))
            for idx, c in enumerate(coords):
                return_value[:, idx] = values[tuple(c)]
        return return_value
    
    return return_function
