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

def mpi_send_dict(comm: MPI.Comm, msg:dict):
    """_
    Helper function to send messages. msg is of type dict, where keys are the ranks to the message that should be sent to the corresponding rank
    """
    mpi_reqs = []
    for key in msg.keys():
        mpi_reqs.append(comm.isend(msg[key], key))
    return mpi_reqs
    
def mpi_recv(comm: MPI.Comm, ranks):
    mpi_reqs = {}
    for rank in ranks:
        mpi_reqs[rank] = comm.irecv(source=rank)
    mpi_msgs = {}
    for key in mpi_reqs.keys():
        mpi_msgs[key] = mpi_reqs[key].wait()
    return mpi_msgs

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
    mesh.topology.create_entities(2)
    mesh.topology.create_connectivity(2, 3)

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
    midpoint_tree_faces = geometry.create_midpoint_tree(mesh, mesh.topology.dim, all_cells)
    
    for i, point in enumerate(local_coords):
        if len(colliding_cells.links(i)) > 0:
            points.append(point)
            cells.append(colliding_cells.links(i)[0])
        else:
            
            # FIXME: it is more accurate to compute the closest midpoint to the facet and not the cell
            
            # point is outside domain, probably because of rounding
            closest_cell_idx = geometry.compute_closest_entity(bb_tree, midpoint_tree_faces, mesh, point)[0]
            # change point such that it is in the function domain
            # -> find midpoint to closest cell and just use this.
            closest_midpoint_cell = msh.compute_midpoints(mesh, mesh.topology.dim, np.array([closest_cell_idx]))[0]
            
            
            cells.append(closest_cell_idx)
            points.append(closest_midpoint_cell)

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

    
    # to avoid creating an ill-posed mapping problem for preCICE, equal coordinates used that are used to define the preCICE mesh across multiple MPI ranks
    # must be determined and removed by all but one MPI rank.
    # ghost cells are not guaranteed so determine neighboring ranks by interprocess facets
    
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    # map interprocess facet to a cell
    map_facet_to_cell = domain.topology.connectivity(domain.topology.dim - 1, domain.topology.dim)
    
    # determine neighbors
    # to map local facets to global facets, create index map of topological dimension 1 smaller than from the mesh
    # -> with global indices, neighbors can be determined if at least one facet index is equal between two mpi ranks
    index_map_facets = domain.topology.index_map(domain.topology.dim - 1)
    # exclude the interpolation points at the ghost facets
    ghost_facets = index_map_facets.ghosts # global indexing
    ghost_facets_owners = index_map_facets.owners
    
    # owned interprocess facets
    owned_interprocess_facets_local = np.sort(np.setdiff1d(domain.topology.interprocess_facets(), index_map_facets.global_to_local(ghost_facets)))
    
    # each facet is shared ONLY by two MPI ranks! -> rule: the process with the smaller rank should keep the interpolation points at the facets
    # -> send to ranks that are larger than self rank the interpolation points from process_boundary_cells. Ranks receiving interpolation points can then remove them locally
    #TODO: 1. for each rank, find boundary cells connected to interprocess facets (use all_interprocess_facets & map_facet_to_cell for that) 
    interprocess_connected_cells = np.zeros_like(ghost_facets)
    current_idx = 0
    for facet in index_map_facets.global_to_local(ghost_facets):
        #if comm.Get_rank() == 1:
        #    print(owned_cell_ids)
        #    print(facet)
        #    print(map_facet_to_cell.links(facet))
        # map_facet_to_cell.links(facet) produces local indexed cells
        links = map_facet_to_cell.links(facet)
        found = False
        if len(links) > 0:
            for link in links:
                # the link should be owned!
                if link < index_map.size_local:
                    interprocess_connected_cells[current_idx] = link
                    current_idx += 1
                    found = True
                    break
            if not found:
                # should actually be never the case, but just to be sure
                raise RuntimeError("No owned cell to a ghost facet found!")
        else:
            # should actually be never the case, but just to be sure
            raise RuntimeError("No link found while determining interpolation coordinates")
        
    #TODO: 2. determine interpolation points for each cell/rank and send it to rank that needs to filter these interpolation points
    # each rank sends a list of global facet indices to the facet owner
    facets_per_rank = {}
    for rank in range(comm_size):
        facets_per_rank[rank] = []
        for idx in range(len(ghost_facets)):
            if ghost_facets_owners[idx] == rank:
                facets_per_rank[rank].append(ghost_facets[idx])
    
    # send to all mpi ranks (except the local rank) facet list
    other_ranks = []
    for r in range(comm_size):
        if r == comm_rank:
            continue
        other_ranks.append(r)
        
    mpi_requests_send = mpi_send_dict(comm, facets_per_rank)
    global_facets = mpi_recv(comm, other_ranks)
    MPI.Request.Waitall(mpi_requests_send)
    
    # now, each mpi rank must interpolate on the cells that belong to the received facets
    # first, determine local cells for each facet
    cells_to_interpolate = {}
    for key in global_facets.keys():
        cells_to_interpolate[key] = np.zeros((len(global_facets[key]),), dtype=np.int64)
        local_facets = index_map_facets.global_to_local(np.array(global_facets[key]))
        for idx, l_facet in enumerate(local_facets):
            cells_to_interpolate[key][idx] = map_facet_to_cell.links(l_facet)[0]
    
    # TODO: Now, interpolate for each rank (key) stored in cells_to_interpolate at corresponding cells (values)
    interpolation_points_per_rank = {}
    for key in cells_to_interpolate.keys():
        query_function.interpolate(query_coordinates, cells_to_interpolate[key])
        interpolation_points_per_rank[key] = interpolation_coordinates[-1]
        m_print(comm_rank, interpolation_coordinates[-1])
    
    mpi_requests_send = mpi_send_dict(comm, interpolation_points_per_rank)
    # get interpolation points that need to be filtered
    excluding_interpolation_coords = mpi_recv(comm, other_ranks)
    MPI.Request.Waitall(mpi_requests_send)
    
    #TODO: 3. each rank receiving interpolation points to filter must filter them from their list that they use to define the preCICE mesh
    # now, the rest of the interpolation points that were sent to the other ranks can be discarded
    interpolation_coordinates = interpolation_coordinates[0]
    interp_coord_set = set(tuple(c) for c in interpolation_coordinates)
    for key in excluding_interpolation_coords.keys():
        # first get the intersection to tell the owning rank which coordinates are sufficient to send during the simulation to avoid filtering and reduce communication volume
        excl_coord = set(tuple(c) for c in excluding_interpolation_coords[key])
        excluding_interpolation_coords[key] = np.array(list(set.intersection(interp_coord_set, excl_coord)))
        # subtract the coordinates to get the coordinates for rank local preCICE mesh
        interp_coord_set = interp_coord_set - excl_coord
    
    # send and receive the coordinates that need to be communicated across different ranks
    mpi_requests_send = mpi_send_dict(comm, excluding_interpolation_coords)
    requested_interpolation_points = mpi_recv(comm, other_ranks)
    MPI.Request.Waitall(mpi_requests_send)
    #TODO: 4. enable a persistent communication between ranks which require interpolation data from neighbors -> at best, send a dict which can simply be used during interpolation!!!!
    
    # interpolation_coordinates: coordinates to define the process-local precice mesh
    # owned_and_candidate_cells_local ... clear
    # requested_interpolation_points ... coordinates need to be sent to other ranks
    # excluding_interpolation_coords ... coordinates that must be received from other ranks
    with open(f"Rank{comm_rank}.txt", "w") as file:
        file.write(np.array2string(interpolation_coordinates))
    
    return interpolation_coordinates, owned_and_candidate_cells_local, requested_interpolation_points, excluding_interpolation_coords

def get_fenicsx_interpolation_points_v2(function_space : fem.FunctionSpace, coupling_boundary, comm: MPI.Comm):
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()
    # send to all mpi ranks (except the local rank) facet list
    other_ranks = []
    for r in range(comm_size):
        if r == comm_rank:
            continue
        other_ranks.append(r)
    
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
        interpolation_coordinates_from_source = comm.recv(source = source_rank) # get this here as a set of tuples
        # save duplicates
        duplicate_coordinates_per_rank[source_rank] = interpolation_coordinates & interpolation_coordinates_from_source
        # filter out duplicates
        interpolation_coordinates = interpolation_coordinates - interpolation_coordinates_from_source
        
    for receiver_rank in range(comm_rank + 1, comm_size):
        comm.send(interpolation_coordinates, receiver_rank)
    
    
    coordinates_to_send = {}
    # receive coordinates that need to be communicated
    for source_rank in range(comm_rank + 1, comm_size):
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
