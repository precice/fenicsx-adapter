"""
This module consists of helper functions used in the Adapter class. Names of the functions are self explanatory
"""

from dolfinx import fem, geometry, mesh as msh
import numpy as np
from enum import Enum
import logging
import copy
from numbers import Number

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

# TODO make it potentially variable?
COORDINATE_DIGITS = 10

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
    bb_tree_facet = geometry.bb_tree(mesh, mesh.geometry.dim - 1)
    local_cells = mesh.topology.index_map(mesh.topology.dim).local_range
    midpoint_tree_faces = geometry.create_midpoint_tree(mesh, mesh.topology.dim-1, np.arange(local_cells[0], local_cells[1]))
    facet_to_cell_map = mesh.topology.connectivity(2,3)

    cells = []
    points = []

    # Find cells whose bounding-box collide with the the points
    cell_candidates = geometry.compute_collisions_points(bb_tree, local_coords)
    # Choose one of the cells that contains the point
    colliding_cells = geometry.compute_colliding_cells(mesh, cell_candidates, local_coords)
    for i, point in enumerate(local_coords):
        if len(colliding_cells.links(i)) > 0:
            points.append(point)
            cells.append(colliding_cells.links(i)[0])
        else:
            # point is outside domain, probably because of rounding
            closest_facet_idx = geometry.compute_closest_entity(bb_tree_facet, midpoint_tree_faces, mesh, point)[0]
            # change point such that it is in the function domain
            # -> find midpoint to closest facet and just use this.
            closest_midpoint_facet = msh.compute_midpoints(mesh, mesh.topology.dim-1, np.array([closest_facet_idx]))[0]
            
            # if links has more than 1 entry, the closest midpoint seems to be between two entities. Just pick one
            closest_cell_idx_3d = facet_to_cell_map.links(closest_facet_idx)[0]
            
            cells.append(closest_cell_idx_3d)
            points.append(closest_midpoint_facet)

    precice_data = fenicsx_function.eval(points, cells)
    return np.array(precice_data)

def get_fenicsx_interpolation_points(function_space : fem.FunctionSpace, coupling_boundary):
    domain = function_space.mesh
    
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
    
    # query and store the interpolation points of owned boundary cells
    interpolation_coordinates = []
    vec_len = function_space.dofmap.bs
    def query_coordinates(x):
        # to avoid UnboundLocalError, interpolation_coordinates is an array to which x is appended to
        interpolation_coordinates.append(np.transpose(copy.deepcopy(x)))
        return np.zeros((vec_len, x.shape[1]))
    query_function = fem.Function(function_space)
    query_function.interpolate(query_coordinates, owned_and_candidate_cells_local)
    
    # round the coordinates to avoid close points (i.e. those that are equal until the 8 decimal place) to be able to use rbf mapping
    tmp = np.zeros_like(interpolation_coordinates[0])
    np.round(interpolation_coordinates[0], COORDINATE_DIGITS, tmp)
    interpolation_coordinates = tmp
    interpolation_coordinates = np.unique(interpolation_coordinates, axis=0)
    
    return interpolation_coordinates, owned_and_candidate_cells_local

def interpolate_fenicsx(values, function_type):
    # check if it is vector or scalar valued
    vector_length = 0
    if function_type is FunctionType.SCALAR:
        # scalar valued function has vector length 1
        vector_length = 1
    else:
        # vector valued function
        # vector length is determined by getting one value of the values dict
        vector_length = len(values[next(iter(values))])
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
