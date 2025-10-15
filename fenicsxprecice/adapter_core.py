"""
This module consists of helper functions used in the Adapter class. Names of the functions are self explanatory
"""

from dolfinx import fem, geometry
import numpy as np
from enum import Enum
import logging
import copy
from numbers import Number

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


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
    for i, point in enumerate(local_coords):
        if len(colliding_cells.links(i)) > 0:
            points.append(point)
            cells.append(colliding_cells.links(i)[0])

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
    owned_idx = list(index_map.local_range)
    owned_cell_ids = np.arange(owned_idx[0], owned_idx[1])
    # map local cell ids to global cell ids to determine ghost cells
    cell_candidates_global = index_map.local_to_global(cell_candidates_local)
    
    # cells to be interpolated over must be owned! Find intersection of cell candidates and owned cells
    owned_and_candidate_cells_global = np.intersect1d(owned_cell_ids, cell_candidates_global)
    owned_and_candidate_cells_local = index_map.global_to_local(owned_and_candidate_cells_global)
    
    # query and store the interpolation points of owned boundary cells
    interpolation_points = []
    def query_coordinates(x):
        interpolation_points.append(np.transpose(copy.deepcopy(x)))
        return x[0]*0
    query_function = fem.Function(function_space)
    query_function.interpolate(query_coordinates, owned_and_candidate_cells_local)
    
    return interpolation_points[0], owned_and_candidate_cells_local

def interpolate_fenicsx(values):
    first_key = next(iter(values))
    # check if it is vector or scalar valued
    vector_length = 0
    if isinstance(values[first_key], Number) or np.isscalar(values[first_key]):
        # scalar valued function
        vector_length = 1
    else:
        # vector valued function
        vector_length = len(values[first_key])
    def return_function(x):
        # truncation to smaller dimension not necessary because fenicsx coordinates are always 3D
        coords = np.transpose(x)
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
