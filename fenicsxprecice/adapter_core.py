"""
This module consists of helper functions used in the Adapter class. Names of the functions are self explanatory
"""

from dolfinx import fem, geometry
import numpy as np
from enum import Enum
import logging
import copy

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
        return copy.deepcopy(self._ids)

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
        elif input_obj.num_sub_spaces == 2:
            return FunctionType.VECTOR
    elif isinstance(input_obj, fem.Function):
        if len(input_obj.x.array.shape) == 1:
            return FunctionType.SCALAR
        elif input_obj.x.array.shape[1] > 1:
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


def get_fenicsx_vertices(function_space, coupling_subdomain, dims):
    """
    Extracts vertices which FEniCSx accesses on this rank and which lie on the given coupling domain, from a given
    function space.

    Parameters
    ----------
    function_space : FEniCSx function space
        Function space on which the finite element problem definition lives.
    coupling_subdomain : FEniCSx Domain
        Subdomain consists of only the coupling interface region.
    dims : int
        Dimension of problem.

    Returns
    -------
    ids : numpy array
        Array of ids of fenicsx vertices.
    coords : numpy array
        The coordinates of fenicsx vertices in a numpy array [N x D] where
        N = number of vertices and D = dimensions of geometry.
    """

    # Get mesh from FEniCSx function space
    mesh = function_space.mesh

    # Get coordinates and IDs of all vertices of the mesh which lie on the coupling boundary.
    try:
        ids = fem.locate_dofs_geometrical(function_space, coupling_subdomain)
        if dims == 2 or dims == 3:
            coords = function_space.tabulate_dof_coordinates()[ids]  # we get 3d coordinates here
        else:
            coords = np.array([])
    except Exception as e:  # fall back to old method  # TODO is that too general? Better use, e.g., IndexError here?
        print("Caught the following exception in the detection of the coupling subdomain:\n{e}")
        print("Falling back to old, point-wise method.")
        ids, coords = [], []
        for idx in range(mesh.geometry.x.shape[0]):
            v = mesh.geometry.x[idx]
            if coupling_subdomain(v):
                ids.append(idx)
                if dims == 2:
                    coords.append([v[0], v[1]])
        ids = np.array(ids)
        coords = np.array(coords)

    return ids, coords
