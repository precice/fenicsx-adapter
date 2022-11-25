"""
This module consists of helper functions used in the Adapter class. Names of the functions are self explanatory
"""

from dolfinx.fem import FunctionSpace, Function
import numpy as np
from enum import Enum
import logging
import copy

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class VertexType(Enum):
    """
    Defines type of vertices that exist in the adapter.
    OWNED vertices are vertices on the coupling interface owned by this rank
    UNOWNED vertices are vertices on the coupling interface which are not owned by this rank. They are borrowed
        vertices from neigbouring ranks
    FENICSX vertices are OWNED + UNOWNED vertices in the order as seen by FEniCSx
    """
    OWNED = 153
    UNOWNED = 471
    FENICSX = 557


class Vertices:
    """
    Vertices class provides a generic skeleton for vertices. A set of vertices has a set of IDs and
    coordinates as defined in FEniCSx.
    """

    def __init__(self, vertex_type):
        self._vertex_type = vertex_type
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
        A DOLFINx function.

    Returns
    -------
    tag : bool
        0 if input_function is SCALAR and 1 if input_function is VECTOR.
    """
    if isinstance(input_obj, FunctionSpace):
        space = input_obj
    elif isinstance(input_obj, Function):
        space = input_obj.function_space
    else:
        raise Exception("Error: determine_function_type must take a Function or FunctionSpace as argument.")

    if space.num_sub_spaces == 0:
        return FunctionType.SCALAR
    elif space.num_sub_spaces == 2:
        return FunctionType.VECTOR
    else:
        raise Exception("Error determining type of the provided DOLFINx Function or FunctionSpace")


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
        on_subdomain = coupling_subdomain(mesh.geometry.x.T)
        ids, = np.where(on_subdomain)
        if dims == 2:
            coords = mesh.geometry.x[ids][:, :2]
        else:
            coords = np.array([])
    except Exception as e:  # fall back to old method  # TODO is that to general? Better use, e.g., IndexError here?
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
