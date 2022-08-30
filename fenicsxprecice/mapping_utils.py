"""
This module consists of helper functions used in the adapter related to data mapping. Names of the functions are self explanatory
"""

from dolfinx.fem import FunctionSpace, Function
from dolfinx.mesh import Mesh
from dolfinx.geometry import BoundingBoxTree, compute_colliding_cells, compute_collisions
import numpy as np


def precompute_eval_vertices(precice_vertices: np.ndarray, mesh: Mesh):
    tree = BoundingBoxTree(mesh, mesh.geometry.dim)
    cell_candidates = compute_collisions(tree, precice_vertices)
    cell = compute_colliding_cells(mesh, cell_candidates, precice_vertices)

    # Adjacency list: convert to array and take the first match for each item.
    # Use offsets (but the last one) to compute indices in the array
    list_of_cells = cell.array[cell.offsets[:-1]]
    return list_of_cells
