from dolfinx import fem, mesh
from mpi4py import MPI
import numpy as np

domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
V = fem.FunctionSpace(domain, ("CG", 1))
print(V)

domain.topology.create_connectivity(1, 2)
#setup topology

boundary_facets = np.flatnonzero(mesh.compute_boundary_facets(domain.topology))
#boundary_facets = mesh.exterior_facet_indices(domain.topology)

def exclude_straight_boundary(V):
    return locate_dofs_geometrical(V, lambda x: not np.isclose(x[0], x_coupling, tol) or np.isclose(x[1], y_top, tol) or np.isclose(x[1], y_bottom, tol))
