import ufl
from ufl import dot, dx, inner
from dolfinx.fem import assemble_scalar, assemble, form
import numpy as np
from mpi4py import MPI

def compute_errors(u_approx, u_ref, total_error_tol=10 ** -4):
    # compute pointwise L2 error
    mesh = u_ref.function_space.mesh
    error_pointwise = form((u_approx - u_ref) ** 2 *dx)
    error_total =  np.sqrt(mesh.comm.allreduce(assemble_scalar(error_pointwise), MPI.SUM))

    assert (error_total < total_error_tol)

    # return error_total, error_pointwise
    return error_total
