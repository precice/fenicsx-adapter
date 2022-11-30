from ufl import dot, form, dx, inner
from dolfinx.fem import assemble_scalar, assemble
import numpy as np
from mpi4py import MPI

def compute_errors(mesh, u_approx, u_ref, total_error_tol=10 ** -4):
    # compute pointwise L2 error
    
    error_normalized = (u_ref - u_approx) / u_ref

    inner_p = inner(error_normalized, error_normalized)
    assembly = assemble_scalar(inner_p * dx)

    error_total = np.sqrt(assembly)

    # error_total = np.sqrt(assemble(inner(error_normalized, error_normalized) * dx))

    # error_pointwise = form(dot(u_approx - u_ref, u_approx - u_ref)*dx)
    # error_total =  np.sqrt(mesh.comm.allreduce(assemble_scalar(error_pointwise), op=MPI.SUM))

    assert (error_total < total_error_tol)

    # return error_total, error_pointwise
    return error_total
