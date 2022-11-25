import ufl
import dolfinx
from petsc4py import PETSc

def project(v, target_func, V, bcs=[]):
    # Ensure we have a mesh and attach to measure
    #V = target_func.function_space
    dx = ufl.dx(V.mesh)

    # Define variational problem for projection
    w = ufl.TestFunction(V)
    Pv = ufl.TrialFunction(V)
    a = dolfinx.fem.form(ufl.inner(Pv, w) * dx)
    L = dolfinx.fem.form(ufl.inner(v, w) * dx)

    # Assemble linear system
    A = dolfinx.fem.petsc.assemble_matrix(a, bcs)
    A.assemble()
    b = dolfinx.fem.petsc.assemble_vector(L)
    dolfinx.fem.petsc.apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(b, bcs)

    # Solve linear system
    solver = PETSc.KSP().create(A.getComm())
    solver.setOperators(A)
    solver.solve(b, target_func.vector)

def compute_errors(u_approx, u_ref, v, total_error_tol=10 ** -4):
    # compute pointwise L2 error
    error_normalized = (u_ref - u_approx) / u_ref
    # project onto function space
    #error_pointwise = project(abs(error_normalized), v)
    V = u_ref.function_space
    error_pointwise = project(v, abs(error_normalized), V)
    # determine L2 norm to estimate total error
    error_total = ufl.sqrt(dolfinx.fem.assemble(ufl.inner(error_pointwise, error_pointwise) * ufl.dx))
    error_pointwise.rename("error", " ")

    assert (error_total < total_error_tol)

    return error_total, error_pointwise
