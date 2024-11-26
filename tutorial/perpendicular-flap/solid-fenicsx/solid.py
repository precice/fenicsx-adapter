import os

import dolfinx as dfx
import numpy as np
import ufl
from dolfinx.fem.petsc import (
    LinearProblem,
    apply_lifting,
    assemble_matrix_mat,
    assemble_vector,
    set_bc,
)
from dolfinx.mesh import CellType, create_rectangle
from fenicsxprecice import Adapter
from mpi4py import MPI
from petsc4py import PETSc

# ------ #
# SOLVER #
# ------ #

# EXTEND the linear petsc.LinearProblem to add values into the bilinear form matrix on
# desired DOFs. Wee need this due to the incoming Force are discrete vector, so wee need
# to avoid domain integration


class DiscreteLinearProblem(LinearProblem):
    def __init__(
        self,
        a,
        L,
        bcs,
        u,
        point_dofs,
        petsc_options=None,
        form_compiler_options=None,
        jit_options=None,
    ):
        if point_dofs is not None:
            if not isinstance(type(point_dofs), np.ndarray):
                point_dofs = np.array([point_dofs], dtype="int32")
        self._point_dofs = point_dofs

        super().__init__(a, L, bcs, u, petsc_options, form_compiler_options, jit_options)

    def solve(self, values):
        """Solve the problem."""
        # Assemble lhs
        self._A.zeroEntries()
        assemble_matrix_mat(self._A, self._a, bcs=self.bcs)
        self._A.assemble()

        # Assemble rhs
        with self._b.localForm() as b_loc:
            b_loc.set(0)

        if self._point_dofs is not None and values is not None:
            # apply load in dofs
            self._b[self._point_dofs] = values
            self._b.assemble()

        assemble_vector(self._b, self._L)

        # Apply boundary conditions to the rhs
        apply_lifting(self._b, [self._a], bcs=[self.bcs])
        self._b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self._b, self.bcs)

        # Solve linear system and update ghost values in the solution
        self._solver.solve(self._b, self._x)
        self.u.x.scatter_forward()

        return self.u


# --------- #
# CONSTANTS #
# --------- #

MPI_COMM = MPI.COMM_WORLD
CURRENT_FOLDER = os.path.dirname(__file__)
PARTICIPANT_CONFIG = os.path.join(CURRENT_FOLDER, "precice-config.json")
RESULTS_DIR = os.path.join(CURRENT_FOLDER, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.chdir(CURRENT_FOLDER)

WRITER = dfx.io.VTKFile(MPI_COMM, f"{RESULTS_DIR}/result.pvd", "w")

WIDTH, HEIGHT = 0.1, 1
NX, NY = 2, 15 * 8

E = 4000000.0
NU = 0.3
RHO = 3000.0

BETA_ = 0.25
GAMMA_ = 0.5

# ------- #
# MESHING #
# ------- #

domain = create_rectangle(
    MPI_COMM,
    [np.array([-WIDTH / 2, 0]), np.array([WIDTH / 2, HEIGHT])],
    [NX, NY],
    cell_type=CellType.quadrilateral,
)
dim = domain.topology.dim

# -------------- #
# Function Space #
# -------------- #
degree = 2
shape = (dim,)
V = dfx.fem.functionspace(domain, ("P", degree, shape))
u = dfx.fem.Function(V, name="Displacement")
f = dfx.fem.Function(V, name="Force")

# ------------------- #
# Boundary conditions #
# ------------------- #
tol = 1e-14


def clamped_boundary(x):
    return abs(x[1]) < tol


def neumann_boundary(x):
    """Determines whether a node is on the coupling boundary."""
    return np.logical_or((np.abs(x[1] - HEIGHT) < tol), np.abs(np.abs(x[0]) - WIDTH / 2) < tol)


fixed_boundary = dfx.fem.locate_dofs_geometrical(V, clamped_boundary)
coupling_boundary = dfx.mesh.locate_entities_boundary(domain, dim - 1, neumann_boundary)

bcs = [dfx.fem.dirichletbc(np.zeros((dim,)), fixed_boundary, V)]

# ------------ #
# PRECICE INIT #
# ------------ #
participant = Adapter(MPI_COMM, PARTICIPANT_CONFIG)
participant.initialize(coupling_boundary, V, u)
dt = participant.dt

# ------------------------ #
# linear elastic equations #
# ------------------------ #
E = dfx.fem.Constant(domain, E)
nu = dfx.fem.Constant(domain, NU)
rho = dfx.fem.Constant(domain, RHO)

lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
mu = E / 2 / (1 + nu)


def epsilon(v):
    return ufl.sym(ufl.grad(v))


def sigma(v):
    return lmbda * ufl.tr(epsilon(v)) * ufl.Identity(dim) + 2 * mu * epsilon(v)


# ------------------- #
# Time discretization #
# ------------------- #
# prev time step
u_old = dfx.fem.Function(V)
v_old = dfx.fem.Function(V)
a_old = dfx.fem.Function(V)

# current time step
a_new = dfx.fem.Function(V)
v_new = dfx.fem.Function(V)

beta = dfx.fem.Constant(domain, BETA_)
gamma = dfx.fem.Constant(domain, GAMMA_)

dx = ufl.Measure("dx", domain=domain)

a = (1 / (beta * dt**2)) * (u - u_old - dt * v_old) - ((1 - 2 * beta) / (2 * beta)) * a_old
a_expr = dfx.fem.Expression(a, V.element.interpolation_points())

v = v_old + dt * ((1 - gamma) * a_old + gamma * a)
v_expr = dfx.fem.Expression(v, V.element.interpolation_points())

# ------------------ #
# mass, a stiffness  #
# ------------------ #
u_ = ufl.TestFunction(V)
du = ufl.TrialFunction(V)


def mass(u, u_):
    return rho * ufl.dot(u, u_) * dx


def stiffness(u, u_):
    return ufl.inner(sigma(u), epsilon(u_)) * dx


Residual = mass(a, u_) + stiffness(u, u_)
Residual_du = ufl.replace(Residual, {u: du})
a_form = ufl.lhs(Residual_du)
L_form = ufl.rhs(Residual_du)


problem = DiscreteLinearProblem(
    a=a_form, L=L_form, u=u, bcs=bcs, point_dofs=participant.interface_dof
)


# parameters for Time-Stepping
t = 0.0

while participant.is_coupling_ongoing():
    if participant.requires_writing_checkpoint():  # write checkpoint
        participant.store_checkpoint((u_old, v_old, a_old, t))

    read_data = participant.read_data()

    u = problem.solve(read_data)

    # Write new displacements to preCICE
    participant.write_data(u)

    # Call to advance coupling, also returns the optimum time step value
    participant.advance(dt)

    # Either revert to old step if timestep has not converged or move to next timestep
    if participant.requires_reading_checkpoint():
        (
            u_cp,
            v_cp,
            a_cp,
            t_cp,
        ) = participant.retrieve_checkpoint()
        u_old.vector.copy(u_cp.vector)
        v_old.vector.copy(v_cp.vector)
        a_old.vector.copy(a_cp.vector)
        t = t_cp
    else:
        v_new.interpolate(v_expr)
        a_new.interpolate(a_expr)
        u.vector.copy(u_old.vector)
        v_new.vector.copy(v_old.vector)
        a_new.vector.copy(a_old.vector)
        t += dt

    if participant.is_time_window_complete():
        f.vector[participant.interface_dof] = read_data
        f.vector.assemble()
        WRITER.write_function(u, t)
        WRITER.write_function(f, t)
        WRITER.close()

WRITER.close()
participant.finalize()
