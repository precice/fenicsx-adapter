from mpi4py import MPI
from dolfinx import mesh
from dolfinx.fem import functionspace
from dolfinx import fem
import numpy
import ufl
from dolfinx import default_scalar_type
from dolfinx.fem.petsc import LinearProblem

import fenicsxprecice

domain = mesh.create_rectangle(
    MPI.COMM_WORLD, [
        numpy.asarray(
            (0, 0)), numpy.asarray(
                (1, 1))], [
                    10, 10], mesh.CellType.quadrilateral)
domain2 = mesh.create_rectangle(
    MPI.COMM_WORLD, [
        numpy.asarray(
            (0, 0)), numpy.asarray(
                (1, 1))], [
                    11, 11], mesh.CellType.quadrilateral)
V = functionspace(domain, ("Lagrange", 1))
V2 = functionspace(domain2, ("Lagrange", 1))
uD = fem.Function(V)
uD.interpolate(lambda x: 1 + x[0])
uD2 = fem.Function(V2)
uD2.interpolate(lambda x: 2 + x[0])


def coupling_bc(x):
    tol = 1E-14
    return numpy.isclose(x[0], 1, tol)


precice = fenicsxprecice.Adapter(adapter_config_filename="precice-adapter-config-L.json", mpi_comm=MPI.COMM_SELF)
precice.initialize({precice.Meshes.Left1: [coupling_bc, V, uD],
                    precice.Meshes.Left2: [coupling_bc, V2, uD2]})

coupling_expression1 = precice.create_coupling_expression(precice.Meshes.Left1)
coupling_expression2 = precice.create_coupling_expression(precice.Meshes.Left2)

while precice.is_coupling_ongoing():

    if precice.requires_writing_checkpoint():
        precice.store_checkpoint(uD, 0, 0)

    read_data1 = precice.read_data(0, precice.Meshes.Left1)
    read_data2 = precice.read_data(0, precice.Meshes.Left2)

    precice.write_data(uD, precice.Meshes.Left1)
    precice.write_data(uD2, precice.Meshes.Left2)

    precice.advance(1)

    if precice.requires_reading_checkpoint():
        pass


precice.finalize()

print(read_data1)  # expected: x[0]+3
print(read_data2)  # expected: x[0]+4
