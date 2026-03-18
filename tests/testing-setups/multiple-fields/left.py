from mpi4py import MPI
from dolfinx import mesh
from dolfinx.fem import functionspace
from dolfinx import fem
import numpy
import ufl
from dolfinx import default_scalar_type
from dolfinx.fem.petsc import LinearProblem

import fenicsxprecice

domain1 = mesh.create_rectangle(
    MPI.COMM_WORLD, [
        numpy.asarray((0, 0)),
        numpy.asarray((1, 1))],
    [10, 10], mesh.CellType.quadrilateral)
domain2 = mesh.create_rectangle(
    MPI.COMM_WORLD, [
        numpy.asarray((0, 0)),
        numpy.asarray((1, 1))],
    [11, 11], mesh.CellType.quadrilateral)
V1 = functionspace(domain1, ("Lagrange", 1))
V2 = functionspace(domain2, ("Lagrange", 1))
uD1 = fem.Function(V1)
uD1.interpolate(lambda x: 1 + x[0])
uDD1 = fem.Function(V1)
uDD1.interpolate(lambda x: 100 + x[0])
uD2 = fem.Function(V2)
uD2.interpolate(lambda x: 2 + x[0])


def coupling_bc(x):
    tol = 1E-14
    return numpy.isclose(x[0], 1, tol)


precice = fenicsxprecice.Adapter(adapter_config_filename="precice-adapter-config-L.json", mpi_comm=MPI.COMM_SELF)
leftOne = fenicsxprecice.CouplingMesh("LeftOne", coupling_bc,
                                      {"LeftInOne": V1},
                                      {"LeftOutOne": uD1, "LeftOutTwo": uDD1})
leftTwo = fenicsxprecice.CouplingMesh("LeftTwo", coupling_bc, {"LeftInTwo": V2}, {"LeftOutThree": uD2})
precice.initialize([leftOne, leftTwo])

while precice.is_coupling_ongoing():

    if precice.requires_writing_checkpoint():
        precice.store_checkpoint(uD1, 0, 0)

    read_data1 = precice.read_data(leftOne.get_name(), "LeftInOne", 0)
    read_data2 = precice.read_data(leftTwo.get_name(), "LeftInTwo", 0)

    precice.write_data(leftOne.get_name(), "LeftOutOne", uD1)
    precice.write_data(leftOne.get_name(), "LeftOutTwo", uDD1)

    precice.advance(1)

    if precice.requires_reading_checkpoint():
        pass


precice.finalize()

print(read_data1)  # expected: x[0]+3
print(read_data2)  # expected: x[0]+4
