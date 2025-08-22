from mpi4py import MPI
from dolfinx import mesh
from dolfinx.fem import functionspace
from dolfinx import fem
import numpy

import fenicsxprecice

domain1 = mesh.create_rectangle(
    MPI.COMM_WORLD, [
        numpy.asarray((0, 0)),
        numpy.asarray((1, 1))],
    [10, 10], mesh.CellType.quadrilateral)
V1 = functionspace(domain1, ("Lagrange", 2))
uD = fem.Function(V1)
uD.interpolate(lambda x: x[0] + x[1]**2 + 1)


def coupling_bc(x):
    tol = 1E-14
    return numpy.isclose(x[0], 1, tol)


precice = fenicsxprecice.Adapter(adapter_config_filename="precice-adapter-config-L.json", mpi_comm=MPI.COMM_SELF)
cmesh = fenicsxprecice.CouplingMesh("LeftMesh", coupling_bc, {"RightValue": V1}, {"LeftValue": uD})
precice.set_mesh_access_region("RightMesh", [(0, 0), (1, 1)])
precice.initialize([cmesh])


coupling_expression = precice.create_coupling_expression(cmesh.get_name())

dofs_coupling = fem.locate_dofs_geometrical(V1, coupling_bc)
dofs_coupling_coordinates = V1.tabulate_dof_coordinates()[dofs_coupling]

coords = dofs_coupling_coordinates[:, :2]
for c, i in zip(coords, range(len(coords))):
    if c[0] < 0:
        coords[i][0] = 1e-17
    if c[1] < 0:
        coords[i][1] = 1e-17
    if c[0] >= 1:
        coords[i][0] = 1 - 1e-17
    if c[1] >= 1:
        coords[i][1] = 1 - 1e-17


while precice.is_coupling_ongoing():

    if precice.requires_writing_checkpoint():
        precice.store_checkpoint(uD, 0, 0)

    read_data = precice.read_data_at_coordinates("RightMesh", "RightValue", coords, 0)
    precice.write_data(cmesh.get_name(), "LeftValue", uD)

    precice.advance(0.25)

    if precice.requires_reading_checkpoint():
        pass


precice.finalize()
# check correctness
# expected: x[0]+x[1]-10
max_diff = 0
for key in read_data.keys():
    diff = key[0] + key[1] - 10 - read_data[key]
    diff = abs(diff)
    if diff > max_diff:
        max_diff = diff

print(max_diff)
