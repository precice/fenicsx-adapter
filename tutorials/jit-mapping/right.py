from mpi4py import MPI
from dolfinx import mesh
from dolfinx.fem import functionspace
from dolfinx import fem, geometry
import numpy

import fenicsxprecice

def coupling_bc(x):
    tol = 1E-14
    return numpy.isclose(x[0], 1, tol)

def test_read(x):
    tmp = numpy.transpose(x)[:, :2]
    for c, i in zip(tmp, range(len(tmp))):
        if c[1] < 0:
            tmp[i][1] = 1e-17
        if c[0] <= 1:
            tmp[i][0] = 1 + 1e-17
        if c[1] >= 1:
            tmp[i][1] = 1 - 1e-17
        if c[0] >= 2:
            tmp[i][0] = 2 - 1e-17
    lst =  list(precice.read_data_at_coordinates("LeftMesh", "LeftValue", tmp, 0).values())
    return lst

domain1 = mesh.create_rectangle(
    MPI.COMM_WORLD, [
        numpy.asarray((1, 0)),
        numpy.asarray((2, 1))],
    [14, 14], mesh.CellType.quadrilateral)
V1 = functionspace(domain1, ("Lagrange", 2))
uD = fem.Function(V1)
uD.interpolate(lambda x: x[0]+x[1]-10)

V_boundary = functionspace(domain1, ("Lagrange", 2)) # or V1
u_boundary = fem.Function(V_boundary)
tdim = domain1.topology.dim
coupling_cells = mesh.locate_entities(domain1, tdim, lambda x: numpy.isclose(x[0], 1, 1/14))


precice = fenicsxprecice.Adapter(adapter_config_filename="precice-adapter-config-R.json", mpi_comm=MPI.COMM_SELF)
cmesh = fenicsxprecice.CouplingMesh("RightMesh", coupling_bc, {"LeftValue": V1}, {"RightValue": uD})
precice.set_mesh_access_region("LeftMesh", [(1,0), (2,1)])
precice.initialize([cmesh])

coupling_expression = precice.create_coupling_expression(cmesh.get_name())


dofs_coupling = fem.locate_dofs_geometrical(V_boundary, coupling_bc)
dofs_coupling_coordinates = V_boundary.tabulate_dof_coordinates()[dofs_coupling]


coords = dofs_coupling_coordinates[:,:2]
for c, i in zip(coords, range(len(coords))):
    if c[1] < 0:
        coords[i][1] = 1e-17
    if c[0] <= 1:
        coords[i][0] = 1 + 1e-17
    if c[1] >= 1:
        coords[i][1] = 1 - 1e-17
    if c[0] >= 2:
        coords[i][0] = 2 - 1e-17

while precice.is_coupling_ongoing():

    if precice.requires_writing_checkpoint():
        precice.store_checkpoint(uD, 0, 0)

    read_data = precice.read_data_at_coordinates("LeftMesh", "LeftValue", coords, 0)
    precice.write_data(cmesh.get_name(), "RightValue", uD)
    
    u_boundary.x.array[dofs_coupling] = list(read_data.values())
    #u_boundary.interpolate(test_read) # does not work
    #u_boundary.interpolate(test_read, coupling_cells) # does not work
    
    precice.advance(0.25)

    if precice.requires_reading_checkpoint():
        pass


precice.finalize()
# check correctness (error expected to be relatively high because the other participant has a different mesh ;) )
# expected: x[0]+x[1]+1
max_diff = 0
for key in read_data.keys():
    diff = key[0]+key[1]**2+1 - read_data[key]
    diff = abs(diff)
    if diff > max_diff:
        max_diff = diff
    
print(max_diff)


bb_tree = geometry.bb_tree(domain1, domain1.geometry.dim)

cells = []
points = []
# Find cells whose bounding-box collide with the the points
cell_candidates = geometry.compute_collisions_points(bb_tree, dofs_coupling_coordinates)
# Choose one of the cells that contains the point
colliding_cells = geometry.compute_colliding_cells(domain1, cell_candidates, dofs_coupling_coordinates)
for i, point in enumerate(dofs_coupling_coordinates):
    if len(colliding_cells.links(i)) > 0:
        points.append(point)
        cells.append(colliding_cells.links(i)[0])
precice_data = u_boundary.eval(points, cells)

print(precice_data.T-list(read_data.values())) # too much difference (its just an interpolation??)
