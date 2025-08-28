"""
Problem setup for partitioned-heat-conduction/fenicsx tutorial
"""
import dolfinx.io.gmshio
from dolfinx.mesh import DiagonalType, create_box
import dolfinx.mesh
from my_enums import DomainPart
import numpy as np
from mpi4py import MPI
import gmsh

y_bottom, y_top = 0, 1
x_left, x_right = 0, 2
z_bottom, z_top = 0, 1
x_coupling = 1.0  # x coordinate of coupling interface


def exclude_straight_boundary(x):
    tol = 1E-14
    return np.logical_or(~np.isclose(x[0], x_coupling, tol),
                         np.logical_or(np.logical_or(np.isclose(x[1], y_top, tol), np.isclose(x[1], y_bottom, tol)),
                         np.logical_or(np.isclose(x[2], z_top, tol), np.isclose(x[2], z_bottom, tol)))

    )


def straight_boundary(x):
    tol = 1E-14
    return np.isclose(x[0], x_coupling, tol)


def get_geometry(domain_part):
    nx = 5
    ny = 5
    nz = 5

    if domain_part is DomainPart.LEFT:
        p0 = (x_left, y_bottom, z_bottom)
        p1 = (x_coupling, y_top, z_top)
    elif domain_part is DomainPart.RIGHT:
        p0 = (x_coupling, y_bottom, z_bottom)
        p1 = (x_right, y_top, z_top)
    else:
        raise Exception("invalid domain_part: {}".format(domain_part))
    mesh = create_box(MPI.COMM_WORLD, [np.asarray(p0), np.asarray(p1)],[nx, ny, nz], dolfinx.mesh.CellType.tetrahedron)
    coupling_boundary = straight_boundary
    remaining_boundary = exclude_straight_boundary

    return mesh, coupling_boundary, remaining_boundary, [(p0[0]-1e-14, p0[1]-1e-14, p0[2]-1e-14),(p1[0]+1e-14, p1[1]+1e-14, p1[2]+1e-14)]

def get_complex_geometry(domain_part):
    gmsh.initialize()
    gmsh.model.add("HeatMesh")
    
    if domain_part is DomainPart.LEFT:
        # outer
        def coupling_bc(x):
            tol = 1E-14
            top = np.logical_and.reduce((
                np.isclose(x[1], 0.75, atol=tol),
                np.logical_or(x[0] >= 1, np.isclose(x[0], 1, tol)),
                np.logical_or(x[2] >= 0.5, np.isclose(x[2], 0.5, tol)),
                np.logical_or(x[2] <= 1.5, np.isclose(x[2], 1.5, tol))
            ))

            bot = np.logical_and.reduce((
                np.isclose(x[1], 0.25, tol),
                np.logical_or(x[0] >= 1, np.isclose(x[0], 1, tol)),
                np.logical_or(x[2] >= 0.5, np.isclose(x[2], 0.5, tol)),
                np.logical_or(x[2] <= 1.5, np.isclose(x[2], 1.5, tol))
            ))

            sideCenter = np.logical_and.reduce((
                np.isclose(x[0], 1, tol),
                x[1] >= 0.25,
                x[1] <= 0.75,
                x[2] >= 0.5,
                x[2] <= 1.5
            ))

            sideLeft = np.logical_and.reduce((
                x[0] >= 1,
                x[1] >= 0.25,
                x[1] <= 0.75,
                np.isclose(x[2], 0.5, tol)
            ))

            sideRight = np.logical_and.reduce((
                x[0] >= 1,
                x[1] >= 0.25,
                x[1] <= 0.75,
                np.isclose(x[2], 1.5, tol)
                ))

            return np.logical_or.reduce((top, bot, sideCenter, sideLeft, sideRight))
        
        def boundary_bc(x):
            tol = 1E-14
            return np.logical_and(np.logical_or(np.logical_or(np.logical_or(np.logical_or(np.logical_or(np.isclose(x[0], 0, tol), np.isclose(x[0], 2, tol)), np.isclose(x[1], 0, tol)), np.isclose(x[1], 1, tol)), np.isclose(x[2], 0, tol)), np.isclose(x[2], 2, tol)), ~np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.isclose(x[0], 2, tol), x[1] >= 0.25), x[1] <= 0.75), x[2] >= 0.5), x[2] <= 1.5))
        
        gmsh.model.occ.addBox(0,0,0,2,1,2,1)
        gmsh.model.occ.synchronize()
        gmsh.model.occ.addBox(1,0.25,.5,1,.5,1, 2)
        gmsh.model.occ.synchronize()
        
        _, _ = gmsh.model.occ.cut([(3, 1)], [(3, 2)], 3, True, True)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [3], 4, "foo")
        gmsh.model.mesh.setOrder(2)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.15)
        gmsh.model.mesh.generate(3)
        outer_mesh, _, _ = dolfinx.io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, 3)
        gmsh.finalize()
        return outer_mesh, coupling_bc, boundary_bc, [(0, 0, 0), (2+1e-14, 1+1e-14, 2+1e-14)]
    else:
        # inner
        
        def coupling_bc(x):
            tol = 1E-14
            return np.logical_or(
                    np.logical_or(np.isclose(x[1], 0.75, tol), np.isclose(x[1], 0.25, tol)),
                    np.logical_or(np.isclose(x[0], 1, tol), 
                                  np.logical_or(np.isclose(x[2], 0.5, tol), np.isclose(x[2], 1.5, tol))))
        
        def boundary_bc(x):
            tol = 1E-14
            return np.isclose(x[0], 2, tol)
        
        
        gmsh.model.occ.addBox(1,0.25,.5,1,.5, 1, 1)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [1], 2, "foo")
        gmsh.model.mesh.setOrder(2)
        gmsh.model.mesh.generate(3)
        inner_mesh, _, _ = dolfinx.io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, 3)
        gmsh.finalize()
        return inner_mesh, coupling_bc, boundary_bc, [(1, 0.25, 0.5), (2+1e-14, 0.75+1e-14, 1.5+1e-14)]
        
        
    
