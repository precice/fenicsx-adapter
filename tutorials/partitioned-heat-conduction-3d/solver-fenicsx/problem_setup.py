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


def get_geometry(domain_part):
    gmsh.initialize()
    gmsh.model.add("HeatMesh")

    if domain_part is DomainPart.OUTER:
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
            or_part = np.logical_or.reduce((
                np.isclose(x[0], 0, tol),
                np.isclose(x[0], 2, tol),
                np.isclose(x[1], 0, tol),
                np.isclose(x[1], 1, tol),
                np.isclose(x[2], 0, tol),
                np.isclose(x[2], 2, tol)
            ))
            and_part = np.logical_and.reduce((
                np.isclose(x[0], 2, tol),
                x[1] >= 0.25,
                x[1] <= 0.75,
                x[2] >= 0.5,
                x[2] <= 1.5,

            ))
            return np.logical_and(or_part, ~and_part)

        gmsh.model.occ.addBox(0, 0, 0, 2, 1, 2, 1)
        gmsh.model.occ.synchronize()
        gmsh.model.occ.addBox(1, 0.25, .5, 1, .5, 1, 2)
        gmsh.model.occ.synchronize()

        _, _ = gmsh.model.occ.cut([(3, 1)], [(3, 2)], 3, True, True)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [3], 4, "foo")
        gmsh.model.mesh.setOrder(2)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.15)
        gmsh.model.mesh.generate(3)
        outer_mesh, _, _ = dolfinx.io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, 3)
        gmsh.finalize()
        return outer_mesh, coupling_bc, boundary_bc, [(0, 0, 0), (2 + 1e-14, 1 + 1e-14, 2 + 1e-14)]
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

        gmsh.model.occ.addBox(1, 0.25, .5, 1, .5, 1, 1)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [1], 2, "foo")
        gmsh.model.mesh.setOrder(2)
        gmsh.model.mesh.generate(3)
        inner_mesh, _, _ = dolfinx.io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, 3)
        gmsh.finalize()
        return inner_mesh, coupling_bc, boundary_bc, [(1, 0.25, 0.5), (2 + 1e-14, 0.75 + 1e-14, 1.5 + 1e-14)]
