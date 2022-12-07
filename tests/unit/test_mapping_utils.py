import unittest
from unittest.mock import MagicMock
from unittest import TestCase
import numpy as np
from mpi4py import MPI
from dolfinx.mesh import create_unit_square
from dolfinx.fem import Function, FunctionSpace, VectorFunctionSpace


class TestAdapterCore(TestCase):
    def test_precompute_eval_vertices(self):
        """
        Test cell collision computation for function evaluation on vertices
        """
        # TODO: handle preCIE using N*2 arrays and not N*3 in 2D simulations
        from fenicsxprecice.mapping_utils import precompute_eval_vertices

        mesh = create_unit_square(MPI.COMM_WORLD, 2, 2)  # create dummy mesh
        precice_vertices = np.array([[0.5, 0.5, 0.0],
                                     [0.2, 0.2, 0.0],
                                     [1.0, 1.0, 0.0]])

        # eval_pos is a copy of precice_vertices but with padded 0s in the z-dim
        eval_pos, cells = precompute_eval_vertices(precice_vertices, mesh)

        # scalar valued
        V = FunctionSpace(mesh, ('P', 2))  # Create function space using mesh

        fenicsx_function = Function(V)
        fenicsx_function.interpolate(lambda x: x[0] + x[1] * x[1])

        expected_data = np.array([0.75, 0.24, 2.0]).reshape((3, 1))

        data = fenicsx_function.eval(eval_pos, cells)
        np.testing.assert_almost_equal(data, expected_data)

        # Vector valued
        V = VectorFunctionSpace(mesh, ('P', 2))

        fenicsx_function = Function(V)
        fenicsx_function.interpolate(lambda x: (x[0] + x[1] * x[1], x[0]))

        expected_data = np.array([[0.75, 0.50], [0.24, 0.20], [2.0, 1.0]])

        data = fenicsx_function.eval(eval_pos, cells)
        np.testing.assert_almost_equal(data, expected_data)
