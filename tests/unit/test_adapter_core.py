from unittest.mock import MagicMock
from unittest import TestCase
import numpy as np
from mpi4py import MPI
from dolfinx.mesh import create_unit_square
from dolfinx.fem import Function, FunctionSpace, VectorFunctionSpace


class TestAdapterCore(TestCase):
    def test_convert_scalar_fenicsx_to_precice(self):
        """
        Test conversion from function to write_data for scalar
        """
        from fenicsxprecice.adapter_core import convert_fenicsx_to_precice
        from sympy import lambdify, symbols

        mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)  # create dummy mesh

        # scalar valued
        V = FunctionSpace(mesh, ('P', 2))  # Create function space using mesh
        x, y = symbols('x[0], x[1]')
        fun_sym = y + x * x
        fun_lambda = lambdify([x, y], fun_sym)

        class my_expression():
            def __call__(self, x):
                return fun_lambda(x[0], x[1])

        fenicsx_function = Function(V)
        fenicsx_function.interpolate(my_expression())

        local_ids = []
        manual_sampling = []
        for i in range(mesh.geometry.x.shape[0]):
            v = mesh.geometry.x[i]
            local_ids.append(i)
            manual_sampling.append([fun_lambda(v[0], v[1])])
        manual_sampling = np.array(manual_sampling).squeeze()

        data = convert_fenicsx_to_precice(fenicsx_function, local_ids)

        np.testing.assert_allclose(data, manual_sampling, atol=10**-16)
