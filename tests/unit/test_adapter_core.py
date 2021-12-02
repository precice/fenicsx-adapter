from unittest.mock import MagicMock
from unittest import TestCase
import numpy as np
from dolfinx import Function, UnitSquareMesh
from dolfinx.fem import FunctionSpace, VectorFunctionSpace


class TestAdapterCore(TestCase):
    def test_get_coupling_boundary_edges(self):
        """
        Test coupling edge detection
        """
        from fenicsxprecice.adapter_core import get_coupling_boundary_edges

        def right_edge(x):
            tol = 1E-14
            return abs(x[0] - 1) < tol

        mesh = UnitSquareMesh(10, 10)  # create dummy mesh
        V = FunctionSpace(mesh, 'P', 2)  # Create function space using mesh
        id_mapping = MagicMock()  # a fake id_mapping returning dummy values

        global_ids = []
        for v in mesh.geometry.x:
            if right_edge(v):
                global_ids.append(v.global_index())

        edge_vertex_ids1, edge_vertex_ids2 = get_coupling_boundary_edges(V, right_edge, global_ids, id_mapping)

        self.assertEqual(len(edge_vertex_ids1), 10)
        self.assertEqual(len(edge_vertex_ids2), 10)

    def test_convert_fenicsx_to_precice(self):
        """
        Test conversion from function to write_data
        """
        from fenicsxprecice.adapter_core import convert_fenicsx_to_precice
        from sympy import lambdify, symbols, printing

        mesh = UnitSquareMesh(10, 10)  # create dummy mesh

        # scalar valued
        V = FunctionSpace(mesh, 'P', 2)  # Create function space using mesh
        x, y = symbols('x[0], x[1]')
        fun_sym = y + x * x
        fun_lambda = lambdify([x, y], fun_sym)
        fenicsx_function = Function(V)
        fenicsx_function.interpolate(fun_lambda)

        local_ids = []
        manual_sampling = []
        for v in mesh.geometry.x:
            local_ids.append(v.index())
            manual_sampling.append(fun_lambda(v.x(0), v.x(1)))

        data = convert_fenicsx_to_precice(fenicsx_function, local_ids)

        np.testing.assert_allclose(data, manual_sampling)

        # vector valued
        W = VectorFunctionSpace(mesh, ('P', 2))  # Create function space using mesh
        fun_sym_x = y + x * x
        fun_sym_y = y * y + x * x * x * 2
        fun_lambda = lambdify([x, y], [fun_sym_x, fun_sym_y])
        fenicsx_function = Function(V)
        fenicsx_function.interpolate(fun_lambda)

        local_ids = []
        manual_sampling = []
        for v in mesh.geometry.x:
            local_ids.append(v.index())
            manual_sampling.append(fun_lambda(v.x(0), v.x(1)))

        data = convert_fenicsx_to_precice(fenicsx_function, local_ids)

        np.testing.assert_allclose(data, manual_sampling)
