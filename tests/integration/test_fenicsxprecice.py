# comments on test layout: https://docs.pytest.org/en/latest/goodpractices.html

from unittest.mock import MagicMock, patch
from unittest import TestCase
from tests import MockedPrecice
import numpy as np
from mpi4py import MPI
from dolfinx.mesh import create_unit_square
from dolfinx.fem import Function, FunctionSpace, VectorFunctionSpace


class MockedArray:
    """
    mock of dolfinx.Function
    """

    def __init__(self):
        self.value = MagicMock()

    def assign(self, new_value):
        """
        mock of dolfinx.Function.assign
        :param new_value:
        :return:
        """
        self.value = new_value.value

    def copy(self):
        returned_array = MockedArray()
        returned_array.value = self.value
        return returned_array

    def value_rank(self):
        return 0


@patch.dict('sys.modules', {'precice': MockedPrecice})
class TestAdapter(TestCase):
    """
    Test suite for basic API functions
    """

    def test_version(self):
        """
        Test that adapter provides a version
        """
        import fenicsxprecice
        fenicsxprecice.__version__


@patch.dict('sys.modules', {'precice': MockedPrecice})
class TestCheckpointing(TestCase):
    """
    Test suite to check if Checkpointing functionality of the Adapter is working.
    """
    dt = 1  # timestep size
    n = 0  # current iteration count
    t = 0  # current time
    u_n_mocked = MockedArray()  # result at the beginning of the timestep
    u_np1_mocked = MockedArray()  # newly computed result
    write_function_mocked = MockedArray()
    u_cp_mocked = MockedArray()  # value of the checkpoint
    t_cp_mocked = t  # time for the checkpoint
    n_cp_mocked = n  # iteration count for the checkpoint
    dummy_config = "tests/precice-adapter-config.json"

    # todo if we support multirate, we should use the lines below for checkpointing
    # for the general case the checkpoint u_cp (and t_cp and n_cp) can differ from u_n and u_np1
    # t_cp_mocked = MagicMock()  # time for the checkpoint
    # n_cp_mocked = nMagicMock()  # iteration count for the checkpoint

    def test_checkpoint_mechanism(self):
        """
        Test correct checkpoint storing
        """
        import fenicsxprecice
        from precice import Interface, action_write_iteration_checkpoint

        def is_action_required_behavior(py_action):
            if py_action == action_write_iteration_checkpoint():
                return True
            else:
                return False

        Interface.initialize = MagicMock(return_value=self.dt)
        Interface.is_action_required = MagicMock(side_effect=is_action_required_behavior)
        Interface.get_dimensions = MagicMock()
        Interface.get_mesh_id = MagicMock()
        Interface.get_data_id = MagicMock()
        Interface.mark_action_fulfilled = MagicMock()
        Interface.is_time_window_complete = MagicMock(return_value=True)
        Interface.advance = MagicMock()

        precice = fenicsxprecice.Adapter(MPI.COMM_WORLD, self.dummy_config)

        precice.store_checkpoint(self.u_n_mocked, self.t, self.n)

        # Replicating control flow where implicit iteration has not converged and solver state needs to be restored
        # to a checkpoint
        precice.advance(self.dt)
        Interface.is_time_window_complete = MagicMock(return_value=False)

        # Check if the checkpoint is stored correctly in the adapter
        self.assertEqual(precice.retrieve_checkpoint() == self.u_n_mocked, self.t, self.n)


@patch.dict('sys.modules', {'precice': MockedPrecice})
class TestExpressionHandling(TestCase):
    """
    Test Expression creation and updating mechanism based on data provided by user.
    """
    dummy_config = "tests/precice-adapter-config.json"

    mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)
    dimension = 2

    def scalar_expr(x): return x[0] + x[1]
    scalar_V = FunctionSpace(mesh, ("P", 1))
    scalar_function = Function(scalar_V)
    scalar_function.interpolate(scalar_expr)

    n_vertices = 11
    fake_id = 15
    vertices_x = [1 for _ in range(n_vertices)]
    vertices_y = np.linspace(0, 1, n_vertices)
    vertex_ids = np.arange(n_vertices)

    n_samples = 1000
    samplepts_x = [1 for _ in range(n_samples)]
    samplepts_y = np.linspace(0, 1, n_samples)

    def test_update_expression_scalar(self):
        """
        Check if a sampling of points on a dolfinx Function interpolated via FEniCSx is matching with the sampling of the
        same points on a FEniCSx Expression created by the Adapter
        """
        from precice import Interface
        import fenicsxprecice
        Interface.get_dimensions = MagicMock(return_value=2)
        Interface.set_mesh_vertices = MagicMock(return_value=self.vertex_ids)
        Interface.get_mesh_id = MagicMock()
        Interface.get_data_id = MagicMock()
        Interface.set_mesh_edge = MagicMock()
        Interface.initialize = MagicMock()
        Interface.initialize_data = MagicMock()
        Interface.is_action_required = MagicMock()
        Interface.mark_action_fulfilled = MagicMock()
        Interface.write_block_scalar_data = MagicMock()

        def right_boundary(x): return abs(x[0] - 1.0) < 10**-14

        precice = fenicsxprecice.Adapter(MPI.COMM_WORLD, self.dummy_config)
        precice._interface = Interface(None, None, None, None)
        precice.initialize(right_boundary, self.scalar_V, self.scalar_function)
        values = np.array([self.scalar_function.eval([x, y, 0], 0)[0]
                           for x, y in zip(self.vertices_x, self.vertices_y)])
        data = {(x, y): v for x, y, v in zip(self.vertices_x, self.vertices_y, values)}
        scalar_coupling_expr = precice.create_coupling_expression()
        precice.update_coupling_expression(scalar_coupling_expr, data)

        expr_samples = np.array([scalar_coupling_expr.eval([x, y, 0], 0)
                                 for x, y in zip(self.samplepts_x, self.samplepts_y)])
        func_samples = np.array([self.scalar_function.eval([x, y, 0], 0)
                                 for x, y in zip(self.samplepts_x, self.samplepts_y)])

        assert (np.allclose(expr_samples, func_samples, 1E-10))
