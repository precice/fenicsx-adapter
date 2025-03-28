from unittest.mock import MagicMock, patch
from unittest import TestCase
from tests import MockedPrecice
from dolfinx import fem, mesh as msh
import basix
from mpi4py import MPI
import numpy as np

x_left, x_right = 0, 1
y_bottom, y_top = 0, 1


def right_boundary(x):
    tol = 1E-14
    return abs(x[0] - x_right) < tol


def scalar_expr(x): return x[0] * x[0] + x[1] * x[1]
def vector_expr(x): return (x[0] + x[1] * x[1], x[0] - x[1] * x[1])


@patch.dict('sys.modules', {'precice': MockedPrecice})
class TestWriteandReadData(TestCase):
    """
    Test suite to test read and write functionality of Adapter. Read and Write functionality is tested for both scalar
    and vector data.
    """
    dummy_config = "tests/precice-adapter-config.json"

    mesh = msh.create_unit_square(MPI.COMM_WORLD, 10, 10)
    dimension = 2

    scalar_V = fem.functionspace(mesh, ("P", 2))
    scalar_function = fem.Function(scalar_V)
    scalar_function.interpolate(scalar_expr)

    vector_elem = basix.ufl.element("P", mesh.topology.cell_name(), 2, shape=(mesh.geometry.dim,))
    vector_V = fem.functionspace(mesh, vector_elem)
    vector_function = fem.Function(vector_V)
    vector_function.interpolate(vector_expr)

    n_vertices = 11
    fake_data_name = 'fake_data'
    vertices_x = [x_right for _ in range(n_vertices)]
    vertices_y = np.linspace(y_bottom, y_top, n_vertices)

    def test_scalar_write(self):
        """
        Test to check if Adapter function write() passes correct parameters to the API function write_data()
        """
        from precice import Participant
        import fenicsxprecice

        Participant.write_data = MagicMock()
        Participant.get_mesh_dimensions = MagicMock(return_value=self.dimension)
        Participant.set_mesh_vertices = MagicMock(return_value=np.arange(self.n_vertices))
        Participant.set_mesh_edge = MagicMock()
        Participant.initialize = MagicMock()
        Participant.requires_initial_data = MagicMock(return_value=False)
        Participant.initialize_data = MagicMock()

        precice = fenicsxprecice.Adapter(MPI.COMM_WORLD, self.dummy_config)
        precice._participant = Participant(None, None, None, None)
        precice.initialize({"Dummy-Mesh": [right_boundary, self.scalar_V, self.scalar_function]})

        precice.write_data("Dummy-Mesh", self.scalar_function)

        expected_data_name = self.fake_data_name
        expected_values = np.array([[scalar_expr([x_right, y])] for y in self.vertices_y])
        expected_vertex_ids = np.arange(self.n_vertices)
        expected_args = [expected_data_name, expected_vertex_ids, expected_values]

        for arg, expected_arg in zip(Participant.write_data.call_args[1], expected_args):
            if isinstance(arg, int):
                self.assertTrue(arg == expected_arg)
            elif isinstance(arg, np.ndarray):
                expected_arg = expected_arg.reshape(arg.shape)
                np.testing.assert_allclose(arg, expected_arg)

    def test_scalar_read(self):
        """
        Test to check if Adapter function read() passes correct parameters to the API function read_data()
        Test to check if data return by API function read_data() is also returned by Adapter function read()
        """
        from precice import Participant
        import fenicsxprecice

        def return_dummy_data(n_points):
            data = np.arange(n_points)
            return data

        Participant.read_data = MagicMock(return_value=return_dummy_data(self.n_vertices))
        Participant.get_mesh_dimensions = MagicMock(return_value=self.dimension)
        Participant.set_mesh_vertices = MagicMock(return_value=np.arange(self.n_vertices))
        Participant.set_mesh_edge = MagicMock()
        Participant.initialize = MagicMock()
        Participant.requires_initial_data = MagicMock(return_value=False)
        Participant.initialize_data = MagicMock()

        precice = fenicsxprecice.Adapter(MPI.COMM_WORLD, self.dummy_config)
        precice._participant = Participant(None, None, None, None)
        precice._read_data_name = self.fake_data_name
        precice.initialize({"Dummy-Mesh": [right_boundary, self.scalar_V, None]})

        read_data = precice.read_data("Dummy-Mesh", 0)

        expected_data_name = self.fake_data_name
        expected_vertex_ids = np.arange(self.n_vertices)
        expected_args = [expected_data_name, expected_vertex_ids]

        for arg, expected_arg in zip(Participant.read_data.call_args[0], expected_args):
            if isinstance(arg, int):
                self.assertTrue(arg == expected_arg)
            elif isinstance(arg, np.ndarray):
                np.testing.assert_allclose(arg, expected_arg)

        np.testing.assert_almost_equal(list(read_data.values()), return_dummy_data(self.n_vertices))
