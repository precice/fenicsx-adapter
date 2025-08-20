class CouplingMesh:

    def __init__(self, mesh_name, coupling_boundary, read_fields=None, write_fields=None):
        """

        Args:
            mesh_name (string): Name of the mesh
            coupling_boundary (function): function describing the coupling boundary
            read_fields (dict, optional): A dict mapping a read data field name to a function space. Defaults to None.
            write_fields (dict, optional): A dict mapping a write data field name to a function or function space. Defaults to None.
        """
        self._mesh_name = mesh_name
        self._coupling_boundary = coupling_boundary
        self._read_fields = read_fields
        self._write_fields = write_fields

    def get_name(self):
        return self._mesh_name

    def get_coupling_boundary(self):
        return self._coupling_boundary

    def get_read_fields(self):
        return self._read_fields

    def get_write_fields(self):
        return self._write_fields
