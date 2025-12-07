"""
This is the configuration module of fenicsxadapter
"""

import json
import os


class Config:
    """
    Handles reading of config. parameters of the fenicsxadapter based on JSON
    configuration file. Initializer calls read_json() method. Instance attributes
    can be accessed by provided getter functions.
    """

    def __init__(self, adapter_config_filename):

        self._config_file_name = None
        self._participant_name = None
        self._meshes = {}

        self.read_json(adapter_config_filename)

    def read_json(self, adapter_config_filename):
        """
        Reads JSON adapter configuration file and saves the data to the respective instance attributes.

        Parameters
        ----------
        adapter_config_filename : string
            Name of the JSON configuration file
        """

        def has_duplicate_data_field_name(l: list, key: str):
            """Checks if the given list of dicts has duplicate values for the same key

            Returns:
                bool: True, if the list has a duplicate
            """
            visited_names = set()
            for data_field in l:
                if data_field[key] in visited_names:
                    return True
                visited_names.add(data_field[key])
            return False

        folder = os.path.dirname(os.path.join(os.getcwd(), adapter_config_filename))
        path = os.path.join(folder, os.path.basename(adapter_config_filename))
        read_file = open(path, "r")
        data = json.load(read_file)
        self._config_file_name = os.path.join(folder, data["precice_config_file_path"])
        self._participant_name = data["participant_name"]

        for interface in data["interfaces"]:
            mesh_name = interface["mesh_name"]
            self._meshes[mesh_name] = {}
            try:
                self._meshes[mesh_name]["write_data"] = interface["write_data"]
                # check for duplicates
                assert not has_duplicate_data_field_name(self._meshes[mesh_name]["write_data"], "name"), \
                    "Invalid config file: Multiple write data fields have the same name"
            except KeyError:
                # not required for one-way coupling, if this participant reads data
                self._meshes[mesh_name]["write_data"] = None

            try:
                self._meshes[mesh_name]["read_data"] = interface["read_data"]
                # check for duplicates
                assert not has_duplicate_data_field_name(self._meshes[mesh_name]["read_data"], "name"), \
                    "Invalid config file: Multiple read data fields have the same name"
            except KeyError:
                # not required for one-way coupling, if this participant writes data
                self._meshes[mesh_name]["read_data"] = None

        read_file.close()

    def get_config_file_name(self):
        return self._config_file_name

    def get_participant_name(self):
        return self._participant_name

    def get_read_data_names(self, mesh_name):
        """
        Parameters
        ----------
        mesh_name : Name of mesh

        """
        return [read_fields["name"] for read_fields in self._meshes[mesh_name]["read_data"]]

    def get_write_data_names(self, mesh_name):
        """
        Parameters
        ----------
        mesh_name : Name of mesh

        """
        return [write_fields["name"] for write_fields in self._meshes[mesh_name]["write_data"]]
