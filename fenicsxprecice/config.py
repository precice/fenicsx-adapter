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

        def has_duplicate_name(l: list):
            """Checks if the given list has duplicates

            Returns:
                bool: True, if the list has a duplicate
            """
            visited_names = set()
            for name in l:
                if name in visited_names:
                    return True
                visited_names.add(name)
            return False

        folder = os.path.dirname(os.path.join(os.getcwd(), adapter_config_filename))
        path = os.path.join(folder, os.path.basename(adapter_config_filename))
        read_file = open(path, "r")
        data = json.load(read_file)
        self._config_file_name = os.path.join(folder, data["config_file_name"])
        self._participant_name = data["participant_name"]

        for mesh_name in data["interfaces"].keys():
            self._meshes[mesh_name] = {}
            try:
                self._meshes[mesh_name]["write_data_names"] = data["interfaces"][mesh_name]["write_data_names"]
                # check for duplicates
                assert not has_duplicate_name(self._meshes[mesh_name]["write_data_names"]), \
                    "Invalid config file: Multiple write data fields have the same name"
            except KeyError:
                # not required for one-way coupling, if this participant reads data
                self._meshes[mesh_name]["write_data_names"] = None

            try:
                self._meshes[mesh_name]["read_data_names"] = data["interfaces"][mesh_name]["read_data_names"]
                # check for duplicates
                assert not has_duplicate_name(self._meshes[mesh_name]["read_data_names"]), \
                    "Invalid config file: Multiple read data fields have the same name"
            except KeyError:
                # not required for one-way coupling, if this participant writes data
                self._meshes[mesh_name]["read_data_names"] = None

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
        return self._meshes[mesh_name]["read_data_names"]

    def get_write_data_names(self, mesh_name):
        """
        Parameters
        ----------
        mesh_name : Name of mesh

        """
        return self._meshes[mesh_name]["write_data_names"]
