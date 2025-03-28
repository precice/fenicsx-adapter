"""
This is the configuration module of fenicsxadapter
"""

import json
import os
import sys


class Config:
    """
    Handles reading of config. parameters of the fenicsxadapter based on JSON
    configuration file. Initializer calls read_json() method. Instance attributes
    can be accessed by provided getter functions.
    """

    def __init__(self, adapter_config_filename):

        self._config_file_name = None
        self._participant_name = None
        self._coupling_mesh_names = None
        self._read_data_names = {}
        self._write_data_names = {}

        self.read_json(adapter_config_filename)

    def read_json(self, adapter_config_filename):
        """
        Reads JSON adapter configuration file and saves the data to the respective instance attributes.

        Parameters
        ----------
        adapter_config_filename : string
            Name of the JSON configuration file
        """
        folder = os.path.dirname(os.path.join(os.getcwd(), adapter_config_filename))
        path = os.path.join(folder, os.path.basename(adapter_config_filename))
        read_file = open(path, "r")
        data = json.load(read_file)
        self._config_file_name = os.path.join(folder, data["config_file_name"])
        self._participant_name = data["participant_name"]
        self._coupling_mesh_names = list(data["interfaces"].keys())

        for mesh_name in self._coupling_mesh_names:
            try:
                self._write_data_names[mesh_name] = data["interfaces"][mesh_name]["write_data_name"]
            except KeyError:
                # not required for one-way coupling, if this participant reads data
                self._write_data_names[mesh_name] = None

            try:
                self._read_data_names[mesh_name] = data["interfaces"][mesh_name]["read_data_name"]
            except KeyError:
                # not required for one-way coupling, if this participant writes data
                self._read_data_names[mesh_name] = None

        read_file.close()
        print(mesh_name)
        print(self._write_data_names)

    def get_config_file_name(self):
        return self._config_file_name

    def get_participant_name(self):
        return self._participant_name

    def get_coupling_mesh_names(self):
        return self._coupling_mesh_names

    def get_read_data_name(self, mesh_name):
        """
        Parameters
        ----------
        mesh_name : fenicsxprecice.Meshes member
            Member of the enum fenicsxprecice.Meshes

        """
        return self._read_data_names[mesh_name]

    def get_write_data_name(self, mesh_name):
        """
        Parameters
        ----------
        mesh_name : fenicsxprecice.Meshes member
            Member of the enum fenicsxprecice.Meshes

        """
        return self._write_data_names[mesh_name]
