import warnings

try:
    from dolfinx import *
except ModuleNotFoundError:
    warnings.warn("No FEniCSx installation found on system. Please check whether it is found correctly. "
                  "The FEniCSx adapter might not work as expected.\n\n")

from .fenicsxprecice import Adapter
from . import _version
__version__ = _version.get_versions()['version']
