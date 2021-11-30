import warnings

try:
    from dolfinx import *
except ModuleNotFoundError:
    warnings.warn("No FEniCSx installation found on system. Please check whether it is found correctly. "
                  "The FEniCSx adapter might not work as expected.\n\n")

from .fenicsxprecice import Adapter
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
