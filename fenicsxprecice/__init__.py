import warnings

try:
    from dolfinx import *
except ModuleNotFoundError:
    warnings.warn("No FEniCSx installation found on system. Please check whether it is found correctly. "
                  "The FEniCSx adapter might not work as expected.\n\n")

from .fenicsxprecice import Adapter
from .coupling_mesh import CouplingMesh

try:
    from importlib.metadata import version
except ImportError:
    # Python < 3.8
    from importlib_metadata import version

try:
    __version__ = version("fenicsxprecice")
except Exception:
    __version__ = "0.0.0+unknown"
