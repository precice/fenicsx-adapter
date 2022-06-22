"""
This module provides a mechanism to imterpolate point data acquired from preCICE into FEniCSx Expressions.
"""

from .adapter_core import FunctionType
from scipy.interpolate import Rbf
from scipy.linalg import lstsq
from dolfinx.fem import Function
import numpy as np
from mpi4py import MPI

import logging

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class CouplingExpression(Function):
    """
    Creates functional representation (for FEniCSx) of nodal data provided by preCICE.
    """

    def __init__(self, function_space):
        self._dimension = function_space.mesh.geometry.dim
        super().__init__(function_space)

    def set_function_type(self, function_type):
        self._function_type = function_type

    def update_boundary_data(self, vals, coords_x, coords_y=None, coords_z=None):
        """
        Update object of this class of type FEniCSx UserExpression with given point data.

        Parameters
        ----------
        vals : double
            Point data to be used to update the Expression.
        coords_x : double
            X coordinate of points of which point data is provided.
        coords_y : double
            Y coordinate of points of which point data is provided.
        coords_z : double
            Z coordinate of points of which point data is provided.
        """
        self._coords_x = coords_x
        if coords_y is None:
            coords_y = np.zeros(self._coords_x.shape)
        self._coords_y = coords_y
        if coords_z is None:
            coords_z = np.zeros(self._coords_x.shape)

        self._coords_y = coords_y
        self._coords_z = coords_z
        self._vals = vals

        self._f = self.create_interpolant()

        if self.is_scalar_valued():
            assert (self._vals.shape[0] == self._coords_x.shape[0])
        elif self.is_vector_valued():
            assert (self._vals.shape[0] == self._coords_x.shape[0])

    def interpolate_precice(self, x):
        # TODO: the correct way to deal with this would be using an abstract class. Since this is technically more
        # complex and the current implementation is a workaround anyway, we do not
        # use the proper solution, but this hack.
        """
        Interpolates at x. Uses buffered interpolant self._f.
        Parameters
        ----------
        x : double
            Point.

        Returns
        -------
        list : python list
            A list containing the interpolated values. If scalar function is interpolated this list has a single
            element. If a vector function is interpolated the list has self._dimensions elements.
        """
        raise Exception("Please use one of the classes derived from this class, that implements an actual strategy for"
                        "interpolation.")

    def create_interpolant(self, x):
        # TODO: the correct way to deal with this would be using an abstract class. Since this is technically more
        # complex and the current implementation is a workaround anyway, we do not
        # use the proper solution, but this hack.
        """
        Creates interpolant from boundary data that has been provided before.

        Parameters
        ----------
        x : double
            Point.

        Returns
        -------
        list : python list
            Interpolant as a list. If scalar function is interpolated this list has a single
            element. If a vector function is interpolated the list has self._dimensions elements.
        """
        raise Exception("Please use one of the classes derived from this class, that implements an actual strategy for"
                        "interpolation.")

    def is_scalar_valued(self):
        """
        Determines if function being interpolated is scalar-valued based on dimension of provided vector self._vals.

        Returns
        -------
        tag : bool
            True if function being interpolated is scalar-valued, False otherwise.
        """
        return self._function_type is FunctionType.SCALAR

    def is_vector_valued(self):
        """
        Determines if function being interpolated is vector-valued based on dimension of provided vector self._vals.

        Returns
        -------
        tag : bool
            True if function being interpolated is vector-valued, False otherwise.
        """
        return self._function_type is FunctionType.VECTOR


class SegregatedRBFInterpolationExpression(CouplingExpression):
    """
    Uses polynomial quadratic fit + RBF interpolation for implementation of CustomExpression.interpolate. Allows for
    arbitrary coupling interfaces.

    See Lindner, F., Mehl, M., & Uekermann, B. (2017). Radial basis function interpolation for black-box multi-physics
    simulations.
    """

    def segregated_interpolant_2d(self, coords_x, coords_y, data):
        assert(coords_x.shape == coords_y.shape)
        # create least squares system to approximate a * x ** 2 + b * x + c ~= y

        def lstsq_interp(x, y, w): return w[0] * x ** 2 + w[1] * y ** 2 + w[2] * x * y + w[3] * x + w[4] * y + w[5]

        A = np.empty((coords_x.shape[0], 0))
        n_unknowns = 6
        for i in range(n_unknowns):
            w = np.zeros([n_unknowns])
            w[i] = 1
            column = lstsq_interp(coords_x, coords_y, w).reshape((coords_x.shape[0], 1))
            A = np.hstack([A, column])

        # solve system
        w, _, _, _ = lstsq(A, data)
        # create fit

        # compute remaining error
        res = data - lstsq_interp(coords_x, coords_y, w)
        # add RBF for error
        rbf_interp = Rbf(coords_x, coords_y, res)

        return lambda x, y: rbf_interp(x, y) + lstsq_interp(x, y, w)

    def create_interpolant(self):
        """
        See base class description.
        """
        assert (self._dimension == 2)  # current implementation only supports two dimensions
        interpolant = []

        if self.is_scalar_valued():  # check if scalar or vector-valued
            for d in range(1):
                interpolant.append(self.segregated_interpolant_2d(self._coords_x, self._coords_y, self._vals))
        elif self.is_vector_valued():
            for d in range(2):
                # TODO check if self._vals[:, d] is required here, above it had to be removed
                raise Exception("Not tested")
                interpolant.append(self.segregated_interpolant_2d(self._coords_x, self._coords_y, self._vals[:, d]))
        else:
            raise Exception("Problem dimension and data dimension not matching.")

        return interpolant

    def interpolate_precice(self, x):
        """
        See base class description.
        """
        assert (self._dimension == 2)  # current implementation only supports two dimensions

        if self.is_scalar_valued():
            return_value = [self._f[0](x[0], x[1])]
        if self.is_vector_valued():
            return_value = self._vals.ndim * [None]
            for i in range(self._vals.ndim):
                return_value[i] = self._f[i](x[0], x[1])
        return return_value
