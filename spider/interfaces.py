#
# Copyright 2024 Dan J. Bower
#
# This file is part of Spider.
#
# Spider is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# Spider is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Spider. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Interfaces"""

from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

logger: logging.Logger = logging.getLogger(__name__)

# FIXME: Check x versus y or y versus x for pressure and temperature


@dataclass
class PropertyABC(ABC):
    """A property whose value is to be evaluated at temperature and pressure.

    Args:
        name: Name of the property

    Attributes:
        name: Name of the property
    """

    name: str

    @abstractmethod
    def _get_value(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray | float:
        """Computes the property value at temperature and pressure.

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            The property value evaluated at temperature and pressure.
        """

    def __call__(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """Returns an array with the same size as pressure"""
        # TODO: Would be more natural to use pressure, but using pressure breaks the code further
        # down the workflow.
        return self._get_value(temperature, pressure) * np.ones_like(temperature)


@dataclass(kw_only=True)
class ConstantProperty(PropertyABC):
    """A property with a constant value

    Args:
        name: Name of the property
        value: The constant value

    Attributes:
        name: Name of the property
        value: The constant value
        ndim: Number of dimensions, which is equal to zero
    """

    value: float
    ndim: int = field(init=False, default=0)

    @override
    def _get_value(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """See base class."""
        del pressure
        return self.value * np.ones_like(temperature)


@dataclass(kw_only=True)
class LookupProperty1D(PropertyABC):
    """A property from a 1-D lookup

    Args:
        name: Name of the property
        value: The 1-D array

    Attributes:
        name: Name of the property
        value: The 1-D array
        ndim: Number of dimensions, which is equal to one
    """

    value: np.ndarray
    ndim: int = field(init=False, default=1)
    _lookup: interp1d = field(init=False)

    def __post_init__(self):
        # Sort the data to ensure x is increasing
        data: np.ndarray = self.value[self.value[:, 0].argsort()]
        self._lookup = interp1d(data[:, 0], data[:, 1])

    @override
    def _get_value(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """See base class."""
        del temperature
        return self._lookup(pressure)


@dataclass(kw_only=True)
class LookupProperty2D(PropertyABC):
    """A property from a 2-D lookup

    Args:
        name: Name of the property
        value: The 2-D array

    Attributes:
        name: Name of the property
        value: The 2-D array
        ndim: Number of dimensions, which is equal to two
    """

    value: np.ndarray
    ndim: int = field(init=False, default=2)
    _lookup: RectBivariateSpline = field(init=False)

    def __post_init__(self):
        # x and y must be increasing otherwise the interpolation might give unexpected behaviour
        x_values = self.value[:, 0].round(decimals=0)
        logger.debug("x_values round = %s", x_values)
        logger.debug("self.value.shape = %s", self.value.shape)
        x_values: np.ndarray = np.unique(self.value[:, 0].round(decimals=0))
        logger.debug("x_values.shape = %s", x_values.shape)
        y_values: np.ndarray = np.unique(self.value[:, 1].round(decimals=0))
        logger.debug("y_values.shape = %s", y_values.shape)
        z_values: np.ndarray = self.value[:, 2]
        z_values = z_values.reshape((x_values.size, y_values.size), order="F")
        self._lookup = RectBivariateSpline(x_values, y_values, z_values, kx=1, ky=1, s=0)

    @override
    def _get_value(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """See base class."""
        return self._lookup(pressure, temperature, grid=False)
