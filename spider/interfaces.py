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

import inspect
import logging
from abc import ABC, abstractmethod
from configparser import SectionProxy
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class DataclassFromConfiguration:
    """A dataclass that can source its attributes from a configuration section"""

    @classmethod
    def from_configuration(cls, *args, section: SectionProxy) -> DataclassFromConfiguration:
        """Creates an instance from a configuration section.

        This reads the configuration data and sources the attributes from this data and performs
        type conversations.

        Args:
            *args: Positional arguments to pass through to the constructor
            config: A configuration section

        Returns:
            A dataclass with its attributes populated
        """
        init_dict: dict[str, Any] = {
            k: section.getany(k) for k in section.keys() if k in inspect.signature(cls).parameters
        }
        return cls(*args, **init_dict)


@dataclass
class ScaledDataclassFromConfiguration(DataclassFromConfiguration):
    """A dataclass that requires its attributes to be scaled."""

    def __post_init__(self):
        self.scale_attributes()

    @abstractmethod
    def scale_attributes(self) -> None:
        """Scales the attributes"""


@dataclass(kw_only=True)
class PropertyABC(ABC):
    """A property whose value can be evaluated at temperature and pressure.

    Args:
        name: Name of the property

    Attributes:
        See Args.
    """

    name: str

    @abstractmethod
    def get_value(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """Computes the property value at temperature and pressure.

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            The property value evaluated at temperature and pressure.
        """

    def __call__(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        return self.get_value(temperature, pressure)


def ensure_size_equal_to_temperature(
    func: Callable[[ConstantProperty, np.ndarray, np.ndarray], float]
) -> Callable:
    """A decorator to ensure that the returned array is the same size as the temperature array.

    This is necessary when a phase is specified with constant properties that should be applied
    across the entire temperature and pressure range.
    """

    def wrapper(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """Wrapper

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            The quantity as an array with the same length as the temperature array.
        """
        result: np.ndarray = func(self, temperature, pressure) * np.ones_like(temperature)

        return result

    return wrapper


@dataclass(kw_only=True)
class ConstantProperty(PropertyABC):
    """A property with a constant value

    Args:
        name: Name of the property
        value: The constant value

    Attributes:
        See Args
        ndim: Number of dimensions (0)
    """

    value: float
    ndim: float = field(init=False, default=0)

    @ensure_size_equal_to_temperature
    def get_value(self, temperature: np.ndarray, pressure: np.ndarray) -> float:
        """Returns the constant value. See base class."""
        del temperature
        del pressure
        return self.value  # The decorator ensures return type is np.ndarray.


@dataclass(kw_only=True)
class LookupProperty1D(PropertyABC):
    """A property from a 1-D lookup

    Args:
        name: Name of the property
        value: The 1-D array

    Attributes:
        See Args
        ndim: Number of dimensions (1)
    """

    value: np.ndarray
    ndim: int = field(init=False, default=1)
    _lookup: interp1d = field(init=False)

    def __post_init__(self):
        # Sort data to ensure x is increasing
        data: np.ndarray = self.value[self.value[:, 0].argsort()]
        self._lookup = interp1d(data[:, 0], data[:, 1])

    def get_value(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        del temperature
        return self._lookup(pressure)


@dataclass(kw_only=True)
class LookupProperty2D(PropertyABC):
    """A property from a 2-D lookup

    Args:
        name: Name of the property
        value: The 2-D array

    Attributes:
        See Args
        ndim: Number of dimensions (2)
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

    def get_value(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        return self._lookup(pressure, temperature, grid=False)
