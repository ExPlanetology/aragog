"""A phase defines EOS and transport properties."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from configparser import SectionProxy
from dataclasses import dataclass
from typing import Callable

import numpy as np

logger: logging.Logger = logging.getLogger(__name__)


def ensure_size_equal_to_temperature(func: Callable) -> Callable:
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


@dataclass(kw_only=True, frozen=True)
class PropertyABC(ABC):
    """A property whose value can be accessed with a get_value method.

    Args:
        name: Name of the property.

    Attributes:
        name: Name of the property.
    """

    name: str

    @abstractmethod
    def get_value(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """Computes the property value for given input arguments.

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            An evaluation based on the provided arguments.
        """

    def __call__(self, *args, **kwargs):
        return self.get_value(*args, **kwargs)


@dataclass(kw_only=True, frozen=True)
class ConstantProperty(PropertyABC):
    """A property with a constant value

    Args:
        name: Name of the property.
        value: The constant value

    Attributes:
        name: Name of the property.
        value: The constant value
    """

    value: float

    @ensure_size_equal_to_temperature
    def get_value(self, temperature: np.ndarray, pressure: np.ndarray) -> float:
        """Returns the constant value. See base class."""
        del temperature
        del pressure
        return self.value  # The decorator ensures return type is np.ndarray.


@dataclass(kw_only=True, frozen=True)
class Phase:
    """Base class for a phase with EOS and transport properties."""

    density: PropertyABC
    gravitational_acceleration: PropertyABC
    heat_capacity: PropertyABC
    thermal_conductivity: PropertyABC
    thermal_expansivity: PropertyABC
    log10_viscosity: PropertyABC

    def dTdPs(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """dTdPs

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            dTdPs
        """
        dTdPs: np.ndarray = (
            self.thermal_expansivity(temperature, pressure)
            * temperature
            / self.density(temperature, pressure)
            / self.heat_capacity(temperature, pressure)
        )
        logger.debug("dTdPs = %s", dTdPs)

        return dTdPs

    def dTdrs(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """dTdrs

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            dTdrs
        """
        dTdrs: np.ndarray = -self.dTdzs(temperature, pressure)
        logger.debug("dTdrs = %s", dTdrs)

        return dTdrs

    def dTdzs(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """dTdzs

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            dTdzs
        """
        dTdzs: np.ndarray = (
            self.density(temperature, pressure)
            * self.dTdPs(temperature, pressure)
            * self.gravitational_acceleration(temperature, pressure)
        )
        logger.debug("dTdzs = %s", dTdzs)

        return dTdzs


def phase_factory(phase_section: SectionProxy) -> Phase:
    """Instantiates a Phase object.

    Args:
        phase_section: Configuration section with phase data

    Returns:
        A Phase object
    """
    init_dict: dict = {}
    for key, value in phase_section.items():
        try:
            value = float(value)
            logger.info("%s (%s) is a number = %f", key, phase_section.name, value)
            init_dict[key] = ConstantProperty(name=key, value=value)

        # TODO: Add other tries to identify 1-D or 2-D lookup data.

        except TypeError:
            raise

    phase: Phase = Phase(**init_dict)

    return phase
