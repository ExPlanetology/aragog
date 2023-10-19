"""A phase defines EOS and transport properties."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
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
        """Wrapper.

        Args:
            temperature: Temperature.
            pressure: Pressure.

        Returns:
            The quantity as an array with the same length as the temperature array.
        """
        result = func(self, temperature, pressure) * np.ones_like(temperature)

        return result

    return wrapper


@dataclass(kw_only=True)
class PhaseABC(ABC):
    """Base class for a phase with EOS and transport properties."""

    gravitational_acceleration_value: float

    @abstractmethod
    def density(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """Density

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            Density
        """

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
            * self.gravitational_acceleration
        )
        logger.debug("dTdzs = %s", dTdzs)

        return dTdzs

    @property
    def gravitational_acceleration(self) -> float:
        """Gravitational acceleration, which is alway positive by definition.

        Returns:
            Gravitational acceleration, which is always positive by definition.
        """
        return abs(self.gravitational_acceleration_value)

    @abstractmethod
    def heat_capacity(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """Heat capacity

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            Heat capacity
        """

    @abstractmethod
    def thermal_conductivity(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """Thermal conductivity

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            Thermal conductivity
        """

    @abstractmethod
    def thermal_expansivity(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """Thermal expansivity

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            Thermal expansivity
        """

    @abstractmethod
    def log10_viscosity(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """log10 of viscosity

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            Log10 of viscosity
        """

    # def phase_boundary(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    #     ...


@dataclass(kw_only=True)
class ConstantPhase(PhaseABC):
    """A phase with constant properties."""

    density_value: float
    heat_capacity_value: float
    thermal_conductivity_value: float
    thermal_expansivity_value: float
    log10_viscosity_value: float
    # _phase_boundary: float

    @ensure_size_equal_to_temperature
    def density(self, *args, **kwargs) -> float:
        del args
        del kwargs
        return self.density_value

    @ensure_size_equal_to_temperature
    def heat_capacity(self, *args, **kwargs) -> float:
        del args
        del kwargs
        return self.heat_capacity_value

    @ensure_size_equal_to_temperature
    def thermal_conductivity(self, *args, **kwargs) -> float:
        del args
        del kwargs
        return self.thermal_conductivity_value

    @ensure_size_equal_to_temperature
    def thermal_expansivity(self, *args, **kwargs) -> float:
        del args
        del kwargs
        return self.thermal_expansivity_value

    @ensure_size_equal_to_temperature
    def log10_viscosity(self, *args, **kwargs) -> float:
        del args
        del kwargs
        return self.log10_viscosity_value
