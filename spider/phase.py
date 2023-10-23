"""A phase defines EOS and transport properties.

See the LICENSE file for licensing information.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from configparser import SectionProxy
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from spider.scalings import Scalings

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
        """Computes the property value at temperature and pressure.

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            An evaluation based on the provided arguments.
        """

    def __call__(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        return self.get_value(temperature, pressure)


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


@dataclass
class PhaseCurrentState:
    """Evaluates and stores the state of a phase at temperature and pressure.

    This minimises the number of function evaluations to get the phase properties.

    Args:
        phase_evaluator: A PhaseEvaluator.
    """

    phase_evaluator: PhaseEvaluator
    density: np.ndarray = field(init=False)
    gravitational_acceleration: np.ndarray = field(init=False)
    heat_capacity: np.ndarray = field(init=False)
    pressure: np.ndarray = field(init=False)
    temperature: np.ndarray = field(init=False)
    thermal_conductivity: np.ndarray = field(init=False)
    thermal_expansivity: np.ndarray = field(init=False)
    viscosity: np.ndarray = field(init=False)
    _dTdrs: np.ndarray = field(init=False)
    _kinematic_viscosity: np.ndarray = field(init=False)

    def eval(self, temperature: np.ndarray, pressure: np.ndarray) -> None:
        """Evaluates and stores the state.

        The order of evaluation matters.

        Args:
            temperature: Temperature
            pressure: Pressure
        """
        self.temperature = temperature
        self.pressure = pressure
        self.density = self.phase_evaluator.density(temperature, pressure)
        self.gravitational_acceleration = self.phase_evaluator.gravitational_acceleration(
            temperature, pressure
        )
        self.heat_capacity = self.phase_evaluator.heat_capacity(temperature, pressure)
        self.thermal_conductivity = self.phase_evaluator.thermal_conductivity(
            temperature, pressure
        )
        self.thermal_expansivity = self.phase_evaluator.thermal_expansivity(temperature, pressure)
        self.viscosity = self.phase_evaluator.viscosity(temperature, pressure)
        self._dTdrs = (
            -self.gravitational_acceleration
            * self.thermal_expansivity
            * temperature
            / self.heat_capacity
        )
        self._kinematic_viscosity = self.viscosity / self.density

    @property
    def dTdrs(self) -> np.ndarray:
        return self._dTdrs

    @property
    def kinematic_viscosity(self) -> np.ndarray:
        return self._kinematic_viscosity


@dataclass(kw_only=True, frozen=True)
class PhaseEvaluator:
    """Contains the objects to evaluate the EOS and transport properties of a phase"""

    density: PropertyABC
    gravitational_acceleration: PropertyABC
    heat_capacity: PropertyABC
    thermal_conductivity: PropertyABC
    thermal_expansivity: PropertyABC
    viscosity: PropertyABC


def phase_from_configuration(phase_section: SectionProxy, scalings: Scalings) -> PhaseEvaluator:
    """Instantiates a PhaseEvaluator object from configuration data.

    Args:
        phase_section: Configuration section with phase data
        scalings: Scalings for the numerical problem

    Returns:
        A Phase object
    """
    init_dict: dict[str, PropertyABC] = {}
    for key, value in phase_section.items():
        try:
            value_float: float = float(value)
            value_float /= getattr(scalings, key)
            logger.info("%s (%s) is a number = %f", key, phase_section.name, value_float)
            init_dict[key] = ConstantProperty(name=key, value=value_float)

        # TODO: Add other tries to identify 1-D or 2-D lookup data.

        except TypeError:
            raise

    phase_evaluator: PhaseEvaluator = PhaseEvaluator(**init_dict)

    return phase_evaluator
