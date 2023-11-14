"""A phase defines EOS and transport properties.

See the LICENSE file for licensing information.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from configparser import SectionProxy
from dataclasses import dataclass, field
from typing import Callable, Self

import numpy as np

from spider.scalings import Scalings

logger: logging.Logger = logging.getLogger(__name__)


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


@dataclass(kw_only=True, frozen=True)
class PropertyABC(ABC):
    """A property whose value can be evaluated at temperature and pressure.

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
            The property value evaluated at temperature and pressure.
        """

    def __call__(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        return self.get_value(temperature, pressure)


@dataclass(kw_only=True, frozen=True)
class ConstantProperty(PropertyABC):
    """A property with a constant value

    Args:
        name: Name of the property
        value: The constant value

    Attributes:
        name: Name of the property
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
class PhaseStateStaggered:
    """Stores the state (material properties) of a phase at the staggered nodes.

    This only evaluates the necessary quantities to solve the system of equations to avoid
    unnecessary function calls that may slow down the code.

    Args:
        phase_evaluator: A PhaseEvaluator

    Attributes:
        capacitance: Thermal capacitance
        density: Density
        heat_capacity: Heat capacity
    """

    phase_evaluator: PhaseEvaluator
    capacitance: np.ndarray = field(init=False)
    density: np.ndarray = field(init=False)
    heat_capacity: np.ndarray = field(init=False)

    def update(self, temperature: np.ndarray, pressure: np.ndarray) -> None:
        """Updates the state.

        The order of evaluation matters.

        Args:
            temperature: Temperature at the staggered nodes
            pressure: Pressure at the staggered nodes
        """
        logger.debug("Updating the state of %s", self.__class__.__name__)
        self.density = self.phase_evaluator.density(temperature, pressure)
        logger.debug("density = %s", self.density)
        self.heat_capacity = self.phase_evaluator.heat_capacity(temperature, pressure)
        logger.debug("heat_capacity = %s", self.heat_capacity)
        self.capacitance = self.density * self.heat_capacity
        logger.debug("capacitance = %s", self.capacitance)


@dataclass
class PhaseStateBasic:
    """Stores the state (material properties) of a phase at the basic nodes.

    This minimises the number of function evaluations to avoid slowing down the code.

    Args:
        phase_evaluator: A PhaseEvaluator.

    Attributes:
        density: Density
        dTdrs: Adiabatic temperature gradient with respect to radius
        gravitational_acceleration: Gravitational acceleration
        heat_capacity: Heat capacity
        kinematic_viscosity: Kinematic viscosity
        thermal_conductivity: Thermal conductivity
        thermal_expansivity: Thermal expansivity
        viscosity: Dynamic viscosity
    """

    phase_evaluator: PhaseEvaluator
    density: np.ndarray = field(init=False)
    gravitational_acceleration: np.ndarray = field(init=False)
    heat_capacity: np.ndarray = field(init=False)
    thermal_conductivity: np.ndarray = field(init=False)
    thermal_expansivity: np.ndarray = field(init=False)
    viscosity: np.ndarray = field(init=False)
    _dTdrs: np.ndarray = field(init=False)
    _kinematic_viscosity: np.ndarray = field(init=False)

    def update(self, temperature: np.ndarray, pressure: np.ndarray) -> None:
        """Updates the state.

        The order of evaluation matters.

        Args:
            temperature: Temperature at the basic nodes
            pressure: Pressure at the basic nodes
        """
        logger.debug("Updating the state of %s", self.__class__.__name__)
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
    """Contains the objects to evaluate the EOS and transport properties of a phase.

    Args:
        density: To evaluate density at temperature and pressure
        gravitational_acceleration: To evaluate gravitational acceleration
        heat_capacity: To evaluate heat capacity
        thermal_conductivity: To evaluate thermal conductivity
        thermal_expansivity: To evaluate thermal expansivity
        viscosity: To evaluate viscosity

    Attributes:
        density: To evaluate density at temperature and pressure
        gravitational_acceleration: To evaluate gravitational acceleration
        heat_capacity: To evaluate heat capacity
        thermal_conductivity: To evaluate thermal conductivity
        thermal_expansivity: To evaluate thermal expansivity
        viscosity: To evaluate viscosity
    """

    density: PropertyABC
    gravitational_acceleration: PropertyABC
    heat_capacity: PropertyABC
    thermal_conductivity: PropertyABC
    thermal_expansivity: PropertyABC
    viscosity: PropertyABC

    @classmethod
    def from_configuration(cls, scalings: Scalings, *, config: SectionProxy) -> Self:
        """Creates a class instance from a configuration section.

        Args:
            scalings: Scalings for the numerical problem
            config: A configuration section with phase data

        Returns:
            A PhaseEvaluator
        """
        init_dict: dict[str, PropertyABC] = {}
        for key, value in config.items():
            try:
                value_float: float = float(value)
                value_float /= getattr(scalings, key)
                logger.debug("%s (%s) is a number = %f", key, config.name, value_float)
                init_dict[key] = ConstantProperty(name=key, value=value_float)

            # TODO: Add other tries to identify 1-D or 2-D lookup data.

            except TypeError:
                raise

        return cls(**init_dict)
