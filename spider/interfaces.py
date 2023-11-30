"""Interfaces

See the LICENSE file for licensing information.
"""

from __future__ import annotations

import inspect
import logging
from abc import abstractmethod
from configparser import SectionProxy
from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np
from scipy import constants
from thermochem import codata

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class DataclassFromConfiguration:
    """A dataclass that can source its attributes from a configuration section"""

    @classmethod
    def from_configuration(cls, *args, config: SectionProxy) -> Self:
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
            k: config.getany(k) for k in config.keys() if k in inspect.signature(cls).parameters
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
class Scalings(ScaledDataclassFromConfiguration):
    """Scalings for the numerical problem.

    Args:
        radius: Radius in metres. Defaults to 1.
        temperature: Temperature in Kelvin. Defaults to 1.
        density: Density in kg/m^3. Defaults to 1.
        time: Time in seconds. Defaults to 1.

    Attributes:
        radius, m
        temperature, K
        density, kg/m^3
        time, s
        area, m^2
        kinetic_energy_per_volume, J/m^3
        gravitational_acceleration, m/s^2
        heat_capacity, J/kg/K
        heat_flux, W/m^2
        power_per_mass, W/kg
        power_per_volume, W/m^3
        pressure, Pa
        temperature_gradient, K/m
        thermal_conductivity, W/m/K
        velocity, m/s
        viscosity, Pa s
        time_years, years
        stefan_boltzmann_constant (non-dimensional)
    """

    # Default scalings
    radius: float = 1
    temperature: float = 1
    density: float = 1
    time: float = 1
    # Scalings (dimensional)
    area: float = field(init=False)
    gravitational_acceleration: float = field(init=False)
    heat_capacity: float = field(init=False)
    heat_flux: float = field(init=False)
    kinetic_energy_per_volume: float = field(init=False)
    power_per_mass: float = field(init=False)
    power_per_volume: float = field(init=False)
    pressure: float = field(init=False)
    temperature_gradient: float = field(init=False)
    time_years: float = field(init=False)  # Equivalent to TIMEYRS in C code version
    thermal_conductivity: float = field(init=False)
    velocity: float = field(init=False)
    viscosity: float = field(init=False)
    # Scalings (non-dimensional)
    stefan_boltzmann_constant: float = field(init=False)

    def scale_attributes(self):
        self.area = np.square(self.radius)
        self.gravitational_acceleration = self.radius / np.square(self.time)
        self.temperature_gradient = self.temperature / self.radius
        self.thermal_expansivity = 1 / self.temperature
        self.pressure = self.density * self.gravitational_acceleration * self.radius
        self.velocity = self.radius / self.time
        self.kinetic_energy_per_volume = self.density * np.square(self.velocity)
        self.heat_capacity = self.kinetic_energy_per_volume / self.density / self.temperature
        self.power_per_volume = self.kinetic_energy_per_volume / self.time
        self.power_per_mass = self.power_per_volume / self.density
        self.heat_flux = self.power_per_volume * self.radius
        self.thermal_conductivity = self.power_per_volume * self.area / self.temperature
        self.viscosity = self.pressure * self.time
        self.time_years = self.time / constants.Julian_year
        # Useful non-dimensional constants
        self.stefan_boltzmann_constant = codata.value("Stefan-Boltzmann constant")  # W/m^2/K^4
        self.stefan_boltzmann_constant /= (
            self.power_per_volume * self.radius / np.power(self.temperature, 4)
        )
        logger.debug("scalings = %s", self)
