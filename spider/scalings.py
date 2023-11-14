"""Scalings for the numerical problem.

See the LICENSE file for licensing information.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import constants
from thermochem import codata

from spider.interfaces import DataclassFromConfiguration

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class Scalings(DataclassFromConfiguration):
    """Scalings for the numerical problem.

    Args:
        radius: Radius in metres. Defaults to 1.
        temperature: Temperature in Kelvin. Defaults to 1.
        density: Density in kg/m^3. Defaults to 1.
        time: Time in seconds. Defaults to 1.

    Attributes:
        radius
        temperature
        density
        time
        area
        energy
        gravitational_acceleration
        heat_capacity
        mass
        power
        pressure
        temperature_gradient
        thermal_conductivity
        velocity
        viscosity
        stefan_boltzmann_constant (non-dimensional)
        year_in_seconds (non-dimensional)
    """

    radius: float = 1
    temperature: float = 1
    density: float = 1
    time: float = 1
    area: float = field(init=False)
    energy: float = field(init=False)
    gravitational_acceleration: float = field(init=False)
    heat_capacity: float = field(init=False)
    mass: float = field(init=False)
    power: float = field(init=False)
    pressure: float = field(init=False)
    temperature_gradient: float = field(init=False)
    thermal_conductivity: float = field(init=False)
    velocity: float = field(init=False)
    viscosity: float = field(init=False)
    stefan_boltzmann_constant: float = field(init=False)
    time_year: float = field(init=False)

    def __post_init__(self):
        self.area = np.square(self.radius)
        self.gravitational_acceleration = self.radius / np.square(self.time)
        self.temperature_gradient = self.temperature / self.radius
        self.thermal_expansivity = 1 / self.temperature
        self.volume = np.power(self.radius, 3)
        self.mass = self.density * self.volume
        self.pressure = self.density * self.gravitational_acceleration * self.radius
        self.velocity = self.radius / self.time
        self.energy = self.mass * np.square(self.velocity)
        self.heat_capacity = self.energy / self.mass / self.temperature
        self.power = self.energy / self.time
        self.thermal_conductivity = self.power / self.radius / self.temperature
        self.viscosity = self.pressure * self.time
        # Useful non-dimensional constants
        self.stefan_boltzmann_constant = codata.value("Stefan-Boltzmann constant")  # W/m^2/K^4
        self.stefan_boltzmann_constant /= (
            self.power / np.square(self.radius) / np.power(self.temperature, 4)
        )
        # one Julian year (365.25 days) in non-dimensional time
        self.time_year = constants.Julian_year / self.time
        logger.debug("scalings = %s", self)
