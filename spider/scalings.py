"""Scalings for the numerical problem.

See the LICENSE file for licensing information.
"""

from __future__ import annotations

import logging
from configparser import SectionProxy
from dataclasses import dataclass
from functools import cached_property

import numpy as np

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(kw_only=True, frozen=True)
class NumericalScalings:
    """Scalings for the numerical problem.

    Args:
        radius: Radius in metres. Defaults to 1.
        temperature: Temperature in Kelvin. Defaults to 1.
        density: Density in kg/m^3. Defaults to 1.
        time: Time in seconds. Defaults to 1.
    """

    radius: float = 1
    temperature: float = 1
    density: float = 1
    time: float = 1

    @cached_property
    def area(self) -> float:
        return np.square(self.radius)

    @cached_property
    def energy(self) -> float:
        return self.mass * np.square(self.velocity)

    @cached_property
    def gravitational_acceleration(self) -> float:
        return self.radius / np.square(self.time)

    @cached_property
    def heat_capacity(self) -> float:
        return self.energy / self.mass / self.temperature

    @cached_property
    def mass(self) -> float:
        return self.density * self.volume

    @cached_property
    def power(self) -> float:
        return self.energy / self.time

    @cached_property
    def pressure(self) -> float:
        return self.density * self.gravitational_acceleration * self.radius

    @cached_property
    def stefan_boltzmann_constant(self) -> float:
        return self.power / np.square(self.radius) / np.power(self.temperature, 4)

    @cached_property
    def temperature_gradient(self) -> float:
        return self.temperature / self.radius

    @cached_property
    def thermal_conductivity(self) -> float:
        return self.power / self.radius / self.temperature

    @cached_property
    def thermal_expansivity(self) -> float:
        return 1 / self.temperature

    @cached_property
    def velocity(self) -> float:
        return self.radius / self.time

    @cached_property
    def viscosity(self) -> float:
        return self.pressure * self.time

    @cached_property
    def volume(self) -> float:
        return np.power(self.radius, 3)


def numerical_scalings_from_configuration(scalings_section: SectionProxy) -> NumericalScalings:
    """Instantiates the scalings for the numerical problem.

    Args:
        scalings_section: Configuration section with scalings.

    Returns:
        The numerical scalings.
    """
    radius: float = scalings_section.getfloat("radius")
    temperature: float = scalings_section.getfloat("temperature")
    density: float = scalings_section.getfloat("density")
    time: float = scalings_section.getfloat("time")

    numerical_scalings: NumericalScalings = NumericalScalings(
        radius=radius, temperature=temperature, density=density, time=time
    )

    return numerical_scalings
