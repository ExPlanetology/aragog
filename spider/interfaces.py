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

import logging
from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np

from spider.utilities import FloatOrArray

logger: logging.Logger = logging.getLogger(__name__)


class PropertyProtocol(Protocol):
    """Property protocol"""

    def eval(self, *args) -> FloatOrArray: ...

    def __call__(self, temperature: np.ndarray, pressure: np.ndarray) -> FloatOrArray: ...


class PhaseEvaluator(ABC):
    """Phase evaluator"""

    temperature: np.ndarray
    pressure: np.ndarray

    def set_temperature(self, temperature: np.ndarray) -> None:
        """Sets the temperature."""
        logger.debug("set_temperature = %s", temperature)
        self.temperature = temperature

    def set_pressure(self, pressure: np.ndarray) -> None:
        """Sets the pressure."""
        logger.debug("set_pressure = %s", pressure)
        self.pressure = pressure

    def update(self) -> None:
        """Precompute quantities to avoid duplicate calculations"""

    def dTdPs(self) -> np.ndarray:
        """TODO: Update reference to sphinx: Solomatov (2007), Treatise on Geophysics, Eq. 3.2"""
        dTdPs: np.ndarray = (
            self.thermal_expansivity() * self.temperature / (self.density() * self.heat_capacity())
        )

        return dTdPs

    def dTdrs(self) -> np.ndarray:
        dTdrs: np.ndarray = (
            -self.gravitational_acceleration()
            * self.thermal_expansivity()
            * self.temperature
            / self.heat_capacity()
        )

        return dTdrs

    def kinematic_viscosity(self) -> FloatOrArray:
        viscosity: FloatOrArray = self.viscosity() / self.density()

        return viscosity

    @abstractmethod
    def density(self) -> FloatOrArray: ...

    @abstractmethod
    def gravitational_acceleration(self) -> FloatOrArray: ...

    @abstractmethod
    def heat_capacity(self) -> FloatOrArray: ...

    @abstractmethod
    def melt_fraction(self) -> FloatOrArray: ...

    @abstractmethod
    def thermal_conductivity(self) -> FloatOrArray: ...

    @abstractmethod
    def thermal_expansivity(self) -> FloatOrArray: ...

    @abstractmethod
    def viscosity(self) -> FloatOrArray: ...
