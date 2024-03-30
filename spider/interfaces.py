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
from typing import Protocol

import numpy as np

from spider.utilities import FloatOrArray

logger: logging.Logger = logging.getLogger(__name__)


class PropertyProtocol(Protocol):
    """Property protocol"""

    def __call__(self, temperature: np.ndarray, pressure: np.ndarray) -> FloatOrArray: ...


class PhaseEvaluatorProtocol(Protocol):
    """Phase evaluator protocol"""

    def density(self, temperature: np.ndarray, pressure: np.ndarray) -> FloatOrArray:
        raise NotImplementedError

    def dTdPs(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def gravitational_acceleration(
        self, temperature: np.ndarray, pressure: np.ndarray
    ) -> FloatOrArray:
        raise NotImplementedError

    def heat_capacity(self, temperature: np.ndarray, pressure: np.ndarray) -> FloatOrArray:
        raise NotImplementedError

    def melt_fraction(self, temperature: np.ndarray, pressure: np.ndarray) -> FloatOrArray:
        raise NotImplementedError

    def thermal_conductivity(self, temperature: np.ndarray, pressure: np.ndarray) -> FloatOrArray:
        raise NotImplementedError

    def thermal_expansivity(self, temperature: np.ndarray, pressure: np.ndarray) -> FloatOrArray:
        raise NotImplementedError

    def viscosity(self, temperature: np.ndarray, pressure: np.ndarray) -> FloatOrArray:
        raise NotImplementedError
