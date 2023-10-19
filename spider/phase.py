"""EOS and transport properties of a phase."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np

logger: logging.Logger = logging.getLogger(__name__)


class PhaseProtocol(Protocol):
    """A protocol for the EOS and transport properties of a phase."""

    def density(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        ...

    def dTdrs(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        ...

    def heat_capacity(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        ...

    def thermal_conductivity(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        ...

    def thermal_expansivity(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        ...

    def log10_viscosity(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        ...

    # def phase_boundary(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    #     ...


@dataclass(kw_only=True)
class ConstantPhase(PhaseProtocol):
    """A phase with constant properties."""

    density_value: float
    gravity_value: float
    heat_capacity_value: float
    thermal_conductivity_value: float
    thermal_expansivity_value: float
    log10_viscosity_value: float
    # _phase_boundary: float

    def __post_init__(self):
        # Convention is for positive gravity.
        self.gravity_value = abs(self.gravity_value)

    def density(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        del pressure
        return self.density_value * np.ones_like(temperature)

    def dTdPs(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        dTdPs: np.ndarray = (
            self.thermal_expansivity(temperature, pressure)
            * temperature
            / self.density(temperature, pressure)
            / self.heat_capacity(temperature, pressure)
        )
        logger.debug("dTdPs = %s", dTdPs)
        return dTdPs * np.ones_like(temperature)

    def dTdrs(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        dTdrs: np.ndarray = -self.dTdzs(temperature, pressure)
        logger.debug("dTdrs = %s", dTdrs)
        return dTdrs

    def dTdzs(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        dTdzs: np.ndarray = (
            self.density(temperature, pressure)
            * self.gravity_value
            * self.dTdPs(temperature, pressure)
        )
        return dTdzs

    def heat_capacity(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        del pressure
        return self.heat_capacity_value * np.ones_like(temperature)

    def thermal_conductivity(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        del pressure
        return self.thermal_conductivity_value * np.ones_like(temperature)

    def thermal_expansivity(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        del pressure
        return self.thermal_expansivity_value * np.ones_like(temperature)

    def log10_viscosity(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        del pressure
        return self.log10_viscosity_value * np.ones_like(temperature)
