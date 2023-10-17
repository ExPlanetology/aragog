"""EOS and transport properties of a phase."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

logger: logging.Logger = logging.getLogger(__name__)


class PhaseProtocol(Protocol):
    """A protocol for the EOS and transport properties of a phase."""

    def density(self, temperature: float, pressure: float) -> float:
        ...

    def dTdzs(self, temperature: float, pressure: float) -> float:
        ...

    def heat_capacity(self, temperature: float, pressure: float) -> float:
        ...

    def thermal_conductivity(self, temperature: float, pressure: float) -> float:
        ...

    def thermal_expansivity(self, temperature: float, pressure: float) -> float:
        ...

    def log10_viscosity(self, temperature: float, pressure: float) -> float:
        ...

    # def phase_boundary(self, temperature: float, pressure: float) -> float:
    #     ...


@dataclass(kw_only=True)
class ConstantPhase(PhaseProtocol):
    """A phase with constant properties."""

    _density: float
    _gravity: float  # TODO: Clean up, required for dTdPs. Must be negative.
    _heat_capacity: float
    _thermal_conductivity: float
    _thermal_expansivity: float
    _log10_viscosity: float
    # _phase_boundary: float

    def density(self, *args, **kwargs) -> float:
        del args
        del kwargs
        return self._density

    def dTdrs(self, *args, **kwargs) -> float:
        return -self.dTdzs(*args, **kwargs)

    def dTdzs(self, temperature: float, pressure: float) -> float:
        dTdzs: float = (
            self.thermal_expansivity(temperature, pressure)
            * -self._gravity
            * temperature
            / self.heat_capacity(temperature, pressure)
        )
        return dTdzs

    def heat_capacity(self, *args, **kwargs) -> float:
        del args
        del kwargs
        return self._heat_capacity

    def thermal_conductivity(self, *args, **kwargs) -> float:
        del args
        del kwargs
        return self._thermal_conductivity

    def thermal_expansivity(self, *args, **kwargs) -> float:
        del args
        del kwargs
        return self._thermal_expansivity

    def log10_viscosity(self, *args, **kwargs) -> float:
        del args
        del kwargs
        return self._log10_viscosity
