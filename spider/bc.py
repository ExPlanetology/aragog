"""Boundary conditions

See the LICENSE file for licensing information.
"""

from __future__ import annotations

import logging
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, Union

import numpy as np

from spider.interfaces import DataclassFromConfiguration
from spider.scalings import Scalings

if TYPE_CHECKING:
    from spider.solver import State

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class BoundaryConditions(DataclassFromConfiguration):
    """Boundary conditions

    Args:
        # TODO

    Attributes:
        # TODO
    """

    scalings: Scalings
    _: KW_ONLY
    outer_boundary_condition: str
    outer_boundary_value: Union[str, float]
    inner_boundary_condition: str
    inner_boundary_value: Union[str, float]
    emissivity: float
    equilibrium_temperature: float
    core_radius: float
    core_density: float
    core_heat_capacity: float

    def _grey_body(self, state: State) -> None:
        """Applies a grey body at the surface."""
        state.heat_flux[-1, :] = (
            self.emissivity
            * self.scalings.stefan_boltzmann_constant
            * (
                np.power(state.top_temperature, 4)
                - (self.equilibrium_temperature / self.scalings.temperature) ** 4
            )
        )

    def apply(self, state: State) -> None:
        """Applies the boundary conditions."""
        self.core_heat_flux(state)
        self._grey_body(state)

    def core_heat_flux(self, state: State) -> None:
        """Applies the heat flux at the core-mantle boundary."""
        # No heat flux from the core.
        state.heat_flux[0, :] = 0
