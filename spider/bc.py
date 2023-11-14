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
        scalings: Scalings
        outer_boundary_condition: TODO
        outer_boundary_value: TODO
        inner_boundary_condition: TODO
        inner_boundary_value: TODO
        emissivity: TODO
        equilibrium_temperature: TODO
        core_radius: TODO
        core_density: TODO
        core_heat_capacity: TODO

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

    def __post_init__(self):
        # Non-dimensionalise
        self.equilibrium_temperature /= self.scalings.temperature

    def grey_body(self, state: State) -> None:
        """Applies a grey body flux at the surface.

        Args:
            state: The state to apply the boundary conditions to
        """
        state.heat_flux[-1, :] = (
            self.emissivity
            * self.scalings.stefan_boltzmann_constant
            * (np.power(state.top_temperature, 4) - self.equilibrium_temperature**4)
        )
        # Below for debugging
        # state.heat_flux[-1, :] = 1e6 / self.scalings.heat_flux
        #  # 10**6 / (self.scalings.power / self.scalings.area)
        # state.heat_flux[-1, :] = 0

    def apply(self, state: State) -> None:
        """Applies the boundary conditions to the state.

        Args:
            state: The state to apply the boundary conditions to
        """
        self.core_heat_flux(state)
        self.grey_body(state)
        logger.info("temperature = %s", state.temperature_basic * self.scalings.temperature)
        logger.info("heat_flux = %s", state.heat_flux * self.scalings.heat_flux)

    def core_heat_flux(self, state: State) -> None:
        """Applies the heat flux at the core-mantle boundary.

        Args:
            state: The state to apply the boundary conditions to
        """
        # No heat flux from the core.
        state.heat_flux[0, :] = 0
