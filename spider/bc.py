"""Boundary conditions

See the LICENSE file for licensing information.
"""

from __future__ import annotations

import logging
from dataclasses import KW_ONLY, dataclass
from typing import Union

import numpy as np

from spider.interfaces import DataclassFromConfiguration
from spider.scalings import Scalings

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

    def _grey_body(self, top_temperature: np.ndarray) -> np.ndarray:
        """Applies a grey body at the surface

        Args:
            top_temperature: An array of the surface temperatures

        Returns:
            An array of the grey body heat flux
        """
        ...

    def apply(self, heat_flux: np.ndarray, top_temperature: np.ndarray) -> None:
        """Applies the boundary conditions.

        Args:
            heat_flux: Heat flux to apply the boundary conditions to
            top_temperature: An array of the surface temperature
        """
        # No heat flux from the core.
        heat_flux[0, :] = 0

        heat_flux[-1, :] = (
            self.emissivity
            * self.scalings.stefan_boltzmann_constant
            * (
                np.power(top_temperature, 4)
                - (self.equilibrium_temperature / self.scalings.temperature) ** 4
            )
        )
