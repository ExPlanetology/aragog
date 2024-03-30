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
"""Core classes and functions"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from spider.mesh import Mesh
from spider.parser import (
    Parameters,
    _BoundaryConditionsSettings,
    _InitialConditionSettings,
)

if TYPE_CHECKING:
    from spider.solver import State

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class BoundaryConditions:
    """Boundary conditions

    Args:
        parameters: Parameters
        mesh: Mesh
    """

    _parameters: Parameters
    _mesh: Mesh

    def __post_init__(self):
        self._settings: _BoundaryConditionsSettings = self._parameters.boundary_conditions

    def conform_temperature_boundary_conditions(
        self, temperature: np.ndarray, temperature_basic: np.ndarray, dTdr: np.ndarray
    ) -> None:
        """Conforms the temperature and dTdr at the basic nodes to temperature boundary conditions.

        Args:
            temperature: Temperature at the staggered nodes
            temperature_basic: Temperature at the basic nodes
            dTdr: Temperature gradient at the basic nodes
        """
        # Core-mantle boundary
        if self._settings.inner_boundary_condition == 3:
            temperature_basic[0, :] = self._settings.inner_boundary_value
            dTdr[0, :] = (
                2 * (temperature[0, :] - temperature_basic[0, :]) / self._mesh.basic.delta_radii[0]
            )
        # Surface
        if self._settings.outer_boundary_condition == 5:
            temperature_basic[-1, :] = self._settings.outer_boundary_value
            dTdr[-1, :] = (
                2
                * (temperature_basic[-1, :] - temperature[-1, :])
                / self._mesh.basic.delta_radii[-1]
            )

    def apply(self, state: State) -> None:
        """Applies the boundary conditions to the state.

        Args:
            state: The state to apply the boundary conditions to
        """
        self.apply_inner_boundary_condition(state)
        self.apply_outer_boundary_condition(state)
        logger.debug("temperature = %s", state.temperature_basic)
        logger.debug("heat_flux = %s", state.heat_flux)

    # TODO: Rename to only be associated with flux boundary conditions
    def apply_outer_boundary_condition(self, state: State) -> None:
        """Applies the outer boundary condition to the state.

        Args:
            state: The state to apply the boundary conditions to

        Equivalent to SURFACE_BC in C code.
            1: Grey-body atmosphere
            2: Zahnle steam atmosphere
            3: Couple to atmodeller
            4: Prescribed heat flux
            5: Prescribed temperature
        """
        if self._settings.outer_boundary_condition == 1:
            self.grey_body(state)
        elif self._settings.outer_boundary_condition == 2:
            raise NotImplementedError
        elif self._settings.outer_boundary_condition == 3:
            msg: str = "Requires coupling to atmodeller"
            logger.error(msg)
            raise NotImplementedError(msg)
        elif self._settings.outer_boundary_condition == 4:
            state.heat_flux[-1, :] = self._settings.outer_boundary_value
        elif self._settings.outer_boundary_condition == 5:
            pass
        else:
            msg: str = (
                f"outer_boundary_condition = {self._settings.outer_boundary_condition} is unknown"
            )
            raise ValueError(msg)

    def grey_body(self, state: State) -> None:
        """Applies a grey body flux at the surface.

        Args:
            state: The state to apply the boundary conditions to
        """
        state.heat_flux[-1, :] = (
            self._settings.emissivity
            * self._settings.scalings_.stefan_boltzmann_constant
            * (np.power(state.top_temperature, 4) - self._settings.equilibrium_temperature**4)
        )

    # TODO: Rename to only be associated with flux boundary conditions
    def apply_inner_boundary_condition(self, state: State) -> None:
        """Applies the inner boundary condition to the state.

        Args:
            state: The state to apply the boundary conditions to

        Equivalent to CORE_BC in C code.
            1: Simple core cooling
            2: Prescribed heat flux
            3: Prescribed temperature
        """
        if self._settings.inner_boundary_condition == 1:
            raise NotImplementedError
        elif self._settings.inner_boundary_condition == 2:
            state.heat_flux[0, :] = self._settings.inner_boundary_value
        elif self._settings.inner_boundary_condition == 3:
            pass
            # raise NotImplementedError
        else:
            msg: str = (
                f"inner_boundary_condition = {self._settings.inner_boundary_condition} is unknown"
            )
            raise ValueError(msg)


@dataclass
class InitialCondition:
    """Initial condition

    Args:
        parameters: Parameters
        mesh: Mesh
    """

    _parameters: Parameters
    _mesh: Mesh

    def __post_init__(self):
        self._settings: _InitialConditionSettings = self._parameters.initial_condition
        self._temperature: np.ndarray = self.get_linear()

    @property
    def temperature(self) -> np.ndarray:
        return self._temperature

    # TODO: Clunky. Set the staggered and basic temperature together, or be clear which one is
    # being set.
    def get_linear(self) -> np.ndarray:
        """Gets a linear temperature profile

        Returns:
            Linear temperature profile for the staggered nodes
        """
        temperature: np.ndarray = np.linspace(
            self._settings.basal_temperature,
            self._settings.surface_temperature,
            self._mesh.staggered.number_of_nodes,
        )
        return temperature
