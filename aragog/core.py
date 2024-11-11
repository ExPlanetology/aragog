#
# Copyright 2024 Dan J. Bower
#
# This file is part of Aragog.
#
# Aragog is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# Aragog is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Aragog. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Core classes and functions"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from aragog.mesh import Mesh
from aragog.parser import (
    Parameters,
    _BoundaryConditionsParameters,
    _InitialConditionParameters,
)

if TYPE_CHECKING:
    from aragog.solver import State

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
        self._settings: _BoundaryConditionsParameters = self._parameters.boundary_conditions

    def conform_temperature_boundary_conditions(
        self, temperature: npt.NDArray, temperature_basic: npt.NDArray, dTdr: npt.NDArray
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
            self.core_cooling(state)
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

    def core_cooling(self, state: State) -> None:
        """Applies a core cooling heat flux according to Eq. (37) of Bower et al., 2018

        Args:
            state: The state to apply the boundary condition to
        """
        core_capacity: float = (
            4
            / 3
            * np.pi
            * np.power(self._mesh.basic.radii[0], 3)
            * self._settings.core_density
            * self._settings.core_heat_capacity
        )
        cell_capacity = self._mesh.basic.volume[0] * state.capacitance_staggered()[0, -1]
        radius_ratio: float = self._mesh.basic.radii[1] / self._mesh.basic.radii[0]
        alpha = np.power(radius_ratio, 2) / ((cell_capacity / (core_capacity * 1.147)) + 1)

        state.heat_flux[0, -1] = alpha * state.heat_flux[1, -1]


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
        self._settings: _InitialConditionParameters = self._parameters.initial_condition

        if self._settings.from_field:
            if self._mesh.staggered.number_of_nodes == len(self._settings.init_temperature):
                self._temperature = self._settings.init_temperature
            else:
                msg: str = (
                    f"the size of the provided init temperature field does not match \
                    the number of staggered points {self._mesh.staggered.number_of_nodes}"
                )
                raise ValueError(msg)
        else:
            self._temperature: npt.NDArray = self.get_linear()

        logger.debug("initial staggered temperature = %s", self._temperature)

    @property
    def temperature(self) -> npt.NDArray:
        return self._temperature

    # TODO: Clunky. Set the staggered and basic temperature together, or be clear which one is
    # being set.
    def get_linear(self) -> npt.NDArray:
        """Gets a linear temperature profile

        Returns:
            Linear temperature profile for the staggered nodes
        """
        temperature: npt.NDArray = np.linspace(
            self._settings.basal_temperature,
            self._settings.surface_temperature,
            self._mesh.staggered.number_of_nodes,
        )
        return temperature
