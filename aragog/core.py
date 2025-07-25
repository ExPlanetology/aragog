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
from scipy.integrate import solve_ivp

from aragog.mesh import Mesh
from aragog.phase import PhaseEvaluatorCollection
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

    def apply_temperature_boundary_conditions(
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

    def apply_temperature_boundary_conditions_melt(
        self, melt_fraction: npt.NDArray, melt_fraction_basic: npt.NDArray, dphidr: npt.NDArray
    ) -> None:
        """Conforms the melt fraction gradient dphidr at the basic nodes
           to temperature boundary conditions.

        Args:
            melt_fraction: Melt fraction at the staggered nodes
            melt_fraction_basic: Melt fraction at the basic nodes
            dphidr: Melt fraction gradient at the basic nodes
        """
        # Core-mantle boundary
        if self._settings.inner_boundary_condition == 3:
            dphidr[0, :] = (
                2 * (melt_fraction[0, :] - melt_fraction_basic[0, :])
                / self._mesh.basic.delta_radii[0]
            )
        # Surface
        if self._settings.outer_boundary_condition == 5:
            dphidr[-1, :] = (
                2
                * (melt_fraction_basic[-1, :] - melt_fraction[-1, :])
                / self._mesh.basic.delta_radii[-1]
            )

    def apply_flux_boundary_conditions(self, state: State) -> None:
        """Applies the boundary conditions to the state.

        Args:
            state: The state to apply the boundary conditions to
        """
        self.apply_flux_inner_boundary_condition(state)
        self.apply_flux_outer_boundary_condition(state)
        logger.debug("temperature = %s", state.temperature_basic)
        logger.debug("heat_flux = %s", state.heat_flux)

    def apply_flux_outer_boundary_condition(self, state: State) -> None:
        """Applies the flux boundary condition to the state at the outer boundary.

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

    def apply_flux_inner_boundary_condition(self, state: State) -> None:
        """Applies the flux boundary condition to the state at the inner boundary.

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
        cell_capacity = self._mesh.basic.volume[0] * state.capacitance_staggered()[0, :]
        radius_ratio: float = self._mesh.basic.radii[1] / self._mesh.basic.radii[0]
        alpha = np.power(radius_ratio, 2) / ((cell_capacity / (core_capacity * 1.147)) + 1)

        state.heat_flux[0, :] = alpha * state.heat_flux[1, :]


@dataclass
class InitialCondition:
    """Initial condition

    Args:
        parameters: Parameters
        mesh: Mesh
        phases: PhaseEvaluatorCollection
    """

    _parameters: Parameters
    _mesh: Mesh
    _phases: PhaseEvaluatorCollection

    def __post_init__(self):
        self._settings: _InitialConditionParameters = self._parameters.initial_condition

        # Three initialisation methods: linear (1), user-defined field (2) or adiabat (3).
        if self._settings.initial_condition == 1:
            self._temperature: npt.NDArray = self.get_linear()
        elif self._settings.initial_condition == 2:
            if self._mesh.staggered.number_of_nodes == len(self._settings.init_temperature):
                self._temperature = self._settings.init_temperature
            else:
                msg: str = (
                    f"the size of the provided init temperature field does not match \
                    the number of staggered points {self._mesh.staggered.number_of_nodes}"
                )
                raise ValueError(msg)
        elif self._settings.initial_condition == 3:
            self._temperature: npt.NDArray = self.get_adiabat(self._mesh.basic_pressure[:,-1])
        else:
            msg: str = (
                f"initial_condition = {self._settings.initial_condition} is unknown"
            )
            raise ValueError(msg)

        logger.debug("initial staggered temperature = %s", self._temperature)

    @property
    def temperature(self) -> npt.NDArray:
        return self._temperature

    def get_linear(self) -> npt.NDArray:
        """Gets a linear temperature profile

        Returns:
            Linear temperature profile for the staggered nodes
            Only works for uniform spatial mesh.
        """
        temperature_basic: npt.NDArray = np.linspace(
            self._settings.basal_temperature,
            self._settings.surface_temperature,
            self._mesh.basic.number_of_nodes,
        )
        return self._mesh.quantity_at_staggered_nodes(temperature_basic)

    def get_adiabat(self, pressure_basic) -> npt.NDArray:
        """Gets an adiabatic temperature profile by integrating
           the adiatiabatic temperature gradient dTdPs from the surface.
           Uses the set surface temperature.

        Args:
            Pressure field on the basic nodes

        Returns:
            Adiabatic temperature profile for the staggered nodes
        """

        def adiabat_ode(P,T):
            self._phases.active.set_pressure(P)
            self._phases.active.set_temperature(T)
            self._phases.active.update()
            return self._phases.active.dTdPs()

        # flip the pressure field top to bottom
        pressure_basic = np.flip(pressure_basic)

        sol = solve_ivp(
             adiabat_ode, (pressure_basic[0], pressure_basic[-1]),
             [self._settings.surface_temperature], t_eval=pressure_basic,
             method='RK45', rtol=1e-6, atol=1e-9)

        # flip back the temperature field from bottom to top
        temperature_basic = np.flip(sol.y[0])

        # Return temperature field at staggered nodes
        return self._mesh.quantity_at_staggered_nodes(temperature_basic)
