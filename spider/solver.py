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
"""Solver"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import OptimizeResult

from spider.core import SpiderData
from spider.parser import Parameters
from spider.phase import PhaseEvaluatorProtocol
from spider.utilities import FloatOrArray

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class PhaseStateStaggered:
    """Stores the state (material properties) at the staggered nodes.

    Args:
        phase_evaluator: A PhaseEvaluatorProtocol

    Attributes:
        phase_evaluator: A PhaseEvaluatorProtocol
        capacitance: Thermal capacitance
        density: Density
        heat_capacity: Heat capacity
    """

    phase_evaluator: PhaseEvaluatorProtocol
    capacitance: FloatOrArray = field(init=False)
    density: FloatOrArray = field(init=False)
    heat_capacity: FloatOrArray = field(init=False)

    def update(self, temperature: np.ndarray, pressure: np.ndarray) -> None:
        """Updates the state.

        Args:
            temperature: Temperature at the staggered nodes
            pressure: Pressure at the staggered nodes
        """
        logger.debug("Updating the state of %s", self.__class__.__name__)
        self.density = self.phase_evaluator.density(temperature, pressure)
        self.heat_capacity = self.phase_evaluator.heat_capacity(temperature, pressure)
        self.capacitance = self.density * self.heat_capacity


@dataclass
class PhaseStateBasic:
    """Stores the state (material properties) at the basic nodes.

    Args:
        phase_evaluator: A PhaseEvaluatorProtocol

    Attributes:
        phase_evaluator: A PhaseEvaluatorProtocol
        density: Density
        dTdrs: Adiabatic temperature gradient with respect to radius
        gravitational_acceleration: Gravitational acceleration
        heat_capacity: Heat capacity
        kinematic_viscosity: Kinematic viscosity
        thermal_conductivity: Thermal conductivity
        thermal_expansivity: Thermal expansivity
        viscosity: Dynamic viscosity
    """

    phase_evaluator: PhaseEvaluatorProtocol
    density: FloatOrArray = field(init=False)
    dTdrs: np.ndarray = field(init=False)
    gravitational_acceleration: FloatOrArray = field(init=False)
    heat_capacity: FloatOrArray = field(init=False)
    kinematic_viscosity: FloatOrArray = field(init=False)
    thermal_conductivity: FloatOrArray = field(init=False)
    thermal_expansivity: FloatOrArray = field(init=False)
    viscosity: FloatOrArray = field(init=False)

    def update(self, temperature: np.ndarray, pressure: np.ndarray) -> None:
        """Updates the state.

        Args:
            temperature: Temperature at the basic nodes
            pressure: Pressure at the basic nodes
        """
        self.density = self.phase_evaluator.density(temperature, pressure)
        self.gravitational_acceleration = self.phase_evaluator.gravitational_acceleration(
            temperature, pressure
        )
        self.heat_capacity = self.phase_evaluator.heat_capacity(temperature, pressure)
        self.thermal_conductivity = self.phase_evaluator.thermal_conductivity(
            temperature, pressure
        )
        self.thermal_expansivity = self.phase_evaluator.thermal_expansivity(temperature, pressure)
        self.viscosity = self.phase_evaluator.viscosity(temperature, pressure)
        self.dTdrs = (
            -self.gravitational_acceleration
            * self.thermal_expansivity
            * temperature
            / self.heat_capacity
        )
        self.kinematic_viscosity = self.viscosity / self.density


@dataclass
class State:
    """Stores and updates the state at temperature and pressure.

    update() minimises the number of function calls by only updating what is required to integrate
    the energy balance.

    Args:
        data: SpiderData

    Attributes:
        basic: State at the basic nodes
        staggered: State at the staggered nodes
        conductive_heat_flux: Conductive heat flux at the basic nodes
        convective_heat_flux: Convective heat flux at the basic nodes
        critical_reynolds_number: Critical Reynolds number
        dTdr: Temperature gradient at the basic nodes with respect to radius
        eddy_diffusivity: Eddy diffusivity at the basic nodes
        gravitational_separation_flux: Gravitational separation flux at the basic nodes
        heating: Heat generation at the staggered nodes
        heat_flux: Heat flux at the basic nodes
        inviscid_regime: True if the flow is inviscid and otherwise False, at the basic nodes
        inviscid_velocity: Inviscid velocity at the basic nodes
        is_convective: True if the flow is convecting and otherwise False, at the basic nodes
        reynolds_number: Reynolds number at the basic nodes
        super_adiabatic_temperature_gradient: Super adiabatic temperature gradient at the basic nod
        temperature_basic: Temperature at the basic nodes
        temperature_staggered: Temperature at the staggered nodes
        bottom_temperature: Temperature at the bottom basic node
        top_temperature: Temperature at the top basic node
        viscous_regime: True if the flow is viscous and otherwise False, at the basic nodes
        viscous_velocity: Viscous velocity at the basic nodes
    """

    data: SpiderData
    basic: PhaseStateBasic = field(init=False)
    staggered: PhaseStateStaggered = field(init=False)
    _dTdr: np.ndarray = field(init=False)
    _eddy_diffusivity: np.ndarray = field(init=False)
    _heat_flux: np.ndarray = field(init=False)
    _heating: np.ndarray = field(init=False)
    _is_convective: np.ndarray = field(init=False)
    _reynolds_number: np.ndarray = field(init=False)
    _super_adiabatic_temperature_gradient: np.ndarray = field(init=False)
    _temperature_basic: np.ndarray = field(init=False)
    _temperature_staggered: np.ndarray = field(init=False)
    _viscous_velocity: np.ndarray = field(init=False)
    _inviscid_velocity: np.ndarray = field(init=False)

    def __post_init__(self):
        self.phase_basic = PhaseStateBasic(self.data.phase)
        self.phase_staggered = PhaseStateStaggered(self.data.phase)

    @property
    def conductive_heat_flux(self) -> np.ndarray:
        """Conductive heat flux"""
        conductive_heat_flux: np.ndarray = -self.phase_basic.thermal_conductivity * self._dTdr

        return conductive_heat_flux

    @property
    def convective_heat_flux(self) -> np.ndarray:
        """Convective heat flux"""
        convective_heat_flux: np.ndarray = (
            -self.phase_basic.density
            * self.phase_basic.heat_capacity
            * self._eddy_diffusivity
            * self._super_adiabatic_temperature_gradient
        )

        return convective_heat_flux

    def radiogenic_heating(self, time: np.ndarray | float) -> np.ndarray | float:
        """Radiogenic heating

        Args:
            time: Time

        Returns:
            Radiogenic heating as a single column (in a 2-D array) if time is a float, otherwise a
                2-D array with each column associated with a single time in the time array.
        """
        radiogenic_heating_float: np.ndarray | float = 0
        for radionuclide in self.data.radionuclides:
            radiogenic_heating_float += radionuclide.get_heating(time)

        radiogenic_heating: np.ndarray | float = radiogenic_heating_float * (
            self.phase_staggered.density / self.phase_staggered.capacitance
        )

        return radiogenic_heating

    @property
    def critical_reynolds_number(self) -> float:
        """Critical Reynolds number from Abe (1993)"""
        return 9 / 8

    @property
    def dTdr(self) -> np.ndarray:
        return self._dTdr

    @property
    def eddy_diffusivity(self) -> np.ndarray:
        return self._eddy_diffusivity

    @property
    def gravitational_separation_flux(self) -> np.ndarray:
        """Gravitational separation"""
        raise NotImplementedError

    @property
    def heating(self) -> np.ndarray:
        """The total heating rate according to the heat sources specified in the configuration."""
        return self._heating

    @property
    def heat_flux(self) -> np.ndarray:
        """The total heat flux according to the fluxes specified in the configuration."""
        return self._heat_flux

    # TODO: Check this again
    @heat_flux.setter
    def heat_flux(self, value):
        """Setter for applying boundary conditions"""
        self._heat_flux = value

    @property
    def inviscid_regime(self) -> np.ndarray:
        return self._reynolds_number > self.critical_reynolds_number

    @property
    def inviscid_velocity(self) -> np.ndarray:
        return self._inviscid_velocity

    @property
    def is_convective(self) -> np.ndarray:
        return self._is_convective

    @property
    def mixing_flux(self) -> np.ndarray:
        """Mixing heat flux"""
        raise NotImplementedError

    @property
    def reynolds_number(self) -> np.ndarray:
        return self._reynolds_number

    @property
    def super_adiabatic_temperature_gradient(self) -> np.ndarray:
        return self._super_adiabatic_temperature_gradient

    @property
    def temperature_basic(self) -> np.ndarray:
        return self._temperature_basic

    @property
    def temperature_staggered(self) -> np.ndarray:
        return self._temperature_staggered

    @property
    def top_temperature(self) -> np.ndarray:
        return self._temperature_basic[-1, :]

    @property
    def bottom_temperature(self) -> np.ndarray:
        return self._temperature_basic[0, :]

    @property
    def viscous_regime(self) -> np.ndarray:
        return self._reynolds_number <= self.critical_reynolds_number

    @property
    def viscous_velocity(self) -> np.ndarray:
        return self._viscous_velocity

    def update(self, temperature: np.ndarray, time: np.ndarray | float) -> None:
        """Updates the state.

        The evaluation order matters because we want to minimise the number of evaluations.

        Args:
            temperature: Temperature at the staggered nodes
            pressure: Pressure at the staggered nodes
            time: Time
        """
        logger.debug("Updating the state")

        logger.debug("Setting the temperature profile")
        self._temperature_staggered = temperature
        self._temperature_basic = self.data.mesh.quantity_at_basic_nodes(temperature)
        logger.debug("temperature_basic = %s", self.temperature_basic)
        self._dTdr = self.data.mesh.d_dr_at_basic_nodes(temperature)
        logger.debug("dTdr = %s", self.dTdr)
        self.data.boundary_conditions.conform_temperature_boundary_conditions(
            temperature, self._temperature_basic, self._dTdr
        )

        self.phase_staggered.update(temperature, self.data.mesh.staggered.eos.pressure)
        self.phase_basic.update(self._temperature_basic, self.data.mesh.basic.eos.pressure)
        self._super_adiabatic_temperature_gradient = self._dTdr - self.phase_basic.dTdrs
        self._is_convective = self._super_adiabatic_temperature_gradient < 0
        velocity_prefactor: np.ndarray = (
            -self.phase_basic.gravitational_acceleration
            * self.phase_basic.thermal_expansivity
            * self._super_adiabatic_temperature_gradient
        )
        # Viscous velocity
        self._viscous_velocity = (
            velocity_prefactor * self.data.mesh.basic.mixing_length_cubed
        ) / (18 * self.phase_basic.kinematic_viscosity)
        self._viscous_velocity[~self.is_convective] = 0  # Must be super-adiabatic
        # Inviscid velocity
        self._inviscid_velocity = (
            velocity_prefactor * self.data.mesh.basic.mixing_length_squared
        ) / 16
        self._inviscid_velocity[~self.is_convective] = 0  # Must be super-adiabatic
        self._inviscid_velocity[self._is_convective] = np.sqrt(
            self._inviscid_velocity[self._is_convective]
        )
        # Reynolds number
        self._reynolds_number = (
            self._viscous_velocity
            * self.data.mesh.basic.mixing_length
            / self.phase_basic.kinematic_viscosity
        )
        # Eddy diffusivity
        self._eddy_diffusivity = np.where(
            self.viscous_regime, self._viscous_velocity, self._inviscid_velocity
        )
        self._eddy_diffusivity *= self.data.mesh.basic.mixing_length
        logger.debug("Before evaluating heat flux")
        # Heat flux
        self._heat_flux = np.zeros_like(self.temperature_basic)
        if self.data.parameters.energy.conduction:
            self._heat_flux += self.conductive_heat_flux
        if self.data.parameters.energy.convection:
            self._heat_flux += self.convective_heat_flux
        if self.data.parameters.energy.gravitational_separation:
            self._heat_flux += self.gravitational_separation_flux
        if self.data.parameters.energy.mixing:
            self._heat_flux += self.mixing_flux
        # Heating
        self._heating = np.zeros_like(self.temperature_staggered)
        if self.data.parameters.energy.radionuclides:
            self._heating += self.radiogenic_heating(time)


class Solver:
    """Creates the system and solves the interior dynamics

    Args:
        filename: Filename of a file with configuration settings
        root: Root path to the flename

    Attributes:
        filename: Filename of a file with configuration settings
        root: Root path to the filename. Defaults to empty
        config: Configuration data
        data: Model data
        state: Model state
    """

    def __init__(self, filename: str | Path, root: str | Path = Path()):
        logger.info("Creating a SPIDER model")
        self.filename = Path(filename)
        self.root = Path(root)
        self.parameters: Parameters
        self.data: SpiderData
        self.state: State
        self._solution: OptimizeResult
        self.parse_configuration()

    def parse_configuration(self) -> None:
        """Parses a configuration file"""
        configuration_file: Path = self.root / self.filename
        logger.info("Parsing configuration file = %s", configuration_file)
        self.parameters = Parameters.from_file(configuration_file)

    def initialize(self) -> None:
        """Initializes the model using configuration data"""
        logger.info("Initializing %s", self.__class__.__name__)
        self.data = SpiderData(self.parameters)
        self.state = State(self.data)

    @property
    def temperature_basic(self) -> np.ndarray:
        """Temperature of the basic mesh in K"""
        return self.data.mesh.quantity_at_basic_nodes(self.temperature_staggered)

    @property
    def temperature_staggered(self) -> np.ndarray:
        """Temperature of the staggered mesh in K"""
        temperature: np.ndarray = self.solution.y * self.data.parameters.scalings.temperature

        return temperature

    @property
    def solution(self) -> OptimizeResult:
        """The solution."""
        return self._solution

    def dTdt(
        self,
        time: np.ndarray | float,
        temperature: np.ndarray,
    ) -> np.ndarray:
        """dT/dt at the staggered nodes

        Args:
            time: Time
            temperature: Temperature at the staggered nodes

        Returns:
            dT/dt at the staggered nodes
        """
        logger.debug("temperature passed into dTdt = %s", temperature)
        # logger.debug("temperature.shape = %s", temperature.shape)
        self.state.update(temperature, time)
        heat_flux: np.ndarray = self.state.heat_flux
        # logger.debug("heat_flux = %s", heat_flux)
        self.data.boundary_conditions.apply(self.state)
        # logger.debug("heat_flux = %s", heat_flux)
        # logger.debug("mesh.basic.area.shape = %s", self.data.mesh.basic.area.shape)

        energy_flux: np.ndarray = heat_flux * self.data.mesh.basic.area
        # logger.debug("energy_flux size = %s", energy_flux.shape)

        delta_energy_flux: np.ndarray = np.diff(energy_flux, axis=0)
        # logger.debug("delta_energy_flux size = %s", delta_energy_flux.shape)
        # logger.debug("capacitance = %s", self.state.phase_staggered.capacitance.shape)
        capacitance: np.ndarray = (
            self.state.phase_staggered.capacitance * self.data.mesh.basic.volume
        )

        dTdt: np.ndarray = -delta_energy_flux / capacitance
        logger.debug("dTdt (fluxes only) = %s", dTdt)

        dTdt += self.state.heating
        logger.debug("dTdt (with internal heating) = %s", dTdt)

        return dTdt

    def solve(self) -> None:
        """Solves the system of ODEs to determine the interior temperature profile."""

        start_time: float = self.data.parameters.solver.start_time
        logger.debug("start_time = %f", start_time)
        end_time: float = self.data.parameters.solver.end_time
        logger.debug("end_time = %f", end_time)
        atol: float = self.data.parameters.solver.atol
        rtol: float = self.data.parameters.solver.rtol

        self._solution = solve_ivp(
            self.dTdt,
            (start_time, end_time),
            self.data.initial_condition.temperature,
            method="BDF",
            vectorized=True,
            atol=atol,
            rtol=rtol,
        )

        logger.info(self.solution)
