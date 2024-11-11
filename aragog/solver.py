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
"""Solver"""

from __future__ import annotations

import copy
import logging
import sys
from dataclasses import InitVar, dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp
from scipy.optimize import OptimizeResult

from aragog.core import BoundaryConditions, InitialCondition
from aragog.interfaces import PhaseEvaluatorProtocol
from aragog.mesh import Mesh
from aragog.parser import Parameters, _EnergyParameters, _Radionuclide
from aragog.phase import PhaseEvaluatorCollection
from aragog.utilities import FloatOrArray

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class State:
    """Stores and updates the state at temperature and pressure.

    Args:
        parameters: Parameters
        evaluator: Evaluator

    Attributes:
        evaluator: Evaluator
        critical_reynolds_number: Critical Reynolds number
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

    parameters: InitVar[Parameters]
    _evaluator: Evaluator
    _settings: _EnergyParameters = field(init=False)
    phase_basic: PhaseEvaluatorProtocol = field(init=False)
    phase_staggered: PhaseEvaluatorProtocol = field(init=False)
    _dTdr: npt.NDArray = field(init=False)
    _eddy_diffusivity: npt.NDArray = field(init=False)
    _heat_flux: npt.NDArray = field(init=False)
    _heating: npt.NDArray = field(init=False)
    _is_convective: npt.NDArray = field(init=False)
    _reynolds_number: npt.NDArray = field(init=False)
    _super_adiabatic_temperature_gradient: npt.NDArray = field(init=False)
    _temperature_basic: npt.NDArray = field(init=False)
    _temperature_staggered: npt.NDArray = field(init=False)
    _viscous_velocity: npt.NDArray = field(init=False)
    _inviscid_velocity: npt.NDArray = field(init=False)

    def __post_init__(self, parameters: Parameters):
        self._settings = parameters.energy
        # Sets the pressure since this will not change during a model run. Must deepcopy first
        # because pressure is set as an attribute.
        self.phase_basic = copy.deepcopy(self._evaluator.phases.active)
        self.phase_basic.set_pressure(self._evaluator.mesh.basic.pressure)
        self.phase_staggered = copy.deepcopy(self._evaluator.phases.active)
        self.phase_staggered.set_pressure(self._evaluator.mesh.staggered.pressure)

    def capacitance_staggered(self) -> FloatOrArray:
        capacitance: FloatOrArray = (
            self.phase_staggered.density() * self.phase_staggered.heat_capacity()
        )

        return capacitance

    def conductive_heat_flux(self) -> npt.NDArray:
        r"""Conductive heat flux:

        .. math::
            J_{cond} = -k \frac{\partial T}{\partial r}

        where :math:`k` is thermal conductivity, :math:`T` is temperature, and :math:`r` is radius.
        """
        conductive_heat_flux: npt.NDArray = self.phase_basic.thermal_conductivity() * -self.dTdr()

        return conductive_heat_flux

    def convective_heat_flux(self) -> npt.NDArray:
        r"""Convective heat flux:

        .. math::
            J_{conv} = -\rho c_p \kappa_h \left( \frac{\partial T}{\partial r}
                - \left( \frac{\partial T}{\partial r} \right)_S \right)

        where :math:`\rho` is density, :math:`c_p` is heat capacity at constant pressure,
        :math:`\kappa_h` is eddy diffusivity, :math:`T` is temperature, :math:`r` is radius, and
        :math:`S` is entropy.
        """
        convective_heat_flux: npt.NDArray = (
            self.phase_basic.density()
            * self.phase_basic.heat_capacity()
            * self.eddy_diffusivity()
            * -self._super_adiabatic_temperature_gradient
        )

        return convective_heat_flux

    def radiogenic_heating(self, time: FloatOrArray) -> FloatOrArray:
        """Radiogenic heating

        Args:
            time: Time

        Returns:
            Radiogenic heating as a single column (in a 2-D array) if time is a float, otherwise a
                2-D array with each column associated with a single time in the time array.
        """
        radiogenic_heating_float: FloatOrArray = 0
        for radionuclide in self._evaluator.radionuclides:
            radiogenic_heating_float += radionuclide.get_heating(time)

        radiogenic_heating: FloatOrArray = radiogenic_heating_float * (
            self.phase_staggered.density() / self.capacitance_staggered()
        )

        return radiogenic_heating

    @property
    def critical_reynolds_number(self) -> float:
        """Critical Reynolds number from Abe (1993)"""
        return 9 / 8

    def dTdr(self) -> npt.NDArray:
        return self._dTdr

    def eddy_diffusivity(self) -> npt.NDArray:
        return self._eddy_diffusivity

    @property
    def gravitational_separation_flux(self) -> npt.NDArray:
        """Gravitational separation"""
        raise NotImplementedError

    @property
    def heating(self) -> npt.NDArray:
        """The total heating rate according to the heat sources specified in the configuration."""
        return self._heating

    @property
    def heat_flux(self) -> npt.NDArray:
        """The total heat flux according to the fluxes specified in the configuration."""
        return self._heat_flux

    # TODO: Check this again
    @heat_flux.setter
    def heat_flux(self, value):
        """Setter for applying boundary conditions"""
        self._heat_flux = value

    @property
    def inviscid_regime(self) -> npt.NDArray:
        return self._reynolds_number > self.critical_reynolds_number

    @property
    def inviscid_velocity(self) -> npt.NDArray:
        return self._inviscid_velocity

    @property
    def is_convective(self) -> npt.NDArray:
        return self._is_convective

    @property
    def mixing_flux(self) -> npt.NDArray:
        """Mixing heat flux"""
        raise NotImplementedError

    @property
    def reynolds_number(self) -> npt.NDArray:
        return self._reynolds_number

    @property
    def super_adiabatic_temperature_gradient(self) -> npt.NDArray:
        return self._super_adiabatic_temperature_gradient

    @property
    def temperature_basic(self) -> npt.NDArray:
        return self._temperature_basic

    @property
    def temperature_staggered(self) -> npt.NDArray:
        return self._temperature_staggered

    @property
    def top_temperature(self) -> npt.NDArray:
        return self._temperature_basic[-1, :]

    @property
    def bottom_temperature(self) -> npt.NDArray:
        return self._temperature_basic[0, :]

    @property
    def viscous_regime(self) -> npt.NDArray:
        return self._reynolds_number <= self.critical_reynolds_number

    @property
    def viscous_velocity(self) -> npt.NDArray:
        return self._viscous_velocity

    def update(self, temperature: npt.NDArray, time: FloatOrArray) -> None:
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
        self._temperature_basic = self._evaluator.mesh.quantity_at_basic_nodes(temperature)
        logger.debug("temperature_basic = %s", self.temperature_basic)
        self._dTdr = self._evaluator.mesh.d_dr_at_basic_nodes(temperature)
        logger.debug("dTdr = %s", self.dTdr())
        self._evaluator.boundary_conditions.conform_temperature_boundary_conditions(
            temperature, self._temperature_basic, self.dTdr()
        )

        self.phase_staggered.set_temperature(temperature)
        self.phase_staggered.update()
        self.phase_basic.set_temperature(self._temperature_basic)
        self.phase_basic.update()

        self._super_adiabatic_temperature_gradient = self.dTdr() - self.phase_basic.dTdrs()
        self._is_convective = self._super_adiabatic_temperature_gradient < 0
        velocity_prefactor: npt.NDArray = (
            self.phase_basic.gravitational_acceleration()
            * self.phase_basic.thermal_expansivity()
            * -self._super_adiabatic_temperature_gradient
        )
        # Viscous velocity
        self._viscous_velocity = (
            velocity_prefactor * self._evaluator.mesh.basic.mixing_length_cubed
        ) / (18 * self.phase_basic.kinematic_viscosity())
        self._viscous_velocity[~self.is_convective] = 0  # Must be super-adiabatic
        # Inviscid velocity
        self._inviscid_velocity = (
            velocity_prefactor * self._evaluator.mesh.basic.mixing_length_squared
        ) / 16
        self._inviscid_velocity[~self.is_convective] = 0  # Must be super-adiabatic
        self._inviscid_velocity[self._is_convective] = np.sqrt(
            self._inviscid_velocity[self._is_convective]
        )
        # Reynolds number
        self._reynolds_number = (
            self._viscous_velocity
            * self._evaluator.mesh.basic.mixing_length
            / self.phase_basic.kinematic_viscosity()
        )
        # Eddy diffusivity
        self._eddy_diffusivity = np.where(
            self.viscous_regime, self._viscous_velocity, self._inviscid_velocity
        )
        self._eddy_diffusivity *= self._evaluator.mesh.basic.mixing_length
        # Heat flux
        self._heat_flux = np.zeros_like(self.temperature_basic)
        if self._settings.conduction:
            self._heat_flux += self.conductive_heat_flux()
        if self._settings.convection:
            self._heat_flux += self.convective_heat_flux()
        if self._settings.gravitational_separation:
            self._heat_flux += self.gravitational_separation_flux
        if self._settings.mixing:
            self._heat_flux += self.mixing_flux
        # Heating
        self._heating = np.zeros_like(self.temperature_staggered)
        if self._settings.radionuclides:
            self._heating += self.radiogenic_heating(time)


@dataclass
class Evaluator:
    """Contains classes that evaluate quantities necessary to compute the interior evolution.

    Args:
        _parameters: Parameters

    Attributes:
        boundary_conditions: Boundary conditions
        initial_condition: Initial condition
        mesh: Mesh
        phases: Evaluators for all phases
        radionuclides: Radionuclides
    """

    _parameters: Parameters
    boundary_conditions: BoundaryConditions = field(init=False)
    initial_condition: InitialCondition = field(init=False)
    mesh: Mesh = field(init=False)
    phases: PhaseEvaluatorCollection = field(init=False)

    def __post_init__(self):
        self.mesh = Mesh(self._parameters)
        self.boundary_conditions = BoundaryConditions(self._parameters, self.mesh)
        self.initial_condition = InitialCondition(self._parameters, self.mesh)
        self.phases = PhaseEvaluatorCollection(self._parameters)

    @property
    def radionuclides(self) -> list[_Radionuclide]:
        return self._parameters.radionuclides


class Solver:
    """Solves the interior dynamics

    Args:
        filename: Filename of a file with configuration settings
        root: Root path to the flename

    Attributes:
        filename: Filename of a file with configuration settings
        root: Root path to the filename. Defaults to empty
        parameters: Parameters
        evaluator: Evaluator
        state: State
    """

    def __init__(self, param: Parameters):
        logger.info("Creating an Aragog model")
        self.parameters: Parameters = param
        self.evaluator: Evaluator
        self.state: State
        self._solution: OptimizeResult

    @classmethod
    def from_file(cls, filename: str | Path, root: str | Path = Path()) -> Self:
        """Parses a configuration file

        Args:
            filename: Filename
            root: Root of the filename

        Returns:
            Parameters
        """
        configuration_file: Path = Path(root) / Path(filename)
        logger.info("Parsing configuration file = %s", configuration_file)
        parameters: Parameters = Parameters.from_file(configuration_file)

        return cls(parameters)

    def initialize(self) -> None:
        """Initializes the model."""
        logger.info("Initializing %s", self.__class__.__name__)
        self.evaluator = Evaluator(self.parameters)
        self.state = State(self.parameters, self.evaluator)

    @property
    def temperature_basic(self) -> npt.NDArray:
        """Temperature of the basic mesh in K"""
        return self.evaluator.mesh.quantity_at_basic_nodes(self.temperature_staggered)

    @property
    def temperature_staggered(self) -> npt.NDArray:
        """Temperature of the staggered mesh in K"""
        temperature: npt.NDArray = self.solution.y * self.parameters.scalings.temperature

        return temperature

    @property
    def solution(self) -> OptimizeResult:
        """The solution."""
        return self._solution

    def dTdt(
        self,
        time: npt.NDArray | float,
        temperature: npt.NDArray,
    ) -> npt.NDArray:
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
        heat_flux: npt.NDArray = self.state.heat_flux
        # logger.debug("heat_flux = %s", heat_flux)
        self.evaluator.boundary_conditions.apply(self.state)
        # logger.debug("heat_flux = %s", heat_flux)
        # logger.debug("mesh.basic.area.shape = %s", self.data.mesh.basic.area.shape)

        energy_flux: npt.NDArray = heat_flux * self.evaluator.mesh.basic.area
        # logger.debug("energy_flux size = %s", energy_flux.shape)

        delta_energy_flux: npt.NDArray = np.diff(energy_flux, axis=0)
        # logger.debug("delta_energy_flux size = %s", delta_energy_flux.shape)
        # logger.debug("capacitance = %s", self.state.phase_staggered.capacitance.shape)
        # FIXME: Update capacitance for mixed phase (enthalpy of fusion contribution)
        capacitance: npt.NDArray = (
            self.state.capacitance_staggered() * self.evaluator.mesh.basic.volume
        )

        dTdt: npt.NDArray = -delta_energy_flux / capacitance
        logger.debug("dTdt (fluxes only) = %s", dTdt)

        dTdt += self.state.heating
        logger.debug("dTdt (with internal heating) = %s", dTdt)

        return dTdt

    def solve(self) -> None:
        """Solves the system of ODEs to determine the interior temperature profile."""

        start_time: float = self.parameters.solver.start_time
        logger.debug("start_time = %f", start_time)
        end_time: float = self.parameters.solver.end_time
        logger.debug("end_time = %f", end_time)
        atol: float = self.parameters.solver.atol
        rtol: float = self.parameters.solver.rtol

        self._solution = solve_ivp(
            self.dTdt,
            (start_time, end_time),
            self.evaluator.initial_condition.temperature,
            method="BDF",
            vectorized=True,
            atol=atol,
            rtol=rtol,
        )

        logger.info(self.solution)
