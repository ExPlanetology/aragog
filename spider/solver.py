"""Solver

See the LICENSE file for licensing information.
"""

from __future__ import annotations

import logging
import sys
from configparser import SectionProxy
from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import OptimizeResult

from spider.core import SpiderConfigParser, SpiderData
from spider.interfaces import DataclassFromConfiguration
from spider.phase import PhaseStateBasic, PhaseStateStaggered

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class State(DataclassFromConfiguration):
    """Stores the state at temperature and pressure.

    Args:
        _data: SpiderData
        conduction: Include conduction flux
        convection: Include convection flux
        gravitational_separation: Include gravitational separation flux
        mixing: Include mixing flux
        radionuclides: Include radionuclides
        tidal: Include tidal heating

    Attributes:
        conduction: Include conduction flux
        convection: Include convection flux
        gravitational_separation: Include gravitational separation flux
        mixing: Include mixing flux
        radionuclides: Include radionuclides
        tidal: Include tidal heating
        phase_basic: Phase properties at the basic nodes
        phase_staggered: Phase properties at the staggered nodes
        conductive_heat_flux: Conductive heat flux at the basic nodes
        convective_heat_flux: Convective heat flux at the basic nodes
        critical_reynolds_number: Critical Reynolds number
        dTdr: Temperature gradient with respect to radius at the basic nodes
        eddy_diffusivity: Eddy diffusivity at the basic nodes
        gravitational_separation: Gravitational separation at the basic nodes
        heat_flux: Heat flux at the basic nodes
        inviscid_regime: Array with True if the flow is inviscid and otherwise False
        inviscid_velocity: Inviscid velocity
        is_convective: Array with True if the flow is convecting and otherwise False
        mixing: Mixing heat flux at the basic nodes
        reynolds_number: Reynolds number
        super_adiabatic_temperature_gradient: Super adiabatic temperature gradient
        temperature_basic: Temperature at the basic nodes
        bottom_temperature: Temperature at the bottom basic node
        top_temperature: Temperature at the top basic node
        viscous_regime: Array with True if the flow is viscous and otherwise False
        viscous_velocity: Viscous velocity
    """

    _data: SpiderData
    _: KW_ONLY
    conduction: bool
    convection: bool
    gravitational_separation: bool
    mixing: bool
    radionuclides: bool
    tidal: bool
    phase_basic: PhaseStateBasic = field(init=False)
    phase_staggered: PhaseStateStaggered = field(init=False)
    _dTdr: np.ndarray = field(init=False)
    _eddy_diffusivity: np.ndarray = field(init=False)
    _heat_flux: np.ndarray = field(init=False)
    _is_convective: np.ndarray = field(init=False)
    _reynolds_number: np.ndarray = field(init=False)
    _super_adiabatic_temperature_gradient: np.ndarray = field(init=False)
    _temperature_basic: np.ndarray = field(init=False)
    _viscous_velocity: np.ndarray = field(init=False)
    _inviscid_velocity: np.ndarray = field(init=False)

    def __post_init__(self):
        self.phase_basic = PhaseStateBasic(self._data.phase)
        self.phase_staggered = PhaseStateStaggered(self._data.phase)

    def conductive_heat_flux(self) -> np.ndarray:
        """Conductive heat flux"""
        conductive_heat_flux: np.ndarray = -self.phase_basic.thermal_conductivity * self._dTdr
        return conductive_heat_flux

    def convective_heat_flux(self) -> np.ndarray:
        """Convective heat flux"""
        convective_heat_flux: np.ndarray = (
            -self.phase_basic.density
            * self.phase_basic.heat_capacity
            * self._eddy_diffusivity
            * self._super_adiabatic_temperature_gradient
        )
        return convective_heat_flux

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

    def gravitational_separation_flux(self) -> np.ndarray:
        """Gravitational separation"""
        raise NotImplementedError

    @property
    def heat_flux(self) -> np.ndarray:
        """The total heat flux according to the fluxes specified in the configuration."""
        return self._heat_flux

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

    def _set_temperature(self, temperature: np.ndarray) -> None:
        """Sets the temperature at the basic nodes

        This also ensures that the temperature profile adheres to any imposed thermal boundary
        conditions at the top or bottom surfaces.

        Args:
            temperature: Temperature at the staggered nodes
        """
        logger.debug("Setting the temperature profile")
        self._temperature_basic = self._data.mesh.quantity_at_basic_nodes(temperature)
        self._dTdr = self._data.mesh.d_dr_at_basic_nodes(temperature)
        self._data.boundary_conditions.conform_temperature_boundary_conditions(
            temperature, self._temperature_basic, self._dTdr
        )
        logger.debug("_temperature_basic = %s", self._temperature_basic)
        logger.debug("_dTdr = %s", self._dTdr)

    def update(self, temperature: np.ndarray, pressure: np.ndarray) -> None:
        """Updates the state.

        The evaluation order matters because we want to minimise the number of evaluations.

        Args:
            temperature: Temperature at the staggered nodes
            pressure: Pressure at the staggered nodes
        """
        logger.debug("Updating the state")
        self._set_temperature(temperature)
        self.phase_staggered.update(temperature, pressure)
        pressure_basic: np.ndarray = self._data.mesh.quantity_at_basic_nodes(pressure)
        self.phase_basic.update(self._temperature_basic, pressure_basic)
        self._super_adiabatic_temperature_gradient = self._dTdr - self.phase_basic.dTdrs
        self._is_convective = self._super_adiabatic_temperature_gradient < 0
        velocity_prefactor: np.ndarray = (
            -self.phase_basic.gravitational_acceleration
            * self.phase_basic.thermal_expansivity
            * self._super_adiabatic_temperature_gradient
        )
        # Viscous velocity
        self._viscous_velocity = (
            velocity_prefactor * self._data.mesh.basic.mixing_length_cubed
        ) / (18 * self.phase_basic.kinematic_viscosity)
        self._viscous_velocity[~self.is_convective] = 0  # Must be super-adiabatic
        # Inviscid velocity
        self._inviscid_velocity = (
            velocity_prefactor * self._data.mesh.basic.mixing_length_squared
        ) / 16
        self._inviscid_velocity[~self.is_convective] = 0  # Must be super-adiabatic
        self._inviscid_velocity[self._is_convective] = np.sqrt(
            self._inviscid_velocity[self._is_convective]
        )
        # Reynolds number
        self._reynolds_number = (
            self._viscous_velocity
            * self._data.mesh.basic.mixing_length
            / self.phase_basic.kinematic_viscosity
        )
        # Eddy diffusivity
        self._eddy_diffusivity = np.where(
            self.viscous_regime, self._viscous_velocity, self._inviscid_velocity
        )
        self._eddy_diffusivity *= self._data.mesh.basic.mixing_length
        # Heat flux
        self._heat_flux: np.ndarray = np.zeros_like(self.temperature_basic)
        if self.conduction:
            self._heat_flux += self.conductive_heat_flux()
        if self.convection:
            self._heat_flux += self.convective_heat_flux()
        if self.gravitational_separation:
            self._heat_flux += self.gravitational_separation_flux()
        if self.mixing:
            self._heat_flux += self.mixing_flux()


@dataclass
class SpiderSolver:
    """Creates the system and solves the interior dynamics

    Args:
        filename: Filename of a file with configuration settings
        root_path: Root path to the flename

    Attributes:
        filename: Filename of a file with configuration settings
        root_path: Root path to the filename
    """

    filename: Union[str, Path]
    root_path: Union[str, Path] = ""
    root: Path = field(init=False)
    config: SpiderConfigParser = field(init=False)
    data: SpiderData = field(init=False)
    state: State = field(init=False)
    _solution: OptimizeResult = field(init=False, default_factory=OptimizeResult)

    def __post_init__(self):
        logger.info("Creating a SPIDER model")
        self.root = Path(self.root_path)
        self.config = SpiderConfigParser(self.root / self.filename)
        self.data = SpiderData(self.config)
        self.state = State.from_configuration(self.data, section=self.config["energy"])

    @property
    def solution(self) -> OptimizeResult:
        """The solution."""
        return self._solution

    def dTdt(
        self,
        time: float,
        temperature: np.ndarray,
        pressure: np.ndarray,
    ) -> np.ndarray:
        """dT/dt at the staggered nodes

        Args:
            time: Time
            temperature: Temperature at the staggered nodes
            pressure: Pressure at the staggered nodes

        Returns:
            dT/dt at the staggered nodes
        """
        logger.debug("temperature passed into dTdt = %s", temperature)
        self.state.update(temperature, pressure)
        heat_flux: np.ndarray = self.state.heat_flux
        logger.debug("heat_flux = %s", heat_flux)
        self.data.boundary_conditions.apply(self.state)
        logger.debug("heat_flux = %s", heat_flux)
        logger.debug("mesh.basic.area.shape = %s", self.data.mesh.basic.area.shape)

        energy_flux: np.ndarray = heat_flux * self.data.mesh.basic.area
        logger.debug("energy_flux size = %s", energy_flux.shape)

        delta_energy_flux: np.ndarray = np.diff(energy_flux, axis=0)
        capacitance: np.ndarray = (
            self.state.phase_staggered.capacitance * self.data.mesh.basic.volume
        )

        dTdt: np.ndarray = -delta_energy_flux / capacitance
        logger.debug("dTdt (fluxes only) = %s", dTdt)

        # dTdt += (
        #     self.state.phase_staggered.density
        #     * self.radiogenic_heating(time)
        #     * self.data.mesh.basic.volume
        #     / capacitance
        # )
        logger.debug("dTdt (with internal heating) = %s", dTdt)

        return dTdt

    def plot(self, num_lines: int = 11) -> None:
        """Plots the solution with labelled lines according to time.

        Args:
            num_lines: Number of lines to plot. Defaults to 11.
        """
        assert self.solution is not None

        # Dimensionalise quantities for plotting
        radii: np.ndarray = self.data.mesh.basic.radii * self.data.scalings.radius * 1.0e-3  # km
        # previous is below
        # temperature: np.ndarray = (
        #    self.data.mesh.quantity_at_basic_nodes(self.solution.y)
        #    * self.data.scalings.temperature  # K
        # )
        self.state.update(self.solution.y, self.solution.y)  # FIXME: Second argument is pressure.
        temperature: np.ndarray = self.state.temperature_basic * self.data.scalings.temperature

        times: np.ndarray = self.solution.t * self.data.scalings.time_years  # years

        plt.figure(figsize=(8, 6))
        ax = plt.subplot(111)

        # Ensure there are at least 2 lines to plot (first and last).
        num_lines = max(2, num_lines)

        # Calculate the time range.
        time_range: float = times[-1] - times[0]

        # Calculate the time step based on the total number of lines.
        time_step: float = time_range / (num_lines - 1)

        # Plot the first line.
        label_first: str = f"{times[0]:.2f}"
        ax.plot(temperature[:, 0], radii, label=label_first)

        # Loop through the selected lines and plot each with a label.
        for i in range(1, num_lines - 1):
            desired_time: float = times[0] + i * time_step
            # Find the closest available time step.
            closest_time_index: int = np.argmin(np.abs(times - desired_time))
            time: float = times[closest_time_index]
            label: str = f"{time:.2f}"  # Create a label based on the time.
            plt.plot(temperature[:, closest_time_index], radii, label=label)

        # Plot the last line.
        times_end: float = times[-1]
        label_last: str = f"{times_end:.2f}"
        ax.plot(temperature[:, -1], radii, label=label_last)

        # Shrink current axis by 20% to allow space for the legend.
        box = ax.get_position()
        ax.set_position((box.x0, box.y0, box.width * 0.8, box.height))

        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Radii (km)")
        ax.set_title("Magma ocean thermal profile")
        ax.grid(True)
        legend = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        legend.set_title("Time (yr)")
        plt.show()

    def solve(self) -> None:
        """Solves the system of ODEs to determine the interior temperature profile."""

        config_solver: SectionProxy = self.config["solver"]
        start_time: float = (
            config_solver.getfloat("start_time_years") / self.data.scalings.time_years
        )
        logger.debug("start_time = %f", start_time)
        end_time: float = config_solver.getfloat("end_time_years") / self.data.scalings.time_years
        logger.debug("end_time = %f", end_time)
        atol: float = config_solver.getfloat("atol")
        rtol: float = config_solver.getfloat("rtol")

        self._solution = solve_ivp(
            self.dTdt,
            (start_time, end_time),
            self.data.initial_condition.temperature,
            method="BDF",
            vectorized=True,
            args=(self.data.initial_condition.temperature,),  # FIXME: Should be pressure.
            atol=atol,
            rtol=rtol,
        )

        logger.info(self.solution)
