"""Solver

See the LICENSE file for licensing information.
"""

from __future__ import annotations

import logging
from configparser import ConfigParser, SectionProxy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import OptimizeResult

from spider.bc import BoundaryConditions
from spider.interfaces import DataclassFromConfiguration, MyConfigParser
from spider.mesh import StaggeredMesh
from spider.phase import PhaseEvaluator, PhaseStateBasic, PhaseStateStaggered
from spider.scalings import Scalings

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class State(DataclassFromConfiguration):
    """Stores the state at temperature and pressure.

    Args:
        _phase_evaluator: A PhaseEvaluator
        _mesh: A StaggeredMesh
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

    _phase_evaluator: PhaseEvaluator
    _mesh: StaggeredMesh
    conduction: bool
    convection: bool
    gravitational_separation: bool
    mixing: bool
    radionuclides: bool
    tidal: bool
    _heat_fluxes_to_include: list[Callable[[], np.ndarray]] = field(
        init=False, default_factory=list
    )
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
        self.phase_basic = PhaseStateBasic(self._phase_evaluator)
        self.phase_staggered = PhaseStateStaggered(self._phase_evaluator)
        self._set_heat_fluxes_to_include()

    def _set_heat_fluxes_to_include(self):
        """Sets the heat fluxes to include in the calculation.

        The desired heat fluxes are known from the configuration, so to avoid writing a switch
        statement that would need to be evaluated every call, instead we can just append the
        necessary methods to a list to be looped over.
        """
        if self.conduction:
            self._heat_fluxes_to_include.append(self.conductive_heat_flux)
        if self.convection:
            self._heat_fluxes_to_include.append(self.convective_heat_flux)
        if self.gravitational_separation:
            self._heat_fluxes_to_include.append(self.gravitational_separation_flux)
        if self.mixing:
            self._heat_fluxes_to_include.append(self.mixing_flux)

    def conductive_heat_flux(self) -> np.ndarray:
        """Conductive heat flux is only accessed once so therefore it is a property."""
        conductive_heat_flux: np.ndarray = -self.phase_basic.thermal_conductivity * self._dTdr
        return conductive_heat_flux

    def convective_heat_flux(self) -> np.ndarray:
        """Convective heat flux is only accessed once so therefore it is a property."""
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
        # logger.warning("dTdr = %s", self._dTdr)
        # print("dTdr = %s", self._dTdr)
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

    def update(self, temperature: np.ndarray, pressure: np.ndarray) -> None:
        """Updates the state.

        The evaluation order matters because we want to minimise the number of evaluations.

        Args:
            temperature: Temperature at the staggered nodes
            pressure: Pressure at the staggered nodes
        """
        logger.debug("Updating the state")
        self.phase_staggered.update(temperature, pressure)
        self._temperature_basic = self._mesh.quantity_at_basic_nodes(temperature)
        pressure_basic: np.ndarray = self._mesh.quantity_at_basic_nodes(pressure)
        self.phase_basic.update(self._temperature_basic, pressure_basic)
        self._dTdr = self._mesh.d_dr_at_basic_nodes(temperature)
        self.dTdr
        self._super_adiabatic_temperature_gradient = self._dTdr - self.phase_basic.dTdrs
        self._is_convective = self._super_adiabatic_temperature_gradient < 0
        velocity_prefactor: np.ndarray = (
            -self.phase_basic.gravitational_acceleration
            * self.phase_basic.thermal_expansivity
            * self._super_adiabatic_temperature_gradient
        )
        # Viscous velocity
        self._viscous_velocity = (velocity_prefactor * self._mesh.basic.mixing_length_cubed) / (
            18 * self.phase_basic.kinematic_viscosity
        )
        self._viscous_velocity[~self.is_convective] = 0  # Must be super-adiabatic
        # Inviscid velocity
        self._inviscid_velocity = (
            velocity_prefactor * self._mesh.basic.mixing_length_squared
        ) / 16
        self._inviscid_velocity[~self.is_convective] = 0  # Must be super-adiabatic
        self._inviscid_velocity[self._is_convective] = np.sqrt(
            self._inviscid_velocity[self._is_convective]
        )
        # Reynolds number
        self._reynolds_number = (
            self._viscous_velocity
            * self._mesh.basic.mixing_length
            / self.phase_basic.kinematic_viscosity
        )
        # Eddy diffusivity
        self._eddy_diffusivity = np.where(
            self.viscous_regime, self._viscous_velocity, self._inviscid_velocity
        )
        self._eddy_diffusivity *= self._mesh.basic.mixing_length
        # Heat flux
        self._heat_flux: np.ndarray = np.zeros_like(self.temperature_basic)
        for heat_flux_ in self._heat_fluxes_to_include:
            self._heat_flux += heat_flux_()


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
    scalings: Scalings = field(init=False)
    mesh: StaggeredMesh = field(init=False)
    phase_liquid_evaluator: PhaseEvaluator = field(init=False)
    phase_solid_evaluator: PhaseEvaluator = field(init=False)
    # Phase for calculations, could be a composite phase.
    phase_evaluator: PhaseEvaluator = field(init=False)
    state: State = field(init=False)
    initial_temperature: np.ndarray = field(init=False)
    bc: BoundaryConditions = field(init=False)
    _solution: OptimizeResult = field(init=False, default_factory=OptimizeResult)

    def __post_init__(self):
        logger.info("Creating a SPIDER model")
        self.root = Path(self.root_path)
        self.config: ConfigParser = MyConfigParser(self.root / self.filename)
        self.scalings = Scalings.from_configuration(config=self.config["scalings"])
        self.mesh = StaggeredMesh.uniform_radii(self.scalings, **self.config["mesh"])
        self.phase_liquid_evaluator = PhaseEvaluator.from_configuration(
            self.scalings, config=self.config["phase_liquid_evaluator"]
        )
        self.phase_solid_evaluator = PhaseEvaluator.from_configuration(
            self.scalings, config=self.config["phase_solid_evaluator"]
        )

        # FIXME: For time being just set phase to liquid phase.
        self.phase_evaluator = self.phase_liquid_evaluator
        self.state = State.from_configuration(
            self.phase_evaluator, self.mesh, config=self.config["energy"]
        )
        # Set the initial condition.
        initial: SectionProxy = self.config["initial_condition"]
        self.bc = BoundaryConditions.from_configuration(
            self.scalings, config=self.config["boundary_conditions"]
        )
        self.initial_temperature = np.linspace(
            initial.getfloat("basal_temperature"),
            initial.getfloat("surface_temperature"),
            self.mesh.staggered.number,
        )
        self.initial_temperature /= self.scalings.temperature

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
        """dT/dt at the staggered nodes.

        Args:
            time: Time.
            temperature: Temperature at the staggered nodes.
            pressure: Pressure at the staggered nodes.

        Returns:
            dT/dt at the staggered nodes.
        """
        logger.debug("temperature passed into dTdt = %s", temperature)
        self.state.update(temperature, pressure)
        heat_flux: np.ndarray = self.state.heat_flux
        logger.debug("heat_flux = %s", heat_flux)
        self.bc.apply(self.state)
        logger.debug("heat_flux = %s", heat_flux)
        logger.debug("mesh.basic.area.shape = %s", self.mesh.basic.area.shape)

        energy_flux: np.ndarray = heat_flux * self.mesh.basic.area.reshape(-1, 1)
        logger.debug("energy_flux = %s", energy_flux)
        logger.debug("energy_flux size = %s", energy_flux.shape)

        delta_energy_flux: np.ndarray = np.diff(energy_flux, axis=0)
        logger.debug("delta_energy_flux = %s", delta_energy_flux)
        capacitance: np.ndarray = (
            self.state.phase_staggered.capacitance * self.mesh.basic.volume.reshape(-1, 1)
        )

        dTdt: np.ndarray = -delta_energy_flux / capacitance

        # FIXME: Need to non-dimensionalise heating
        # dTdt += (
        #     self.phase.density
        #     * total_heating(self.config, time)
        #     * self.mesh.basic.volume
        #     / capacitance
        # )
        logger.info("dTdt = %s", dTdt)

        return dTdt

    def plot(self, num_lines: int = 11) -> None:
        """Plots the solution with labelled lines according to time.

        Args:
            num_lines: Number of lines to plot. Defaults to 11.
        """
        assert self.solution is not None

        # Dimensionalise quantities for plotting
        radii: np.ndarray = self.mesh.basic.radii * self.scalings.radius * 1.0e-3  # km
        temperature: np.ndarray = (
            self.mesh.quantity_at_basic_nodes(self.solution.y) * self.scalings.temperature  # K
        )
        times: np.ndarray = self.solution.t / self.scalings.time_year  # years

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
        start_time: float = config_solver.getfloat("start_time_years") * self.scalings.time_year
        logger.debug("start_time = %f", start_time)
        end_time: float = config_solver.getfloat("end_time_years") * self.scalings.time_year
        logger.debug("end_time = %f", end_time)
        atol: float = config_solver.getfloat("atol")
        rtol: float = config_solver.getfloat("rtol")

        self._solution = solve_ivp(
            self.dTdt,
            (start_time, end_time),
            self.initial_temperature,
            method="BDF",
            vectorized=True,
            args=(self.initial_temperature,),  # FIXME: Should be pressure.
            atol=atol,
            rtol=rtol,
        )

        logger.info(self.solution)
