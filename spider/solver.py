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

from spider import STEFAN_BOLTZMANN_CONSTANT, YEAR_IN_SECONDS
from spider.energy import total_heat_flux, total_heating
from spider.mesh import StaggeredMesh, mesh_from_configuration
from spider.phase import PhaseCurrentState, PhaseEvaluator, phase_from_configuration
from spider.scalings import Scalings, scalings_from_configuration

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class SpiderSolver:
    filename: Union[str, Path]
    root_path: Union[str, Path] = ""
    mesh: StaggeredMesh = field(init=False)
    phase_liquid_evaluator: PhaseEvaluator = field(init=False)
    phase_solid_evaluator: PhaseEvaluator = field(init=False)
    # Phase for calculations, could be a composite phase.
    phase_evaluator: PhaseEvaluator = field(init=False)
    phase_current_state_basic: PhaseCurrentState = field(init=False)
    phase_current_state_staggered: PhaseCurrentState = field(init=False)
    initial_temperature: np.ndarray = field(init=False)
    initial_time: float = field(init=False, default=0)
    end_time: float = field(init=False, default=0)
    scalings: Scalings = field(init=False)
    _solution: OptimizeResult = field(init=False, default_factory=OptimizeResult)

    def __post_init__(self):
        logger.info("Creating a SPIDER model")
        self.config: ConfigParser = MyConfigParser(self.filename)
        self.root: Path = Path(self.root_path)
        self.scalings = scalings_from_configuration(self.config["scalings"])
        self.mesh = mesh_from_configuration(self.config["mesh"], self.scalings)
        self.phase_liquid_evaluator = phase_from_configuration(
            self.config["phase_liquid_evaluator"], self.scalings
        )
        self.phase_solid_evaluator = phase_from_configuration(
            self.config["phase_solid_evaluator"], self.scalings
        )
        # FIXME: For time being just set phase to liquid phase.
        self.phase_evaluator = self.phase_liquid_evaluator
        self.phase_current_state_basic = PhaseCurrentState(self.phase_evaluator)
        self.phase_current_state_staggered = PhaseCurrentState(self.phase_evaluator)
        # Set the time stepping.
        self.initial_time = self.config.getfloat("timestepping", "start_time_years")
        self.initial_time *= YEAR_IN_SECONDS / self.scalings.time
        self.end_time = self.config.getfloat("timestepping", "end_time_years")
        self.end_time *= YEAR_IN_SECONDS / self.scalings.time
        # Set the initial condition.
        initial: SectionProxy = self.config["initial_condition"]
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
            mesh: Mesh.
            pressure: Pressure at the staggered nodes.

        Returns:
            dT/dt at the staggered nodes.
        """
        temperature_basic: np.ndarray = self.mesh.quantity_at_basic_nodes(temperature)
        pressure_basic: np.ndarray = self.mesh.quantity_at_basic_nodes(pressure)
        self.phase_current_state_staggered.eval(temperature, pressure)
        self.phase_current_state_basic.eval(temperature_basic, pressure_basic)

        eddy_diffusivity: np.ndarray = self.eddy_diffusivity()
        energy: SectionProxy = self.config["energy"]
        heat_flux: np.ndarray = total_heat_flux(
            energy,
            self.mesh,
            self.phase_current_state_basic,
            eddy_diffusivity,
            temperature,
            pressure,
        )

        # TODO: Clean up boundary conditions.
        # No heat flux from the core.
        heat_flux[0] = 0
        # Blackbody cooling.
        equilibrium_temperature: float = self.config.getfloat(
            "boundary_conditions", "equilibrium_temperature"
        )
        equilibrium_temperature /= self.scalings.temperature
        heat_flux[-1] = (
            self.config.getfloat("boundary_conditions", "emissivity")
            * STEFAN_BOLTZMANN_CONSTANT
            / self.scalings.stefan_boltzmann_constant
            * (
                self.mesh.quantity_at_basic_nodes(temperature)[-1] ** 4
                - equilibrium_temperature**4
            )
        )

        energy_flux: np.ndarray = heat_flux * self.mesh.basic.area
        logger.info("energy_flux = %s", energy_flux)

        delta_energy_flux: np.ndarray = np.diff(energy_flux)
        logger.info("delta_energy_flux = %s", delta_energy_flux)
        capacitance: np.ndarray = (
            self.phase_current_state_staggered.heat_capacity
            * self.phase_current_state_staggered.density
            * self.mesh.basic.volume
        )

        dTdt: np.ndarray = -delta_energy_flux / capacitance

        # FIXME: Need to non-dimensionalise heating
        # dTdt += (
        #     self.phase.density
        #     * total_heating(self.config, time)
        #     * self.mesh.basic.volume
        #     / capacitance
        # )
        # logger.info("dTdt = %s", dTdt)

        return dTdt

    def plot(self, num_lines: int = 11) -> None:
        """Plots the solution with labelled lines according to time.

        Args:
            num_lines: Number of lines to plot. Defaults to 11.
        """
        assert self.solution is not None
        radii: np.ndarray = self.mesh.basic.radii
        y_basic: np.ndarray = self.mesh.quantity_at_basic_nodes(self.solution.y)
        times: np.ndarray = self.solution.t

        plt.figure(figsize=(8, 6))
        ax = plt.subplot(111)

        # Ensure there are at least 2 lines to plot (first and last).
        num_lines = max(2, num_lines)

        # Calculate the time range.
        time_range: float = times[-1] - times[0]

        # Calculate the time step based on the total number of lines.
        time_step: float = time_range / (num_lines - 1)

        # Plot the first line.
        label_first: str = f"{times[0]/YEAR_IN_SECONDS:.2f}"
        ax.plot(y_basic[:, 0], radii, label=label_first)

        # Loop through the selected lines and plot each with a label.
        for i in range(1, num_lines - 1):
            desired_time: float = times[0] + i * time_step
            # Find the closest available time step.
            closest_time_index: int = np.argmin(np.abs(times - desired_time))
            time: float = times[closest_time_index]
            label: str = f"{time/YEAR_IN_SECONDS:.2f}"  # Create a label based on the time.
            plt.plot(y_basic[:, closest_time_index], radii, label=label)

        # Plot the last line.
        label_last: str = f"{times[-1]/YEAR_IN_SECONDS:.2f}"
        ax.plot(y_basic[:, -1], radii, label=label_last)

        # Shrink current axis by 20%.
        box = ax.get_position()
        ax.set_position((box.x0, box.y0, box.width * 0.8, box.height))

        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Radii (m)")
        ax.set_title("Magma ocean thermal profile")
        ax.grid(True)
        legend = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        legend.set_title("Time (yr)")
        plt.show()

    def solve(self, atol: float = 1.0e-6, rtol: float = 1.0e-6) -> None:
        """Solves the system of ODEs to determine the interior temperature profile.

        Args:
            atol: Absolute tolerance. Defaults to 1.0e-6.
            rtol: Relative tolerance. Defaults to 1.0e-6.
        """
        self._solution = solve_ivp(
            self.dTdt,
            (self.initial_time, self.end_time),
            self.initial_temperature,
            method="BDF",
            vectorized=False,  # TODO: True could speed up BDF according to the documentation.
            args=(self.initial_temperature,),  # FIXME: Should be pressure.
            atol=atol,
            rtol=rtol,
        )

        logger.info(self.solution)

    def super_adiabatic_temperature_gradient(self) -> np.ndarray:
        """Super-adiabatic temperature gradient at the basic nodes

        By definition, this is negative if the system is convective, positive if not.

        Returns:
            Super adiabatic temperature gradient
        """
        super_adiabatic: np.ndarray = (
            self.mesh.d_dr_at_basic_nodes(self.phase_current_state_staggered.temperature)
            - self.phase_current_state_basic.dTdrs
        )
        logger.info("super_adiabatic_temperature_gradient = %s", super_adiabatic)
        return super_adiabatic

    def is_convective(self) -> np.ndarray:
        """Is convection occurring at the basic nodes.

        Args:
            temperature: Temperature at the staggered nodes
            pressure: Pressure at the staggered nodes

        Returns:
            True if convective, otherwise False
        """
        is_convective: np.ndarray = self.super_adiabatic_temperature_gradient() < 0
        logger.info("is_convective = %s", is_convective)

        return is_convective

    def velocity_inviscid(self) -> np.ndarray:
        """Velocity of convective parcels controlled by dynamic pressure at the basic nodes.

        Args:
            temperature: Temperature at the staggered nodes
            pressure: Pressure at the staggered nodes

        Returns:
            Inviscid convective velocity
        """
        velocity: np.ndarray = (
            -self.phase_current_state_basic.gravitational_acceleration
            * self.phase_current_state_basic.thermal_expansivity
            * np.square(self.mesh.basic.mixing_length)
        )
        velocity *= self.super_adiabatic_temperature_gradient()
        velocity /= 16
        # A convective velocity requires a super-adiabatic temperature gradient.
        velocity[velocity < 0] = 0
        velocity[velocity > 0] = np.sqrt(velocity[velocity > 0])

        logger.info("velocity_inviscid = %s", velocity)

        return velocity

    def velocity_viscous(self) -> np.ndarray:
        """Velocity of convective parcels controlled by viscous drag at the basic nodes.

        Returns:
            Viscous convective velocity
        """
        velocity: np.ndarray = (
            -self.phase_current_state_basic.gravitational_acceleration
            * self.phase_current_state_basic.thermal_expansivity
            * np.power(self.mesh.basic.mixing_length, 3)
        )
        velocity *= self.super_adiabatic_temperature_gradient()
        velocity /= 18 * self.phase_current_state_basic.kinematic_viscosity
        # A convective velocity requires a super-adiabatic temperature gradient.
        velocity[velocity < 0] = 0
        logger.info("velocity_viscous = %s", velocity)

        return velocity

    def reynolds_number(self) -> np.ndarray:
        """Reynolds number at the basic nodes

        Args:
            temperature: Temperature at the staggered nodes
            pressure: Pressure at the staggered nodes

        Returns:
            Reynolds number
        """
        reynolds: np.ndarray = self.velocity_viscous() * self.mesh.basic.mixing_length
        reynolds /= self.phase_current_state_basic.kinematic_viscosity
        logger.info("reynolds_number = %s", reynolds)

        return reynolds

    def eddy_diffusivity(self) -> np.ndarray:
        """Eddy diffusivity at the basic nodes

        Args:
            temperature: Temperature at the staggered nodes
            pressure: Pressure at the staggered nodes

        Returns:
            Eddy diffusivity
        """
        critical_reynolds_number: float = self.config["energy"].getfloat(
            "critical_reynolds_number"
        )
        eddy_diffusivity: np.ndarray = np.zeros(self.mesh.basic.number)
        reynolds_number: np.ndarray = self.reynolds_number()

        eddy_diffusivity[reynolds_number <= critical_reynolds_number] = self.velocity_viscous()[
            reynolds_number <= critical_reynolds_number
        ]
        eddy_diffusivity[reynolds_number > critical_reynolds_number] = self.velocity_inviscid()[
            reynolds_number > critical_reynolds_number
        ]
        eddy_diffusivity *= self.mesh.basic.mixing_length
        logger.info("eddy_diffusivity = %s", eddy_diffusivity)

        return eddy_diffusivity


class MyConfigParser(ConfigParser):
    """A configuration parser with some default options

    Args:
        *filenames: Filenames of one or several configuration files
    """

    getpath: Callable[..., Path]  # For typing.

    def __init__(self, *filenames):
        kwargs: dict = {"comment_prefixes": ("#",), "converters": {"path": Path}}
        super().__init__(**kwargs)
        self.read(filenames)
