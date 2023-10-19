"""Solver."""

from __future__ import annotations

import configparser
import logging
from configparser import SectionProxy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import OptimizeResult

from spider import STEFAN_BOLTZMANN_CONSTANT, YEAR_IN_SECONDS
from spider.energy import total_heat_flux
from spider.mesh import SpiderMesh
from spider.phase import ConstantPhase, PhaseABC

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class SpiderSolver:
    filename: Union[str, Path]
    root_path: Union[str, Path] = ""
    mesh: SpiderMesh = field(init=False)
    phase: PhaseABC = field(init=False)
    initial_temperature: np.ndarray = field(init=False)
    initial_time: float = field(init=False, default=0)
    end_time: float = field(init=False, default=0)
    _solution: OptimizeResult = field(init=False, default_factory=OptimizeResult)

    def __post_init__(self):
        logger.info("Creating a SPIDER model")
        self.config: configparser.ConfigParser = MyConfigParser(self.filename)
        self.root: Path = Path(self.root_path)
        # Set the mesh.
        mesh: SectionProxy = self.config["mesh"]
        self.mesh = SpiderMesh.uniform_radii(
            mesh.getfloat("inner_radius"),
            mesh.getfloat("outer_radius"),
            mesh.getint("number_of_nodes"),
        )
        # Set the phase.
        gravitational_acceleration_value: float = self.config.getfloat(
            "DEFAULT", "gravitational_acceleration"
        )
        # FIXME: Currently only uses the liquid phase. Also don't assume constant phase.
        phase: SectionProxy = self.config["phase_liquid"]
        self.phase = ConstantPhase(
            density_value=phase.getfloat("density"),
            gravitational_acceleration_value=gravitational_acceleration_value,
            heat_capacity_value=phase.getfloat("heat_capacity"),
            thermal_conductivity_value=phase.getfloat("thermal_conductivity"),
            thermal_expansivity_value=phase.getfloat("thermal_expansivity"),
            log10_viscosity_value=phase.getfloat("log10_viscosity"),
        )
        # Set the time stepping.
        self.initial_time = self.config.getfloat("timestepping", "start_time")
        self.end_time = self.config.getfloat("timestepping", "end_time")
        # Set the initial condition.
        initial: SectionProxy = self.config["initial_condition"]
        self.initial_temperature = np.linspace(
            initial.getfloat("basal_temperature"),
            initial.getfloat("surface_temperature"),
            self.mesh.staggered.number,
        )

    @property
    def solution(self) -> OptimizeResult:
        """The solution."""
        return self._solution

    def dTdt(
        self,
        time: float,
        temperature: np.ndarray,
        mesh: SpiderMesh,
        phase: PhaseABC,
        pressure: np.ndarray,
    ) -> np.ndarray:
        """dT/dt at the staggered nodes.

        Args:
            time: Time.
            temperature: Temperature at the staggered nodes.
            mesh: Mesh.
            phase: Phase.
            pressure: Pressure at the staggered nodes.

        Returns:
            dT/dt at the staggered nodes.
        """
        energy: SectionProxy = self.config["energy"]
        heat_flux: np.ndarray = total_heat_flux(energy, mesh, phase, temperature, pressure)

        # TODO: Clean up boundary conditions.
        # No heat flux from the core.
        heat_flux[0] = 0
        # Blackbody cooling.
        heat_flux[-1] = (
            self.config.getfloat("boundary_conditions", "emissivity")
            * STEFAN_BOLTZMANN_CONSTANT
            * (
                mesh.quantity_at_basic_nodes(temperature)[-1] ** 4
                - self.config.getfloat("boundary_conditions", "equilibrium_temperature") ** 4
            )
        )

        energy_flux: np.ndarray = heat_flux * mesh.basic.area
        logger.info("energy_flux = %s", energy_flux)

        delta_energy_flux: np.ndarray = np.diff(energy_flux)
        logger.info("delta_energy_flux = %s", delta_energy_flux)
        capacitance: np.ndarray = (
            phase.heat_capacity(temperature, pressure)
            * phase.density(temperature, pressure)
            * mesh.basic.volume
        )

        dTdt: np.ndarray = -delta_energy_flux / capacitance
        logger.info("dTdt = %s", dTdt)

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
        legend.set_title("Time (Myr)")
        plt.show()

    def solve(self) -> None:
        """Solves the system of ODEs to determine the interior temperature profile."""
        self._solution = solve_ivp(
            self.dTdt,
            (self.initial_time, self.end_time),
            self.initial_temperature,
            method="BDF",
            vectorized=False,  # TODO: True would speed up BDF according to the documentation.
            args=(self.mesh, self.phase, self.initial_temperature),  # FIXME: Should be pressure.
        )
        logger.info(self.solution)


class MyConfigParser(configparser.ConfigParser):
    """A configuration parser with some default options.

    Args:
        *filenames: Filenames of one or several configuration files.
    """

    getpath: Callable[..., Path]  # For typing.

    def __init__(self, *filenames):
        kwargs: dict = {"comment_prefixes": ("#",), "converters": {"path": Path}}
        super().__init__(**kwargs)
        self.read(filenames)
