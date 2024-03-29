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
"""Output"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import OptimizeResult

from spider.core import SpiderData

logger: logging.Logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from spider.solver import SpiderSolver, State


class Output:
    """Stores inputs and outputs of the models."""

    def __init__(self, solver: SpiderSolver):
        self.solver: SpiderSolver = solver
        self.solution: OptimizeResult = self.solver.solution
        self.data: SpiderData = self.solver.data
        self.state: State = self.solver.state

    @property
    def shape_basic(self) -> np.ndarray:
        """Shape of the basic data"""
        return np.array([self.solution.y.shape[0] + 1, self.solution.y.shape[1]])

    @property
    def shape_staggered(self) -> np.ndarray:
        """Shape of the staggered data"""
        return self.solution.y.shape

    @property
    def convective_heat_flux_basic(self) -> np.ndarray:
        """Convective heat flux"""
        return self.state.convective_heat_flux() * self.data.parameters.scalings.heat_flux

    @property
    def density_basic(self) -> np.ndarray:
        """Density"""
        return (
            self.solver.state.phase_basic.density
            * np.ones(self.shape_basic)
            * self.data.parameters.scalings.density
        )

    @property
    def liquidus_K_staggered(self) -> np.ndarray:
        """Liquidus"""
        return (
            self.data.phase.liquidus(self.solution.y, self.data.mesh.staggered.eos.pressure)
            * self.data.parameters.scalings.temperature
        )

    @property
    def melt_fraction_staggered(self) -> np.ndarray:
        """Melt fraction"""
        return self.solver.data.phase.melt_fraction(
            self.solution.y, self.data.mesh.staggered.eos.pressure
        ) * np.ones(self.shape_staggered)

    @property
    def radii_km_basic(self) -> np.ndarray:
        """Radii of the basic mesh in km"""
        return self.data.mesh.basic.radii * self.data.parameters.scalings.radius * 1.0e-3

    @property
    def pressure_GPa_basic(self) -> np.ndarray:
        """Pressure of the basic mesh in GPa"""
        return self.data.mesh.basic.eos.pressure * self.data.parameters.scalings.pressure * 1.0e-9

    @property
    def pressure_GPa_staggered(self) -> np.ndarray:
        """Pressure of the staggered mesh in GPa"""
        return (
            self.data.mesh.staggered.eos.pressure * self.data.parameters.scalings.pressure * 1.0e-9
        )

    @property
    def solidus_K_staggered(self) -> np.ndarray:
        """Solidus"""
        return (
            self.data.phase.solidus(self.solution.y, self.data.mesh.staggered.eos.pressure)
            * self.data.parameters.scalings.temperature
        )

    @property
    def temperature_K_basic(self) -> np.ndarray:
        """Temperature of the basic mesh in K"""
        return self.state.temperature_basic * self.data.parameters.scalings.temperature

    @property
    def temperature_K_staggered(self) -> np.ndarray:
        """Temperature of the staggered mesh in K"""
        return self.solver.temperature_staggered

    @property
    def log10_viscosity_basic(self) -> np.ndarray:
        """Viscosity of the basic mesh"""
        return np.log10(
            self.state.phase_basic.viscosity
            * self.data.parameters.scalings.viscosity
            * np.ones(self.shape_basic)
        )

    @property
    def times(self) -> np.ndarray:
        """Times in years"""
        return self.solution.t * self.data.parameters.scalings.time_years

    @property
    def time_range(self) -> float:
        return self.times[-1] - self.times[0]

    def plot(self, num_lines: int = 11) -> None:
        """Plots the solution with labelled lines according to time.

        Args:
            num_lines: Number of lines to plot. Defaults to 11.
        """
        assert self.solution is not None

        self.state.update(self.solution.y, self.solution.t)

        _, axs = plt.subplots(1, 4, sharey=True)

        # Ensure there are at least 2 lines to plot (first and last).
        num_lines = max(2, num_lines)

        # Calculate the time step based on the total number of lines.
        time_step: float = self.time_range / (num_lines - 1)

        # plot temperature
        try:
            axs[0].plot(self.liquidus_K_staggered, self.pressure_GPa_staggered, "k--")
            axs[0].plot(self.solidus_K_staggered, self.pressure_GPa_staggered, "k--")
        except AttributeError:
            pass

        # Plot the first line.
        def plot_times(ax, x: np.ndarray, y: np.ndarray) -> None:
            label_first: str = f"{self.times[0]:.2f}"
            ax.plot(x[:, 0], y, label=label_first)

            # Loop through the selected lines and plot each with a label.
            for i in range(1, num_lines - 1):
                desired_time: float = self.times[0] + i * time_step
                # Find the closest available time step.
                closest_time_index: np.intp = np.argmin(np.abs(self.times - desired_time))
                time: float = self.times[closest_time_index]
                label: str = f"{time:.2f}"  # Create a label based on the time.
                ax.plot(
                    x[:, closest_time_index],
                    y,
                    label=label,
                )

            # Plot the last line.
            times_end: float = self.times[-1]
            label_last: str = f"{times_end:.2f}"
            ax.plot(x[:, -1], y, label=label_last)

        plot_times(axs[0], self.temperature_K_basic, self.pressure_GPa_basic)
        axs[0].set_ylabel("Pressure (GPa)")
        axs[0].set_xlabel("Temperature (K)")
        axs[0].set_title("Temperature")

        plot_times(axs[1], self.melt_fraction_staggered, self.pressure_GPa_staggered)
        axs[1].set_xlabel("Melt fraction")
        axs[1].set_title("Melt fraction")

        plot_times(axs[2], self.log10_viscosity_basic, self.pressure_GPa_basic)
        axs[2].set_xlabel("Log10(viscosity)")
        axs[2].set_title("Log10(viscosity)")

        plot_times(axs[3], np.log10(self.convective_heat_flux_basic), self.pressure_GPa_basic)
        axs[3].set_xlabel("Convective heat flux")
        axs[3].set_title("Convective heat flux")

        # Shrink current axis by 20% to allow space for the legend.
        # box = ax.get_position()
        # ax.set_position((box.x0, box.y0, box.width * 0.8, box.height))

        # axs[0].grid(True)

        legend = axs[2].legend()  # (loc="center left", bbox_to_anchor=(1, 0.5))
        legend.set_title("Time (yr)")
        plt.gca().invert_yaxis()

        plt.show()
