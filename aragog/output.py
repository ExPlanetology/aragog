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
"""Output"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import numpy.typing as npt
from scipy.optimize import OptimizeResult

from aragog import __version__
from aragog.parser import Parameters
from aragog.solver import Evaluator
from aragog.utilities import FloatOrArray

logger: logging.Logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from aragog.solver import Solver, State


class Output:
    """Stores inputs and outputs of the models."""

    def __init__(self, solver: Solver):
        self.solver: Solver = solver
        self.parameters: Parameters = solver.parameters
        self.solution: OptimizeResult = self.solver.solution
        self.evaluator: Evaluator = self.solver.evaluator
        self.state: State = self.solver.state

    @property
    def shape_basic(self) -> npt.NDArray:
        """Shape of the basic data"""
        return np.array([self.solution.y.shape[0] + 1, self.solution.y.shape[1]])

    @property
    def shape_staggered(self) -> npt.NDArray:
        """Shape of the staggered data"""
        return self.solution.y.shape

    @property
    def convective_heat_flux_basic(self) -> npt.NDArray:
        """Convective heat flux"""
        return self.state.convective_heat_flux() * self.parameters.scalings.heat_flux

    @property
    def density_basic(self) -> npt.NDArray:
        """Density"""
        return (
            self.state.phase_basic.density()
            * np.ones(self.shape_basic)
            * self.parameters.scalings.density
        )

    @property
    def dTdr(self) -> npt.NDArray:
        """dTdr"""
        return self.solver.state.dTdr() * self.parameters.scalings.temperature_gradient

    @property
    def dTdrs(self) -> npt.NDArray:
        """dTdrs"""
        return (  # FIXME
            self.state.phase_basic.dTdrs() * self.parameters.scalings.temperature_gradient
        )

    @property
    def heat_capacity_basic(self) -> npt.NDArray:
        """Heat capacity"""
        return (
            self.state.phase_basic.heat_capacity()
            * np.ones(self.shape_basic)
            * self.parameters.scalings.heat_capacity
        )

    @property
    def heating(self) -> npt.NDArray:
        """Internal heat generation at staggered nodes"""
        return self.state.heating * self.parameters.scalings.power_per_mass

    @property
    def heating_radio(self) -> npt.NDArray:
        """Internal heat generation from radioactive decay at staggered nodes"""
        return self.state.heating_radio * self.parameters.scalings.power_per_mass

    @property
    def heating_tidal(self) -> npt.NDArray:
        """Internal heat generation from tidal heat dissipation at staggered nodes"""
        return self.state.heating_tidal * self.parameters.scalings.power_per_mass

    @property
    def liquidus_K_staggered(self) -> npt.NDArray:
        """Liquidus"""
        return self.evaluator.phases.mixed.liquidus() * self.parameters.scalings.temperature

    @property
    def melt_fraction_staggered(self) -> FloatOrArray:
        """Melt fraction on the staggered mesh"""
        return self.state.phase_staggered.melt_fraction()

    @property
    def melt_fraction_basic(self) -> FloatOrArray:
        """Melt fraction on the basic mesh"""
        return self.state.phase_basic.melt_fraction()

    @property
    def rheological_front(self) -> float:
        """Rheological front at the last solve iteration given user defined threshold.
        It is defined as a dimensionless distance with respect to the outer radius.
        """

        _phi_global = float(self.melt_fraction_global)

        # if global melt fraction is close to one everywhere (magma ocean) rf is the inner radius
        if _phi_global > 0.99:
            rf: float = self.evaluator.mesh.basic.radii[0]
        # if global melt fraction is close to zero everywhere (solidified) rf is the outer radius
        elif _phi_global < 0.01:
            rf = self.evaluator.mesh.basic.radii[-1]
        # general case
        else:
            idx = np.argmin(
                np.abs(
                    self.melt_fraction_basic[:, -1]
                    - self.parameters.phase_mixed.rheological_transition_melt_fraction
                )
            )
            rf = self.evaluator.mesh.basic.radii[idx]

        # Return dimensionless rheological front
        return (self.evaluator.mesh.basic.radii[-1] - rf) / self.evaluator.mesh.basic.radii[-1]

    @property
    def melt_fraction_global(self) -> float:
        """Volume-averaged melt fraction"""
        return self.evaluator.mesh.volume_average(self.melt_fraction_staggered[:, -1])

    @property
    def radii_km_basic(self) -> npt.NDArray:
        """Radii of the basic mesh in km"""
        return self.evaluator.mesh.basic.radii * self.parameters.scalings.radius * 1.0e-3

    @property
    def pressure_GPa_basic(self) -> npt.NDArray:
        """Pressure of the basic mesh in GPa"""
        return self.evaluator.mesh.basic.eos.pressure * self.parameters.scalings.pressure * 1.0e-9

    @property
    def pressure_GPa_staggered(self) -> npt.NDArray:
        """Pressure of the staggered mesh in GPa"""
        return (
            self.evaluator.mesh.staggered.eos.pressure * self.parameters.scalings.pressure * 1.0e-9
        )

    @property
    def mantle_mass(self) -> float:
        """Mantle mass computed from the AdamsWilliamsonEOS"""
        return (
            self.evaluator.mesh.basic.eos.get_mass_within_radii(
                self.evaluator.mesh.basic.outer_boundary
            )
            * self.parameters.scalings.density
            * np.power(self.parameters.scalings.radius, 3)
        )

    @property
    def mass_staggered(self) -> npt.NDArray:
        """Mass of each layer on staggered mesh"""
        return (
            # shells centred on staggered nodes
            self.evaluator.mesh.staggered.eos.get_mass_within_shell(
                # shell upper and lower radii set by basic nodes
                self.evaluator.mesh.basic.radii
            )
            * self.parameters.scalings.density
            * np.power(self.parameters.scalings.radius, 3)
        )

    @property
    def core_mass(self) -> float:
        """Core mass computed with constant density"""

        # core radius
        R_core = self.evaluator.mesh.basic.inner_boundary * self.parameters.scalings.radius

        # core volume
        volume = 4 * np.pi * (R_core**3) / 3

        # core density
        rho = self.parameters.scalings.density * self.parameters.boundary_conditions.core_density

        # core mass
        return rho * volume

    @property
    def solidus_K_staggered(self) -> npt.NDArray:
        """Solidus"""
        return self.evaluator.phases.mixed.solidus() * self.parameters.scalings.temperature

    @property
    def super_adiabatic_temperature_gradient_basic(self) -> npt.NDArray:
        """Super adiabatic temperature gradient"""
        return (
            self.state.super_adiabatic_temperature_gradient
            * self.parameters.scalings.temperature_gradient
        )

    @property
    def temperature_K_basic(self) -> npt.NDArray:
        """Temperature of the basic mesh in K"""
        return self.state.temperature_basic * self.parameters.scalings.temperature

    @property
    def temperature_K_staggered(self) -> npt.NDArray:
        """Temperature of the staggered mesh in K"""
        return self.solver.temperature_staggered

    @property
    def thermal_expansivity_basic(self) -> npt.NDArray:
        """Thermal expansivity"""
        return (
            self.state.phase_basic.thermal_expansivity()
            * np.ones(self.shape_basic)
            * self.parameters.scalings.thermal_expansivity
        )

    @property
    def log10_viscosity_basic(self) -> npt.NDArray:
        """Viscosity of the basic mesh"""
        return np.log10(
            self.state.phase_basic.viscosity()
            * self.parameters.scalings.viscosity
            * np.ones(self.shape_basic)
        )

    @property
    def log10_viscosity_staggered(self) -> npt.NDArray:
        """Viscosity of the staggered mesh"""
        return np.log10(
            self.state.phase_staggered.viscosity()
            * self.parameters.scalings.viscosity
            * np.ones(self.shape_staggered)
        )

    @property
    def solution_top_temperature(self) -> float:
        """Solution (last iteration) temperature at the top of the domain (planet surface)"""
        return self.temperature_K_basic[-1, -1]

    @property
    def times(self) -> npt.NDArray:
        """Times in years"""
        return self.solution.t * self.parameters.scalings.time_years

    @property
    def time_range(self) -> float:
        return self.times[-1] - self.times[0]

    def write_at_time(self, file_path: str, tidx: int = -1) -> None:
        """Write the state of the model at a particular time to a NetCDF4 file on the disk.

        Args:
            file_path: Path to the output file
            tidx: Index on the time axis at which to access the data
        """

        logger.debug("Writing i=%d NetCDF file to %s", tidx, file_path)

        # Update the state
        assert self.solution is not None
        self.state.update(self.solution.y, self.solution.t)

        # Open the dataset
        ds: nc.Dataset = nc.Dataset(file_path, mode="w")

        # Metadata
        ds.description = "Aragog output data"
        ds.argog_version = __version__

        # Function to save scalar quantities
        def _add_scalar_variable(key: str, value: float, units: str):
            ds.createVariable(key, np.float64)
            ds[key][0] = float(value)
            ds[key].units = units

        # Save scalar quantities
        _add_scalar_variable("time", self.times[tidx], "yr")
        _add_scalar_variable("phi_global", self.melt_fraction_global, "")
        _add_scalar_variable("mantle_mass", self.mantle_mass, "kg")
        _add_scalar_variable("rheo_front", self.rheological_front, "")

        # Create dimensions for mesh quantities
        ds.createDimension("basic", self.shape_basic[0])
        ds.createDimension("staggered", self.shape_staggered[0])

        # Function to save mesh quantities
        def _add_mesh_variable(key: str, some_property: Any, units: str):
            if key[-2:] == "_b":
                mesh = "basic"
            elif key[-2:] == "_s":
                mesh = "staggered"
            else:
                raise KeyError(f"NetCDF variable name must end in _b or _s: {key}")
            ds.createVariable(
                key,
                np.float64,
                (mesh,),
            )
            ds[key][:] = some_property[:, tidx]
            ds[key].units = units

        # Save mesh quantities
        _add_mesh_variable("radius_b", self.radii_km_basic, "km")
        _add_mesh_variable("pres_b", self.pressure_GPa_basic, "GPa")
        _add_mesh_variable("temp_b", self.temperature_K_basic, "K")
        _add_mesh_variable("phi_b", self.melt_fraction_basic, "")
        _add_mesh_variable("Fconv_b", self.convective_heat_flux_basic, "W m-2")
        _add_mesh_variable("log10visc_b", self.log10_viscosity_basic, "Pa s")
        _add_mesh_variable("log10visc_s", self.log10_viscosity_staggered, "Pa s")
        _add_mesh_variable("density_b", self.density_basic, "kg m-3")
        _add_mesh_variable("heatcap_b", self.heat_capacity_basic, "J kg-1 K-1")
        _add_mesh_variable("mass_s", self.mass_staggered, "kg")
        _add_mesh_variable("Hradio_s", self.heating_radio, "W kg-1")
        _add_mesh_variable("Htidal_s", self.heating_tidal, "W kg-1")
        _add_mesh_variable("Htotal_s", self.heating, "W kg-1")

        # Close the dataset
        ds.close()

    def plot(self, num_lines: int = 11, figsize: tuple = (25, 10)) -> None:
        """Plots the solution with labelled lines according to time.

        Args:
            num_lines: Number of lines to plot. Defaults to 11.
            figsize: Size of the figure. Defaults to (25, 10).
        """
        assert self.solution is not None

        self.state.update(self.solution.y, self.solution.t)

        _, axs = plt.subplots(1, 10, sharey=True, figsize=figsize)

        # Ensure there are at least 2 lines to plot (first and last).
        num_lines = max(2, num_lines)

        # Calculate the time step based on the total number of lines.
        time_step: float = self.time_range / (num_lines - 1)

        # plot temperature
        try:
            axs[0].scatter(self.liquidus_K_staggered, self.pressure_GPa_staggered)
            axs[0].scatter(self.solidus_K_staggered, self.pressure_GPa_staggered)
        except AttributeError:
            pass

        # Plot the first line.
        def plot_times(ax, x: npt.NDArray, y: npt.NDArray) -> None:
            # If `x` is a float, create an array of the same length as `y` filled with `x`
            if np.isscalar(x):
                x = np.full((len(y), len(self.times)), x, dtype=np.float64)
            elif x.ndim == 1:
                # If `x` is 1D, reshape it or repeat it across the time dimension
                x = np.tile(x[:, np.newaxis], (1, len(self.times)))

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

        plot_times(axs[3], self.convective_heat_flux_basic, self.pressure_GPa_basic)
        axs[3].set_xlabel("Convective heat flux")
        axs[3].set_title("Convective heat flux")

        plot_times(
            axs[4],
            self.super_adiabatic_temperature_gradient_basic,
            self.pressure_GPa_basic,
        )
        axs[4].set_xlabel("Super adiabatic temperature gradient")
        axs[4].set_title("Super adiabatic temperature gradient")

        plot_times(
            axs[5],
            self.dTdr,
            self.pressure_GPa_basic,
        )
        axs[5].set_xlabel("dTdr")
        axs[5].set_title("dTdr")

        plot_times(
            axs[6],
            self.dTdrs,
            self.pressure_GPa_basic,
        )
        axs[6].set_xlabel("dTdrs")
        axs[6].set_title("dTdrs")

        plot_times(
            axs[7],
            self.density_basic,
            self.pressure_GPa_basic,
        )
        axs[7].set_xlabel("Density")
        axs[7].set_title("Density")

        plot_times(
            axs[8],
            self.heat_capacity_basic,
            self.pressure_GPa_basic,
        )
        axs[8].set_xlabel("Heat capacity")
        axs[8].set_title("Heat capacity")

        plot_times(
            axs[9],
            self.thermal_expansivity_basic,
            self.pressure_GPa_basic,
        )
        axs[9].set_xlabel("Thermal expansivity")
        axs[9].set_title("Thermal expansivity")

        # Shrink current axis by 20% to allow space for the legend.
        # box = ax.get_position()
        # ax.set_position((box.x0, box.y0, box.width * 0.8, box.height))

        # axs[0].grid(True)

        legend = axs[2].legend()  # (loc="center left", bbox_to_anchor=(1, 0.5))
        legend.set_title("Time (yr)")
        plt.gca().invert_yaxis()

        plt.show()
