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
"""Tests properties of phases."""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt

from aragog import Solver, __version__, debug_logger
from aragog.interfaces import MixedPhaseEvaluatorProtocol, PhaseEvaluatorProtocol
from aragog.utilities import FloatOrArray

logger: logging.Logger = debug_logger()
logger.setLevel(logging.INFO)

# Temperature and pressure for surface and near the CMB
temperature: npt.NDArray = np.atleast_2d([1500, 4000]).T
pressure: npt.NDArray = np.atleast_2d([0, 135e9]).T


def test_version():
    """Test version."""
    assert __version__ == "0.2.3-alpha"


def test_liquid_constant_properties(helper):
    """Constant liquid properties"""

    with helper.get_cfg_file("abe_liquid.cfg") as cfg_file:
        solver: Solver = Solver.from_file(cfg_file)
    solver.parameters.phase_mixed.phase = "liquid"
    solver.initialize()

    phase: PhaseEvaluatorProtocol = solver.evaluator.phases.active

    temperature_scaled = temperature / solver.parameters.scalings.temperature
    pressure_scaled = pressure / solver.parameters.scalings.pressure

    phase.set_temperature(temperature_scaled)
    phase.set_pressure(pressure_scaled)
    phase.update()

    density: FloatOrArray = phase.density()
    assert np.isclose(density, 1, atol=helper.atol, rtol=helper.rtol).all()

    heat_capacity: FloatOrArray = phase.heat_capacity()
    assert np.isclose(heat_capacity, 981415.05391486, atol=helper.atol, rtol=helper.rtol).all()

    thermal_conductivity: FloatOrArray = phase.thermal_conductivity()
    assert np.isclose(
        thermal_conductivity, 7.630297519858265e-08, atol=helper.atol, rtol=helper.rtol
    ).all()

    thermal_expansivity: FloatOrArray = phase.thermal_expansivity()
    assert np.isclose(thermal_expansivity, 0.04, atol=helper.atol, rtol=helper.rtol).all()

    viscosity: FloatOrArray = phase.viscosity()
    assert np.isclose(viscosity, 1.9436979006540116e-09, atol=helper.atol, rtol=helper.rtol).all()


def test_solid_constant_properties(helper):
    """Constant solid properties"""

    with helper.get_cfg_file("abe_mixed.cfg") as cfg_file:
        solver: Solver = Solver.from_file(cfg_file)
    solver.parameters.phase_mixed.phase = "solid"
    solver.initialize()

    phase: PhaseEvaluatorProtocol = solver.evaluator.phases.active

    temperature_scaled = temperature / solver.parameters.scalings.temperature
    pressure_scaled = pressure / solver.parameters.scalings.pressure

    phase.set_temperature(temperature_scaled)
    phase.set_pressure(pressure_scaled)
    phase.update()

    density: FloatOrArray = phase.density()
    assert np.isclose(density, 1.05, atol=helper.atol, rtol=helper.rtol).all()

    heat_capacity: FloatOrArray = phase.heat_capacity()
    assert np.isclose(heat_capacity, 981415.05391486, atol=helper.atol, rtol=helper.rtol).all()

    thermal_conductivity: FloatOrArray = phase.thermal_conductivity()
    assert np.isclose(
        thermal_conductivity, 7.630297519858265e-08, atol=helper.atol, rtol=helper.rtol
    ).all()

    thermal_expansivity: FloatOrArray = phase.thermal_expansivity()
    assert np.isclose(thermal_expansivity, 0.04, atol=helper.atol, rtol=helper.rtol).all()

    viscosity: FloatOrArray = phase.viscosity()
    assert np.isclose(viscosity, 1.9436979e10, atol=helper.atol, rtol=helper.rtol).all()


def test_lookup_property_1D(helper):
    """1D lookup property"""

    with helper.get_cfg_file("abe_mixed.cfg") as cfg_file:
        solver: Solver = Solver.from_file(cfg_file)
    solver.initialize()
    phase: MixedPhaseEvaluatorProtocol = solver.evaluator.phases.composite

    temperature_scaled = temperature / solver.parameters.scalings.temperature
    pressure_scaled = pressure / solver.parameters.scalings.pressure

    phase.set_temperature(temperature_scaled)
    phase.set_pressure(pressure_scaled)
    phase.update()

    solidus: npt.NDArray = phase.solidus()
    solidus_target: npt.NDArray = np.atleast_2d([0.34515095, 1.05180909]).T
    assert np.isclose(solidus, solidus_target, atol=helper.atol, rtol=helper.rtol).all()

    liquidus: npt.NDArray = phase.liquidus()
    liquidus_target: npt.NDArray = np.atleast_2d([0.4500425, 1.15670029]).T
    assert np.isclose(liquidus, liquidus_target, atol=helper.atol, rtol=helper.rtol).all()


def test_lookup_property_2D(helper):
    """2D lookup property"""

    with helper.get_cfg_file("abe_mixed_lookup.cfg") as cfg_file:
        solver: Solver = Solver.from_file(cfg_file)
    solver.parameters.phase_mixed.phase = "liquid"
    solver.initialize()

    phase: PhaseEvaluatorProtocol = solver.evaluator.phases.active

    temperature_: npt.NDArray = np.atleast_2d(
        [[1000, 1500, 2500, 2500, 2500], [1250, 2000, 2000, 2250, 2500]]
    ).T
    pressure_: npt.NDArray = np.atleast_2d([0, 1.4e11, 0, 1.4e11, 0.7e11]).T

    temperature_scaled = temperature_ / solver.parameters.scalings.temperature
    pressure_scaled = pressure_ / solver.parameters.scalings.pressure

    phase.set_temperature(temperature_scaled)
    phase.set_pressure(pressure_scaled)
    phase.update()

    density_melt: FloatOrArray = phase.density()
    logger.debug("density_melt = %s", density_melt)

    density_melt_target: npt.NDArray = np.atleast_2d(
        [[0.5, 0.5625, 0.3125, 0.4375, 0.375], [0.46875, 0.5, 0.375, 0.46875, 0.375]]
    ).T
    assert np.isclose(density_melt, density_melt_target, atol=helper.atol, rtol=helper.rtol).all()


def test_mixed_density(helper):
    """Mixed phase density"""

    with helper.get_cfg_file("abe_mixed.cfg") as cfg_file:
        solver: Solver = Solver.from_file(cfg_file)
    solver.initialize()
    phase: PhaseEvaluatorProtocol = solver.evaluator.phases.active

    # Chosen to be the melting curve, i.e. 50% melt fraction
    temperature_: npt.NDArray = np.atleast_2d([1590.3869054958254, 4521.708837963126]).T
    pressure_: npt.NDArray = np.atleast_2d([0, 1.4e11]).T

    temperature_scaled = temperature_ / solver.parameters.scalings.temperature
    pressure_scaled = pressure_ / solver.parameters.scalings.pressure

    phase.set_temperature(temperature_scaled)
    phase.set_pressure(pressure_scaled)
    phase.update()

    melt_fraction: FloatOrArray = phase.melt_fraction()
    logger.debug("melt_fraction = %s", melt_fraction)

    density_mixed: FloatOrArray = phase.density()
    logger.debug("density_mixed = %s", density_mixed)

    density_mixed_target: npt.NDArray = np.atleast_2d([1.02439141, 1.02438914]).T
    assert np.isclose(
        density_mixed, density_mixed_target, atol=helper.atol, rtol=helper.rtol
    ).all()
