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
"""Tests phases"""

# For testing we access some private members, so pylint: disable=W0212

from __future__ import annotations

import logging

import numpy as np

from spider import CFG_TEST_DATA, SpiderSolver, __version__, debug_logger
from spider.phase import PhaseEvaluatorProtocol
from spider.utilities import FloatOrArray

logger: logging.Logger = debug_logger()
logger.setLevel(logging.INFO)

ATOL: float = 1e-8
RTOL: float = 1e-8

# Temperature and pressure for surface and near the CMB
temperature: np.ndarray = np.array([1500, 4000]).reshape(-1, 1)
pressure: np.ndarray = np.array([0, 135e9]).reshape(-1, 1)


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


def test_liquid_constant_properties():
    """Constant liquid properties"""

    solver: SpiderSolver = SpiderSolver("abe_mixed.cfg", CFG_TEST_DATA)
    solver.initialize()
    phase: PhaseEvaluatorProtocol = solver.data._liquid

    temperature_scaled = temperature / solver.parameters.scalings.temperature
    pressure_scaled = pressure / solver.parameters.scalings.pressure

    density: FloatOrArray = phase.density(temperature_scaled, pressure_scaled)
    assert np.isclose(density, 1, atol=ATOL, rtol=RTOL).all()

    heat_capacity: FloatOrArray = phase.heat_capacity(temperature_scaled, pressure_scaled)
    assert np.isclose(heat_capacity, 981415.05391486, atol=ATOL, rtol=RTOL).all()

    thermal_conductivity: FloatOrArray = phase.thermal_conductivity(
        temperature_scaled, pressure_scaled
    )
    assert np.isclose(thermal_conductivity, 7.630297519858265e-08, atol=ATOL, rtol=RTOL).all()

    thermal_expansivity: FloatOrArray = phase.thermal_expansivity(
        temperature_scaled, pressure_scaled
    )
    assert np.isclose(thermal_expansivity, 0.04, atol=ATOL, rtol=RTOL).all()

    viscosity: FloatOrArray = phase.viscosity(temperature_scaled, pressure_scaled)
    assert np.isclose(viscosity, 1.9436979006540116e-09, atol=ATOL, rtol=RTOL).all()


def test_solid_constant_properties():
    """Constant solid properties"""

    solver: SpiderSolver = SpiderSolver("abe_mixed.cfg", CFG_TEST_DATA)
    solver.initialize()
    phase: PhaseEvaluatorProtocol = solver.data._solid

    temperature_scaled = temperature / solver.parameters.scalings.temperature
    pressure_scaled = pressure / solver.parameters.scalings.pressure

    density: FloatOrArray = phase.density(temperature_scaled, pressure_scaled)
    assert np.isclose(density, 1.05, atol=ATOL, rtol=RTOL).all()

    heat_capacity: FloatOrArray = phase.heat_capacity(temperature_scaled, pressure_scaled)
    assert np.isclose(heat_capacity, 981415.05391486, atol=ATOL, rtol=RTOL).all()

    thermal_conductivity: FloatOrArray = phase.thermal_conductivity(
        temperature_scaled, pressure_scaled
    )
    assert np.isclose(thermal_conductivity, 7.630297519858265e-08, atol=ATOL, rtol=RTOL).all()

    thermal_expansivity: FloatOrArray = phase.thermal_expansivity(
        temperature_scaled, pressure_scaled
    )
    assert np.isclose(thermal_expansivity, 0.04, atol=ATOL, rtol=RTOL).all()

    viscosity: FloatOrArray = phase.viscosity(temperature_scaled, pressure_scaled)
    assert np.isclose(viscosity, 1.9436979e10, atol=ATOL, rtol=RTOL).all()


def test_lookup_property_1D():
    """1D lookup property"""

    solver: SpiderSolver = SpiderSolver("abe_mixed_lookup.cfg", CFG_TEST_DATA)
    solver.initialize()
    phase: PhaseEvaluatorProtocol = solver.data._mixed

    temperature_scaled = temperature / solver.parameters.scalings.temperature
    pressure_scaled = pressure / solver.parameters.scalings.pressure

    solidus: np.ndarray = phase.solidus(temperature_scaled, pressure_scaled)
    solidus_target: np.ndarray = np.array([[0.34515095], [1.05180909]])
    assert np.isclose(solidus, solidus_target, atol=ATOL, rtol=RTOL).all()

    liquidus: np.ndarray = phase.liquidus(temperature_scaled, pressure_scaled)
    liquidus_target: np.ndarray = np.array([[0.4500425], [1.15670029]])
    assert np.isclose(liquidus, liquidus_target, atol=ATOL, rtol=RTOL).all()


def test_lookup_property_2D():
    """2D lookup property"""

    solver: SpiderSolver = SpiderSolver("abe_mixed_lookup.cfg", CFG_TEST_DATA)
    solver.initialize()
    phase: PhaseEvaluatorProtocol = solver.data._liquid

    temperature_: np.ndarray = np.array([1000, 1500, 2500, 2500, 2500])
    pressure_: np.ndarray = np.array([0, 1.4e11, 0, 1.4e11, 0.7e11])

    temperature_scaled = temperature_ / solver.parameters.scalings.temperature
    pressure_scaled = pressure_ / solver.parameters.scalings.pressure

    density_melt: FloatOrArray = phase.density(temperature_scaled, pressure_scaled)
    density_melt_target: np.ndarray = np.array([0.5, 0.5625, 0.3125, 0.4375, 0.375])
    assert np.isclose(density_melt, density_melt_target, atol=ATOL, rtol=RTOL).all()


def test_mixed_density():
    """Mixed phase density"""

    solver: SpiderSolver = SpiderSolver("abe_mixed.cfg", CFG_TEST_DATA)
    solver.initialize()
    phase: PhaseEvaluatorProtocol = solver.data._mixed

    # Chosen to be the melting curve, i.e. 50% melt fraction
    temperature_: np.ndarray = np.array([1590.3869054958254, 4521.708837963126]).reshape(-1, 1)
    pressure_: np.ndarray = np.array([0, 1.4e11]).reshape(-1, 1)

    temperature_scaled = temperature_ / solver.parameters.scalings.temperature
    pressure_scaled = pressure_ / solver.parameters.scalings.pressure

    density_melt: FloatOrArray = phase.density(temperature_scaled, pressure_scaled)
    density_melt_target: np.ndarray = np.array([1.02439024, 1.02439024])
    assert np.isclose(density_melt, density_melt_target, atol=ATOL, rtol=RTOL).all()
