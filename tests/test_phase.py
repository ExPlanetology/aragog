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

from __future__ import annotations

import logging

import numpy as np

from spider import CFG_TEST_DATA, SpiderSolver, __version__, debug_logger
from spider.phase import PhaseEvaluatorProtocol

logger: logging.Logger = debug_logger()
# logger.setLevel(logging.INFO)

ATOL: float = 1e-8
RTOL: float = 1e-8

# temperature and pressure for surface and near the CMB
temperature: np.ndarray = np.array([1500, 4000])
pressure: np.ndarray = np.array([0, 135e9])


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


def test_liquid_constant_properties():
    """Constant liquid properties"""

    solver: SpiderSolver = SpiderSolver("abe_mixed.cfg", CFG_TEST_DATA)
    solver.initialize()
    phase: PhaseEvaluatorProtocol = solver.data.liquid

    density: np.ndarray = phase.density(temperature, pressure)
    assert np.isclose(density, 1, atol=ATOL, rtol=RTOL).all()

    heat_capacity: np.ndarray = phase.heat_capacity(temperature, pressure)
    assert np.isclose(heat_capacity, 981415.05391486, atol=ATOL, rtol=RTOL).all()

    thermal_conductivity: np.ndarray = phase.thermal_conductivity(temperature, pressure)
    assert np.isclose(thermal_conductivity, 7.630297519858265e-08, atol=ATOL, rtol=RTOL).all()

    thermal_expansivity: np.ndarray = phase.thermal_expansivity(temperature, pressure)
    assert np.isclose(thermal_expansivity, 0.04, atol=ATOL, rtol=RTOL).all()

    viscosity: np.ndarray = phase.viscosity(temperature, pressure)
    assert np.isclose(viscosity, 1.9436979006540116e-09, atol=ATOL, rtol=RTOL).all()


def test_solid_constant_properties():
    """Constant solid properties"""

    solver: SpiderSolver = SpiderSolver("abe_mixed.cfg", CFG_TEST_DATA)
    solver.initialize()
    phase: PhaseEvaluatorProtocol = solver.data.solid

    density: np.ndarray = phase.density(temperature, pressure)
    assert np.isclose(density, 1.05, atol=ATOL, rtol=RTOL).all()

    heat_capacity: np.ndarray = phase.heat_capacity(temperature, pressure)
    assert np.isclose(heat_capacity, 981415.05391486, atol=ATOL, rtol=RTOL).all()

    thermal_conductivity: np.ndarray = phase.thermal_conductivity(temperature, pressure)
    assert np.isclose(thermal_conductivity, 7.630297519858265e-08, atol=ATOL, rtol=RTOL).all()

    thermal_expansivity: np.ndarray = phase.thermal_expansivity(temperature, pressure)
    assert np.isclose(thermal_expansivity, 0.04, atol=ATOL, rtol=RTOL).all()

    viscosity: np.ndarray = phase.viscosity(temperature, pressure)
    assert np.isclose(viscosity, 1.9436979e10, atol=ATOL, rtol=RTOL).all()
