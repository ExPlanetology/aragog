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
"""Simple tests to recover the thermal structure of molten, solid, and mixed phase interiors."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from spider import (
    CFG_TEST_DATA,
    REFERENCE_TEST_DATA,
    SpiderSolver,
    __version__,
    debug_logger,
)

logger: logging.Logger = debug_logger()
logger.setLevel(logging.INFO)

ATOL: float = 1e-8
RTOL: float = 1e-8


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


def test_liquid_no_heating():
    """Cooling of a purely molten magma ocean."""

    solver: SpiderSolver = SpiderSolver("abe_liquid.cfg", CFG_TEST_DATA)
    solver.initialize()
    solver.solve()
    calculated: np.ndarray = solver.get_temperature()[:, -1]
    # spider_solver.plot()
    # np.savetxt("abe_liquid_no_heating.dat", calculated)

    expected: np.ndarray = np.loadtxt(REFERENCE_TEST_DATA / Path("abe_liquid_no_heating.dat"))
    logger.info("calculated = %s", calculated)
    logger.info("expected = %s", expected)

    assert np.isclose(calculated, expected, atol=ATOL, rtol=RTOL).all()


def test_solid_no_heating():
    """Cooling of a purely solid mantle."""

    solver: SpiderSolver = SpiderSolver(Path("abe_solid.cfg"), CFG_TEST_DATA)
    solver.initialize()
    solver.solve()
    calculated: np.ndarray = solver.get_temperature()[:, -1]
    # solver.plot()
    # np.savetxt("abe_solid_no_heating.dat", calculated)

    expected: np.ndarray = np.loadtxt(REFERENCE_TEST_DATA / Path("abe_solid_no_heating.dat"))
    logger.info("calculated = %s", calculated)
    logger.info("expected = %s", expected)

    assert np.isclose(calculated, expected, atol=ATOL, rtol=RTOL).all()


def test_solid_with_heating():
    """Cooling of a purely solid mantle with radiogenic heating."""

    solver: SpiderSolver = SpiderSolver(Path("abe_solid.cfg"), CFG_TEST_DATA)
    solver.parameters.energy.radionuclides = True
    solver.initialize()
    solver.solve()
    calculated: np.ndarray = solver.get_temperature()[:, -1]
    # solver.plot()
    np.savetxt("abe_solid_with_heating.dat", calculated)

    expected: np.ndarray = np.loadtxt(REFERENCE_TEST_DATA / Path("abe_solid_with_heating.dat"))
    logger.info("calculated = %s", calculated)
    logger.info("expected = %s", expected)

    assert np.isclose(calculated, expected, atol=ATOL, rtol=RTOL).all()


@pytest.mark.skip(reason="mixed phases not yet implemented")
def test_mixed():
    """Test Abe (1993."""

    spider_solver: SpiderSolver = SpiderSolver(Path("abe_mixed.cfg"), CFG_TEST_DATA)
    spider_solver.initialize()
    spider_solver.solve()
    calculated: np.ndarray = spider_solver.get_temperature()[:, -1]
    spider_solver.plot()
    # np.savetxt("abe_mixed_no_heating.dat", calculated)

    expected: np.ndarray = np.loadtxt(REFERENCE_TEST_DATA / Path("abe_mixed_no_heating.txt"))
    logger.info("calculated = %s", calculated)
    logger.info("expected = %s", expected)

    assert np.isclose(calculated, expected, atol=ATOL, rtol=RTOL).all()


@pytest.mark.skip(reason="mixed phases not yet implemented")
def test_mixed_lookup():
    """Test Abe (1993."""

    spider_solver: SpiderSolver = SpiderSolver(Path("abe_mixed_lookup.cfg"), CFG_TEST_DATA)
    spider_solver.initialize()
    spider_solver.solve()
    calculated: np.ndarray = spider_solver.get_temperature()[:, -1]
    # spider_solver.plot()
    # np.savetxt("testout.dat", calculated)

    expected: np.ndarray = np.loadtxt(REFERENCE_TEST_DATA / Path("abe_solid_with_heating.txt"))
    logger.info("calculated = %s", calculated)
    logger.info("expected = %s", expected)

    assert np.isclose(calculated, expected, atol=ATOL, rtol=RTOL).all()
