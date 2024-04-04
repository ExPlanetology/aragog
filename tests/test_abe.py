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
"""Simple tests to recover the thermal structure of molten, solid, and mixed phase interiors."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from aragog import REFERENCE_TEST_DATA, Solver, __version__, debug_logger

# from aragog.output import Output
from aragog.utilities import profile_decorator

logger: logging.Logger = debug_logger()
logger.setLevel(logging.INFO)

ATOL: float = 1e-5
RTOL: float = 1e-5


def test_version():
    """Test version."""
    assert __version__ == "0.1.0-alpha"


@profile_decorator
def test_liquid_no_heating(helper):
    """Cooling of a purely molten magma ocean."""

    with helper.get_cfg_file("abe_liquid.cfg") as cfg_file:
        solver: Solver = Solver(cfg_file)

    solver.initialize()
    solver.solve()
    calculated: np.ndarray = solver.temperature_staggered[:, -1]
    # np.savetxt("abe_liquid_no_heating.dat", calculated)

    # output: Output = Output(solver)
    # output.plot()

    expected: np.ndarray = np.loadtxt(REFERENCE_TEST_DATA / Path("abe_liquid_no_heating.dat"))
    logger.info("calculated = %s", calculated)
    logger.info("expected = %s", expected)

    assert np.isclose(calculated, expected, atol=ATOL, rtol=RTOL).all()


@profile_decorator
def test_solid_no_heating(helper):
    """Cooling of a purely solid mantle."""

    with helper.get_cfg_file("abe_solid.cfg") as cfg_file:
        solver: Solver = Solver(cfg_file)

    solver.initialize()
    solver.solve()
    calculated: np.ndarray = solver.temperature_staggered[:, -1]
    # np.savetxt("abe_solid_no_heating.dat", calculated)

    # output: Output = Output(solver)
    # output.plot()

    expected: np.ndarray = np.loadtxt(REFERENCE_TEST_DATA / Path("abe_solid_no_heating.dat"))
    logger.info("calculated = %s", calculated)
    logger.info("expected = %s", expected)

    assert np.isclose(calculated, expected, atol=ATOL, rtol=RTOL).all()


@profile_decorator
def test_solid_with_heating(helper):
    """Cooling of a purely solid mantle with radiogenic heating."""

    with helper.get_cfg_file("abe_solid.cfg") as cfg_file:
        solver: Solver = Solver(cfg_file)

    solver.parameters.energy.radionuclides = True
    solver.initialize()
    solver.solve()
    calculated: np.ndarray = solver.temperature_staggered[:, -1]
    # np.savetxt("abe_solid_with_heating.dat", calculated)

    # output: Output = Output(solver)
    # output.plot()

    expected: np.ndarray = np.loadtxt(REFERENCE_TEST_DATA / Path("abe_solid_with_heating.dat"))
    logger.info("calculated = %s", calculated)
    logger.info("expected = %s", expected)

    assert np.isclose(calculated, expected, atol=ATOL, rtol=RTOL).all()


@profile_decorator
def test_mixed(helper):
    """Test Abe (1993."""

    with helper.get_cfg_file("abe_mixed.cfg") as cfg_file:
        solver: Solver = Solver(cfg_file)

    solver.initialize()
    solver.solve()
    calculated: np.ndarray = solver.temperature_staggered[:, -1]
    # np.savetxt("abe_mixed.dat", calculated)

    # output: Output = Output(solver)
    # output.plot()

    expected: np.ndarray = np.loadtxt(REFERENCE_TEST_DATA / Path("abe_mixed.dat"))
    logger.info("calculated = %s", calculated)
    logger.info("expected = %s", expected)

    assert np.isclose(calculated, expected, atol=ATOL, rtol=RTOL).all()
