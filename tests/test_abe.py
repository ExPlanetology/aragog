"""Tests for Abe (1993) model.

See the LICENSE file for licensing information.

"""

import logging
from pathlib import Path

import numpy as np

from spider import (
    CFG_TEST_DATA,
    REFERENCE_TEST_DATA,
    SpiderSolver,
    __version__,
    debug_logger,
)

# from tests.conftest import profile_decorator

logger: logging.Logger = debug_logger()
# Comment out for default debug logger, but this will slow down the tests
# logger.setLevel(logging.WARNING)

atol: float = 1e-8
rtol: float = 1e-8


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


# @profile_decorator
def test_liquid_no_heating():
    """Test Abe (1993."""

    spider_solver: SpiderSolver = SpiderSolver("abe_liquid.cfg", CFG_TEST_DATA)
    spider_solver.config["energy"]["radionuclides"] = "False"
    spider_solver.initialize()
    spider_solver.solve()
    calculated: np.ndarray = spider_solver.get_temperature()[:, -1]
    # spider_solver.plot()

    # np.savetxt("testout.dat", calculated)

    expected: np.ndarray = np.loadtxt(REFERENCE_TEST_DATA / Path("abe_liquid_no_heating.txt"))
    # print("calculated = %s" % calculated)
    # print("expected = %s" % expected)

    assert np.isclose(calculated, expected, atol=atol, rtol=rtol).all()


# @profile_decorator
def test_solid_no_heating():
    """Test Abe (1993."""

    spider_solver: SpiderSolver = SpiderSolver(Path("abe_solid.cfg"), CFG_TEST_DATA)
    spider_solver.config["energy"]["radionuclides"] = "False"
    spider_solver.initialize()
    spider_solver.solve()
    calculated: np.ndarray = spider_solver.get_temperature()[:, -1]
    # spider_solver.plot()

    # np.savetxt("testout.dat", calculated)

    expected: np.ndarray = np.loadtxt(REFERENCE_TEST_DATA / Path("abe_solid_no_heating.txt"))
    # print("calculated = %s" % calculated)
    # print("expected = %s" % expected)

    assert np.isclose(calculated, expected, atol=atol, rtol=rtol).all()


# @profile_decorator
def test_solid_with_heating():
    """Test Abe (1993."""

    spider_solver: SpiderSolver = SpiderSolver(Path("abe_solid.cfg"), CFG_TEST_DATA)
    spider_solver.initialize()
    spider_solver.solve()
    calculated: np.ndarray = spider_solver.get_temperature()[:, -1]
    # spider_solver.plot()

    # np.savetxt("testout.dat", calculated)

    expected: np.ndarray = np.loadtxt(REFERENCE_TEST_DATA / Path("abe_solid_with_heating.txt"))
    # print("calculated = %s" % calculated)
    # print("expected = %s" % expected)

    assert np.isclose(calculated, expected, atol=atol, rtol=rtol).all()


# @profile_decorator
def test_mixed():
    """Test Abe (1993."""

    spider_solver: SpiderSolver = SpiderSolver(Path("abe_mixed.cfg"), CFG_TEST_DATA)
    spider_solver.initialize()
    spider_solver.solve()
    calculated: np.ndarray = spider_solver.get_temperature()[:, -1]
    # spider_solver.plot()

    # np.savetxt("testout.dat", calculated)

    # expected: np.ndarray = np.loadtxt(REFERENCE_TEST_DATA / Path("abe_solid_with_heating.txt"))
    # print("calculated = %s" % calculated)
    # print("expected = %s" % expected)

    # assert np.isclose(calculated, expected, atol=atol, rtol=rtol).all()

    assert True


# @profile_decorator
def test_mixed_lookup():
    """Test Abe (1993."""

    spider_solver: SpiderSolver = SpiderSolver(Path("abe_mixed_lookup.cfg"), CFG_TEST_DATA)
    spider_solver.initialize()
    spider_solver.solve()
    calculated: np.ndarray = spider_solver.get_temperature()[:, -1]
    # spider_solver.plot()

    # np.savetxt("testout.dat", calculated)

    # expected: np.ndarray = np.loadtxt(REFERENCE_TEST_DATA / Path("abe_solid_with_heating.txt"))
    # print("calculated = %s" % calculated)
    # print("expected = %s" % expected)

    # assert np.isclose(calculated, expected, atol=atol, rtol=rtol).all()

    assert True
