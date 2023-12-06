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

# logger: logging.Logger = debug_logger()
# Comment out for default debug logger, but this will slow down the tests
# logger.setLevel(logging.WARNING)

atol: float = 1e-4
rtol: float = 1e-4


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


# @profile_decorator
def test_liquid_no_heating():
    """Test Abe (1993."""

    with CFG_TEST_DATA as cfg_test_data:
        spider_solver: SpiderSolver = SpiderSolver(Path("abe.cfg"), cfg_test_data)
    spider_solver.config["energy"]["radionuclides"] = "False"
    spider_solver.solve()
    calculated: np.ndarray = (
        spider_solver.solution.y[:, -1] * spider_solver.data.scalings.temperature
    )
    spider_solver.plot()
    with REFERENCE_TEST_DATA as test_data:
        expected: np.ndarray = np.loadtxt(test_data / Path("abe_liquid_no_heating.txt"))
    # print("calculated = %s" % calculated)
    # print("expected = %s" % expected)

    assert np.isclose(calculated, expected, atol=atol, rtol=rtol).all()


def test_liquid_with_heating():
    """Test Abe (1993."""

    with CFG_TEST_DATA as cfg_test_data:
        spider_solver: SpiderSolver = SpiderSolver(Path("abe.cfg"), cfg_test_data)
    spider_solver.solve()
    calculated: np.ndarray = (
        spider_solver.solution.y[:, -1] * spider_solver.data.scalings.temperature
    )
    spider_solver.plot()
    with REFERENCE_TEST_DATA as test_data:
        expected: np.ndarray = np.loadtxt(test_data / Path("abe_liquid_with_heating.txt"))
    # print("calculated = %s" % calculated)
    # print("expected = %s" % expected)

    assert np.isclose(calculated, expected, atol=atol, rtol=rtol).all()
