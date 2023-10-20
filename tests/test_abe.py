"""Tests for Abe (1993) model.

See the LICENSE file for licensing information.

"""

import logging
from pathlib import Path

import numpy as np

from spider import TEST_CFG_PATH, SpiderSolver, __version__, debug_logger

logger: logging.Logger = debug_logger()


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


def test_liquid_no_heating():
    """Test Abe (1993."""

    spider_solver: SpiderSolver = SpiderSolver(TEST_CFG_PATH / Path("abe.cfg"))
    spider_solver.config["energy"]["radionuclides"] = "False"
    spider_solver.solve()
    calculated: np.ndarray = spider_solver.solution.y[:, -1]
    # spider_solver.plot(5)
    expected: np.ndarray = np.array(
        [
            2690.62065359,
            2661.44427282,
            2632.57698289,
            2604.01578396,
            2575.75768453,
            2547.79970352,
            2520.13887215,
            2492.77223554,
            2465.69685414,
        ]
    )
    logger.debug("calculated = %s", calculated)
    logger.debug("expected = %s", expected)

    assert np.isclose(calculated, expected).all()


def test_liquid_with_heating():
    """Test Abe (1993."""

    spider_solver: SpiderSolver = SpiderSolver(TEST_CFG_PATH / Path("abe.cfg"))
    spider_solver.solve()
    calculated: np.ndarray = spider_solver.solution.y[:, -1]
    # spider_solver.plot(5)
    expected: np.ndarray = np.array(
        [
            2711.21861752,
            2681.81871523,
            2652.73011273,
            2623.94979216,
            2595.47474364,
            2567.30196746,
            2539.42847593,
            2511.85129512,
            2484.56746625,
        ]
    )
    logger.debug("calculated = %s", calculated)
    logger.debug("expected = %s", expected)

    assert np.isclose(calculated, expected).all()
