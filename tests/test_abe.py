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
            3996.33374388,
            3980.37068029,
            3963.39616045,
            3945.93043011,
            3928.23812489,
            3910.44707964,
            3892.6149291,
            3874.76624817,
            3856.91108967,
            3839.05344335,
            3821.19485817,
            3803.33592599,
            3785.47687079,
            3767.61777796,
            3749.75868367,
            3731.89961303,
            3714.0406061,
            3696.18175727,
            3678.32330493,
            3660.46585052,
            3642.61089106,
            3624.76208635,
            3606.92818544,
            3589.1295229,
            3571.41169338,
            3553.87232983,
            3536.70870729,
            3520.29400124,
            3505.30871595,
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
            3996.3337782,
            3980.37071457,
            3963.39619471,
            3945.93046435,
            3928.23815913,
            3910.44711388,
            3892.61496334,
            3874.76628241,
            3856.91112391,
            3839.05347759,
            3821.19489241,
            3803.33596023,
            3785.47690503,
            3767.6178122,
            3749.75871791,
            3731.89964727,
            3714.04064034,
            3696.18179151,
            3678.32333917,
            3660.46588476,
            3642.6109253,
            3624.76212059,
            3606.92821968,
            3589.12955714,
            3571.41172762,
            3553.87236406,
            3536.7087415,
            3520.29403542,
            3505.30875009,
        ]
    )
    logger.debug("calculated = %s", calculated)
    logger.debug("expected = %s", expected)

    # spider_solver.plot()

    assert np.isclose(calculated, expected).all()
