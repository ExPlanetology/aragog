"""Tests for Abe (1993) model.

See the LICENSE file for licensing information.

"""

import logging
from pathlib import Path

import numpy as np

from spider import TEST_CFG_PATH, SpiderSolver, __version__, debug_logger

logger: logging.Logger = debug_logger()

atol: float = 1e-4
rtol: float = 1e-4


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


def test_liquid_no_heating():
    """Test Abe (1993."""

    spider_solver: SpiderSolver = SpiderSolver(TEST_CFG_PATH / Path("abe.cfg"))
    spider_solver.config["energy"]["radionuclides"] = "False"
    spider_solver.solve()
    calculated: np.ndarray = spider_solver.solution.y[:, -1] * spider_solver.scalings.temperature
    spider_solver.plot()
    expected: np.ndarray = np.array(
        [
            958.03547486,
            954.79956243,
            951.57424092,
            948.35953268,
            945.15543511,
            941.96193518,
            938.77901429,
            935.60665038,
            932.44481911,
            929.29349448,
            926.15264934,
            923.0222556,
            919.90228449,
            916.79270666,
            913.69349239,
            910.60461154,
            907.52603372,
            904.45772827,
            901.3996644,
            898.35181111,
            895.31413733,
            892.28661183,
            889.26920337,
            886.26188059,
            883.26461214,
            880.27736661,
            877.30011257,
            874.3328186,
            871.37545327,
        ]
    )
    logger.debug("calculated = %s", calculated)
    logger.debug("expected = %s", expected)

    assert np.isclose(calculated, expected, atol=atol, rtol=rtol).all()


def test_liquid_with_heating():
    """Test Abe (1993."""

    spider_solver: SpiderSolver = SpiderSolver(TEST_CFG_PATH / Path("abe.cfg"))
    spider_solver.solve()
    calculated: np.ndarray = spider_solver.solution.y[:, -1] * spider_solver.scalings.temperature
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

    assert np.isclose(calculated, expected, atol=atol, rtol=rtol).all()
