"""Tests for Abe (1993) model.

See the LICENSE file for licensing information.

"""

import logging
from pathlib import Path

import numpy as np

from spider import TEST_CFG_PATH, SpiderSolver, __version__, debug_logger

# from tests.conftest import profile_decorator

logger: logging.Logger = debug_logger()
# Comment out for default debug logger, but this will slow down the tests
logger.setLevel(logging.WARNING)

atol: float = 1e-4
rtol: float = 1e-4


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


# @profile_decorator
def test_liquid_no_heating():
    """Test Abe (1993."""

    spider_solver: SpiderSolver = SpiderSolver(Path("abe.cfg"), TEST_CFG_PATH)
    spider_solver.config["energy"]["radionuclides"] = "False"
    spider_solver.solve()
    calculated: np.ndarray = spider_solver.solution.y[:, -1] * spider_solver.scalings.temperature
    spider_solver.plot()
    expected: np.ndarray = np.array(
        [
            959.178695,
            958.22862981,
            957.27946347,
            956.33119724,
            955.38384099,
            954.43739623,
            953.49185964,
            952.54722691,
            951.60350876,
            950.66070512,
            949.71881055,
            948.77781902,
            947.83774271,
            946.89857482,
            945.96031353,
            945.02297018,
            944.08653867,
            943.15100654,
            942.21638377,
            941.28267517,
            940.34987559,
            939.41798441,
            938.48699415,
            937.55691357,
            936.62772811,
            935.69944916,
            934.77207709,
            933.84560138,
            932.92004127,
            931.99537587,
            931.07161881,
            930.14876217,
            929.22680308,
            928.30574148,
            927.38556869,
            926.46630307,
            925.54793741,
            924.63046335,
            923.71387934,
            922.79820247,
            921.88341768,
            920.96951665,
            920.05652002,
            919.14441183,
            918.23318981,
            917.32285746,
            916.41342154,
            915.50487633,
            914.5972129,
            913.69043294,
            912.78455022,
            911.87955031,
            910.9754366,
            910.07220513,
            909.16985378,
            908.26838245,
            907.36779415,
            906.4680899,
            905.56927106,
            904.67133019,
            903.77426469,
            902.87808339,
            901.98277444,
            901.08834628,
            900.19478866,
            899.30210335,
            898.41029797,
            897.51936313,
            896.62930318,
            895.74011193,
            894.85179746,
            893.96434999,
            893.07777211,
            892.19206309,
            891.30722432,
            890.42325002,
            889.54014536,
            888.65790505,
            887.77653247,
            886.89601819,
            886.01637166,
            885.13758502,
            884.25965832,
            883.3825979,
            882.50639587,
            881.6310494,
            880.75656478,
            879.88293752,
            879.01016525,
            878.13824882,
            877.26719069,
            876.39698804,
            875.52763783,
            874.65913863,
            873.79149007,
            872.92469798,
            872.05875679,
            871.19366466,
            870.32941855,
        ]
    )
    print("calculated = %s" % calculated)
    print("expected = %s" % expected)

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
    # logger.debug("calculated = %s", calculated)
    # logger.debug("expected = %s", expected)

    # spider_solver.plot()

    assert np.isclose(calculated, expected, atol=atol, rtol=rtol).all()
