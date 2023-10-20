"""Package level variables and initialises the package logger

See the LICENSE file for licensing information.
"""

__version__: str = "0.1.0"

import importlib.resources
import logging
from pathlib import Path

from scipy import constants
from thermochem import codata

# Module constants.
GAS_CONSTANT: float = codata.value("molar gas constant")  # J/K/mol.
GRAVITATIONAL_CONSTANT: float = codata.value("Newtonian constant of gravitation")  # m^3/kg/s^2.
STEFAN_BOLTZMANN_CONSTANT: float = codata.value("Stefan-Boltzmann constant")  # W/m2/K^4.
OCEAN_MOLES: float = 7.68894973907177e22  # Moles of H2 (or H2O) in one present-day Earth ocean.
YEAR_IN_SECONDS: float = constants.Julian_year

DATA_ROOT_PATH = importlib.resources.files("%s.data" % __package__)

# Create the package logger.
# https://docs.python.org/3/howto/logging.html#library-config
logger: logging.Logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def complex_formatter() -> logging.Formatter:
    """Complex formatter."""
    fmt: str = "[%(asctime)s - %(name)-30s - %(lineno)03d - %(levelname)-9s - %(funcName)s()] - %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"
    formatter: logging.Formatter = logging.Formatter(fmt, datefmt=datefmt)
    return formatter


def simple_formatter() -> logging.Formatter:
    """Simple formatter."""
    fmt: str = "[%(asctime)s - %(name)-30s - %(levelname)-9s] - %(message)s"
    datefmt: str = "%H:%M:%S"
    formatter: logging.Formatter = logging.Formatter(fmt, datefmt=datefmt)
    return formatter


def debug_logger() -> logging.Logger:
    """Setup the logging for debugging: DEBUG to the console."""
    # Console logger.
    logger: logging.Logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    console_handler: logging.Handler = logging.StreamHandler()
    console_formatter: logging.Formatter = simple_formatter()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


def debug_file_logger() -> logging.Logger:
    """Setup the logging to a file (DEBUG) and to the console (INFO)."""
    # Console logger.
    logger: logging.Logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    console_handler: logging.Handler = logging.StreamHandler()
    console_formatter: logging.Formatter = simple_formatter()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    # File logger.
    file_handler: logging.Handler = logging.FileHandler("%s.log" % __package__)
    file_formatter: logging.Formatter = complex_formatter()
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    return logger


TEST_DATA = importlib.resources.files("tests")
TEST_CFG_PATH: Path = TEST_DATA / Path("cfg")  # type: ignore

from spider.solver import SpiderSolver
