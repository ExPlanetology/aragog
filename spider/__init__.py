"""Package level variables and initialises the package logger

See the LICENSE file for licensing information.
"""

__version__: str = "0.1.0"

import importlib.resources
import logging
from importlib.abc import Traversable
from pathlib import Path

DATA_ROOT_PATH: Traversable = importlib.resources.files("%s.data" % __package__)

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
