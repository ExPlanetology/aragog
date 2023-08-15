"""Package level variables and initialises the package logger."""

__version__: str = "0.1.0"

import importlib.resources
import logging

from thermochem import codata

# Create the package logger.
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a handler for the logger.
handler: logging.Handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

# Create a formatter for the log messages.
# Simple formatter.
fmt: str = "%(asctime)s - %(name)-30s - %(levelname)-9s - %(message)s"
datefmt: str = "%H:%M:%S"

# Complex formatter.
# fmt: str = "[%(asctime)s - %(name)-20s - %(lineno)03d - %(levelname)-9s -
# %(funcName)s()] %(message)s"
# datefmt: str = "Y-%m-%d %H:%M:%S"
formatter: logging.Formatter = logging.Formatter(fmt, datefmt=datefmt)
handler.setFormatter(formatter)

# Add the handler to the logger.
logger.addHandler(handler)

logger.info("%s version %s", __name__, __version__)

# Module constants.
GAS_CONSTANT: float = codata.value("molar gas constant")  # J/K/mol.
GRAVITATIONAL_CONSTANT: float = codata.value("Newtonian constant of gravitation")  # m^3/kg/s^2.
OCEAN_MOLES: float = 7.68894973907177e22  # Moles of H2 (or H2O) in one present-day Earth ocean.

DATA_ROOT_PATH = importlib.resources.files("spider.data")
