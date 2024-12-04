import logging
from aragog import Solver, __version__, debug_logger
from aragog import CFG_DATA
from aragog.output import Output
from aragog.interfaces import MixedPhaseEvaluatorProtocol, PhaseEvaluatorProtocol
from aragog.utilities import FloatOrArray

import importlib.resources
from contextlib import AbstractContextManager
from importlib.abc import Traversable
from pathlib import Path
import numpy as np

from aragog import CFG_DATA


# Set up logger to write to a file
logging.basicConfig(
    filename="output.log",
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_logger")
logger.info("This is a test log entry.")

filename=CFG_DATA.joinpath("abe_liquid.cfg")
solver: Solver = Solver.from_file(filename)
solver.initialize()
solver.solve()

output: Output = Output(solver)
output.plot()
