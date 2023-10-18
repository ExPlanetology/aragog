#!/usr/bin/env python

"""Abe model from Abe (1993)."""

import logging

import numpy as np

from spider import YEAR_IN_SECONDS, debug_file_logger, debug_logger
from spider.mesh import SpiderMesh
from spider.phase import ConstantPhase, PhaseProtocol
from spider.solver import SpiderSolver

debug_logger()
# debug_file_logger()

logger: logging.Logger = logging.getLogger(__name__)

outer_radius: float = 6371e3
inner_radius: float = outer_radius - 1000e3
number_of_coordinates: int = 10

gravity: float = 9.81

spider_mesh: SpiderMesh = SpiderMesh.uniform_radii(
    inner_radius, outer_radius, number_of_coordinates
)
print(spider_mesh)

# Must be float to avoid casting problems with conductive_heat_flux.
liquid: PhaseProtocol = ConstantPhase(
    _density=4000.0,
    _gravity=gravity,
    _log10_viscosity=2,
    _heat_capacity=1000.0,
    _thermal_conductivity=4.0,
    _thermal_expansivity=1.0e-5,
)
solid: PhaseProtocol = ConstantPhase(
    _density=4200.0,
    _gravity=gravity,
    _log10_viscosity=21,
    _heat_capacity=1000.0,
    _thermal_conductivity=4.0,
    _thermal_expansivity=1.0e-5,
)

initial_temperature: np.ndarray = np.array([4000, 3900, 3800, 3700, 3600, 3500, 3400, 3300, 3200])

# spider_mesh.d_dr_at_basic_nodes(initial_temperature)

end_time: float = YEAR_IN_SECONDS * 1e3  # 1 Myr.

solver = SpiderSolver(spider_mesh, liquid, initial_temperature, 0, end_time)

# solver.dTdt(0, initial_temperature, spider_mesh, liquid, initial_temperature)

solver.solve()
