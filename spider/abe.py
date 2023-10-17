#!/usr/bin/env python

"""Abe model from Abe (1993)."""

import logging

from spider.mesh import SpiderMesh
from spider.phase import ConstantPhase, PhaseProtocol

logger: logging.Logger = logging.getLogger(__name__)

outer_radius: float = 6371e3
inner_radius: float = outer_radius - 1000e3
number_of_coordinates: int = 100

gravity: float = -9.81  # NOTE: Negative.

spider_mesh: SpiderMesh = SpiderMesh.uniform_radii(
    inner_radius, outer_radius, number_of_coordinates
)

liquid: PhaseProtocol = ConstantPhase(
    _density=4000,
    _gravity=gravity,
    _log10_viscosity=2,
    _heat_capacity=1000,
    _thermal_conductivity=4,
    _thermal_expansivity=1.0e-5,
)
solid: PhaseProtocol = ConstantPhase(
    _density=4200,
    _gravity=gravity,
    _log10_viscosity=21,
    _heat_capacity=1000,
    _thermal_conductivity=4,
    _thermal_expansivity=1.0e-5,
)
