"""Energy."""

from __future__ import annotations

import logging
from configparser import SectionProxy

import numpy as np

from spider.mesh import SpiderMesh
from spider.phase import PhaseABC

logger: logging.Logger = logging.getLogger(__name__)


def conductive_heat_flux(
    mesh: SpiderMesh,
    phase: PhaseABC,
    temperature: np.ndarray,
    pressure: np.ndarray,
) -> np.ndarray:
    """Conductive heat flux.

    Args:
        mesh: Mesh.
        phase: Phase.
        temperature: Temperature at the staggered nodes.
        pressure: Pressure at the staggered nodes.

    Returns:
        Conductive heat flux.
    """
    temperature_basic: np.ndarray = mesh.quantity_at_basic_nodes(temperature)
    pressure_basic: np.ndarray = mesh.quantity_at_basic_nodes(pressure)
    heat_flux: np.ndarray = -phase.thermal_conductivity(
        temperature_basic, pressure_basic
    ) * mesh.d_dr_at_basic_nodes(temperature)
    logger.info("conductive_heat_flux = %s", heat_flux)

    return heat_flux


def convective_heat_flux(
    mesh: SpiderMesh,
    phase: PhaseABC,
    temperature: np.ndarray,
    pressure: np.ndarray,
) -> np.ndarray:
    """Convective heat flux.

    Args:
        mesh: Mesh.
        phase: Phase.
        temperature: Temperature at the staggered nodes.
        pressure: Pressure at the staggered nodes.

    Returns:
        Convective heat flux.
    """
    temperature_basic: np.ndarray = mesh.quantity_at_basic_nodes(temperature)
    pressure_basic: np.ndarray = mesh.quantity_at_basic_nodes(pressure)
    heat_flux: np.ndarray = -phase.density(
        temperature_basic, pressure_basic
    ) * phase.heat_capacity(temperature_basic, pressure_basic)

    heat_flux *= 1.0e6  # e6  # FIXME: Multiply by kappa_h. Constant value just for testing.

    heat_flux *= mesh.d_dr_at_basic_nodes(temperature) - phase.dTdrs(
        temperature_basic, pressure_basic
    )
    logger.info("convective_heat_flux = %s", heat_flux)

    return heat_flux


def total_heat_flux(
    energy: SectionProxy,
    mesh: SpiderMesh,
    phase: PhaseABC,
    temperature: np.ndarray,
    pressure: np.ndarray,
) -> np.ndarray:
    """Total heat flux.

    Args:
        energy: Energy section from the configuration.
        mesh: Mesh.
        phase: Phase.
        temperature: Temperature at the staggered nodes.
        pressure: Pressure at the staggered nodes.

    Returns:
        Total heat flux.
    """
    total_heat_flux: np.ndarray = np.zeros(mesh.basic.number)

    if energy.getboolean("convection"):
        total_heat_flux += convective_heat_flux(mesh, phase, temperature, pressure)

    if energy.getboolean("conduction"):
        total_heat_flux += conductive_heat_flux(mesh, phase, temperature, pressure)

    return total_heat_flux
