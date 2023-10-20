"""Energy

See the LICENSE file for licensing information.
"""

from __future__ import annotations

import logging
from configparser import ConfigParser, SectionProxy

import numpy as np

from spider import YEAR_IN_SECONDS
from spider.mesh import StaggeredMesh
from spider.phase import Phase

logger: logging.Logger = logging.getLogger(__name__)


def conductive_heat_flux(
    mesh: StaggeredMesh,
    phase: Phase,
    temperature: np.ndarray,
    pressure: np.ndarray,
) -> np.ndarray:
    """Conductive heat flux

    Args:
        mesh: Mesh
        phase: Phase
        temperature: Temperature at the staggered nodes
        pressure: Pressure at the staggered nodes

    Returns:
        Conductive heat flux
    """
    temperature_basic: np.ndarray = mesh.quantity_at_basic_nodes(temperature)
    pressure_basic: np.ndarray = mesh.quantity_at_basic_nodes(pressure)
    heat_flux: np.ndarray = -phase.thermal_conductivity(
        temperature_basic, pressure_basic
    ) * mesh.d_dr_at_basic_nodes(temperature)
    logger.info("conductive_heat_flux = %s", heat_flux)

    return heat_flux


def convective_heat_flux(
    mesh: StaggeredMesh,
    phase: Phase,
    eddy_diffusivity: np.ndarray,
    temperature: np.ndarray,
    pressure: np.ndarray,
) -> np.ndarray:
    """Convective heat flux

    Args:
        mesh: Mesh
        phase: Phase
        eddy_diffusivity: Eddy diffusivity at the basic nodes
        temperature: Temperature at the staggered nodes
        pressure: Pressure at the staggered nodes

    Returns:
        Convective heat flux
    """
    temperature_basic: np.ndarray = mesh.quantity_at_basic_nodes(temperature)
    pressure_basic: np.ndarray = mesh.quantity_at_basic_nodes(pressure)
    heat_flux: np.ndarray = (
        -phase.density(temperature_basic, pressure_basic)
        * phase.heat_capacity(temperature_basic, pressure_basic)
        * eddy_diffusivity
    )

    # heat_flux *= 1.0e6  # FIXME: Multiply by kappa_h. Constant value just for testing.

    heat_flux *= mesh.d_dr_at_basic_nodes(temperature) - phase.dTdrs(
        temperature_basic, pressure_basic
    )
    logger.info("convective_heat_flux = %s", heat_flux)

    return heat_flux


def total_heat_flux(
    energy: SectionProxy,
    mesh: StaggeredMesh,
    phase: Phase,
    eddy_diffusivity: np.ndarray,
    temperature: np.ndarray,
    pressure: np.ndarray,
) -> np.ndarray:
    """Total heat flux

    Args:
        energy: Energy section from the configuration
        mesh: Mesh
        phase: Phase
        eddy_diffusivity: Eddy diffusivity at the basic nodes
        temperature: Temperature at the staggered nodes
        pressure: Pressure at the staggered nodes

    Returns:
        Total heat flux
    """
    total_heat_flux: np.ndarray = np.zeros(mesh.basic.number)

    if energy.getboolean("convection"):
        total_heat_flux += convective_heat_flux(
            mesh, phase, eddy_diffusivity, temperature, pressure
        )

    if energy.getboolean("conduction"):
        total_heat_flux += conductive_heat_flux(mesh, phase, temperature, pressure)

    return total_heat_flux


def radiogenic_heating(
    radionuclide: SectionProxy,
    time: float,
) -> float:
    """Radiogenic heating

    Args:
        radionuclide: A radionuclide section from the configuration
        time: Time in seconds

    Returns:
        Radiogenic heating
    """
    # Time must be in years because t0 and half_life are in years.
    time /= YEAR_IN_SECONDS
    arg: float = (
        np.log(2) * (radionuclide.getfloat("t0") - time) / radionuclide.getfloat("half_life")
    )
    heating: float = (
        radionuclide.getfloat("heat_production")
        * radionuclide.getfloat("abundance")
        * radionuclide.getfloat("concentration")
        * np.exp(arg)
    )
    radionuclide_name: str = radionuclide.name.split("_")[-1]
    logger.debug("Heating rate due to %s = %f", radionuclide_name, heating)

    return heating


def tidal_heating(time: float) -> float:
    """Tidal heating

    Args:
        time: Time in seconds

    Returns:
        Tidal heating

    Raises:
        NotImplementedError
    """
    del time

    raise NotImplementedError


def total_radiogenic_heating(config: ConfigParser, time: float) -> float:
    """Total radiogenic heating

    Args:
        config: Configuration with zero or several sections beginning with 'radionuclide_'
        time: Time in seconds

    Returns:
        Total radiogenic heating
    """
    radionuclides: list[SectionProxy] = [
        config[section] for section in config.sections() if section.startswith("radionuclide_")
    ]
    total_radiogenic_heating: float = 0
    for radionuclide in radionuclides:
        total_radiogenic_heating += radiogenic_heating(radionuclide, time)

    logger.debug("Total radiogenic heating rate = %f", total_radiogenic_heating)

    return total_radiogenic_heating


def total_heating(config: ConfigParser, time: float) -> float:
    """Total heating

    Args:
        config: Configuration
        time: Time in seconds

    Returns:
        Total heating
    """
    energy: SectionProxy = config["energy"]

    total_heating: float = 0

    if energy.getboolean("radionuclides"):
        total_heating += total_radiogenic_heating(config, time)

    if energy.getboolean("tidal"):
        total_heating += tidal_heating(time)

    return total_heating
