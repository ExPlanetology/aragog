"""Energy

See the LICENSE file for licensing information.
"""

from __future__ import annotations

import logging
from configparser import ConfigParser, SectionProxy

import numpy as np

logger: logging.Logger = logging.getLogger(__name__)


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
    # time /= YEAR_IN_SECONDS
    # FIXME: Time will now come in non-dimensional
    arg: float = (
        np.log(2)
        * (radionuclide.getfloat("t0_years") - time)
        / radionuclide.getfloat("half_life_years")
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
