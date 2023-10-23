"""Scalings for the numerical problem.

See the LICENSE file for licensing information.
"""

from __future__ import annotations

import logging
from configparser import SectionProxy
from dataclasses import dataclass

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(kw_only=True, frozen=True)
class NumericalScalings:
    """Scalings for the numerical problem.

    Args:
        radius: Radius in metres. Defaults to 1.
        temperature: Temperature in Kelvin. Defaults to 1.
        time: Time in seconds. Defaults to 1.
    """

    radius: float = 1
    temperature: float = 1
    time: float = 1

    @property
    def temperature_gradient(self) -> float:
        return self.temperature / self.radius


def numerical_scalings_from_configuration(scalings_section: SectionProxy) -> NumericalScalings:
    """Instantiates the scalings for the numerical problem.

    Args:
        scalings_section: Configuration section with scalings.

    Returns:
        The numerical scalings.
    """
    radius: float = scalings_section.getfloat("radius")
    temperature: float = scalings_section.getfloat("temperature")
    time: float = scalings_section.getfloat("time")

    numerical_scalings: NumericalScalings = NumericalScalings(
        radius=radius, temperature=temperature, time=time
    )

    return numerical_scalings
