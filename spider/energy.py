"""Energy

See the LICENSE file for licensing information.
"""

from __future__ import annotations

import logging
from dataclasses import KW_ONLY, dataclass

import numpy as np

from spider.interfaces import ScaledDataclassFromConfiguration, Scalings

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class Radionuclide(ScaledDataclassFromConfiguration):
    """Radionuclide

    Args:
        scalings: Scalings
        name: Name of the radionuclide
        t0_years: TODO
        abundance: TODO
        concentration: TODO
        heat_production: TODO
        half_life_years: TODO

    Attributes:
        # TODO
    """

    scalings: Scalings
    name: str
    _: KW_ONLY
    t0_years: float
    abundance: float
    concentration: float
    heat_production: float
    half_life_years: float

    def scale_attributes(self):
        self.t0_years /= self.scalings.time_years
        self.concentration *= 1e-6  # to mass fraction
        self.heat_production /= self.scalings.power_per_mass
        self.half_life_years /= self.scalings.time_years

    def radiogenic_heating(self, time: float) -> float:
        """Radiogenic heating

        Args:
            time: Time in non-dimensional units

        Returns:
            Radiogenic heating in non-dimensional units
        """
        arg: float = np.log(2) * (self.t0_years - time) / self.half_life_years
        heating: float = self.heat_production * self.abundance * self.concentration * np.exp(arg)
        logger.debug("Radiogenic heating due to %s = %f", self.name, heating)

        return heating
