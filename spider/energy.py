"""Energy

See the LICENSE file for licensing information.
"""

from __future__ import annotations

import logging
from configparser import ConfigParser, SectionProxy
from dataclasses import KW_ONLY, dataclass, field

import numpy as np

from spider.interfaces import DataclassFromConfiguration, Scalings

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class Radionuclide(DataclassFromConfiguration):
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

    def __post_init__(self):
        # Non-dimensionalise
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


@dataclass
class RadiogenicHeating:
    """Radiogenic heating

    Args:
        TODO
    """

    scalings: Scalings
    _: KW_ONLY
    config: ConfigParser
    _radionuclides: list[Radionuclide] = field(init=False, default_factory=list)

    def __post_init__(self):
        radionuclide_sections: list[SectionProxy] = [
            self.config[section]
            for section in self.config.sections()
            if section.startswith("radionuclide_")
        ]
        for radionuclide_section in radionuclide_sections:
            radionuclide: Radionuclide = Radionuclide.from_configuration(
                self.scalings,
                radionuclide_section.name.split("_")[-1],
                config=radionuclide_section,
            )
            self._radionuclides.append(radionuclide)

    def __call__(self, time: float) -> float:
        heating: float = 0
        for radionuclide in self._radionuclides:
            heating += radionuclide.radiogenic_heating(time)

        return heating
