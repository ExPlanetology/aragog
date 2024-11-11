#
# Copyright 2024 Dan J. Bower
#
# This file is part of Aragog.
#
# Aragog is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# Aragog is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Aragog. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Parses the configuration file and scales and stores the parameters."""

from __future__ import annotations

import logging
import sys
from dataclasses import Field, dataclass, field, fields
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import constants
from thermochem import codata
from typed_configparser import ConfigParser

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

logger: logging.Logger = logging.getLogger(__name__)


def _get_dataclass_from_section_name() -> dict[str, Any]:
    """Maps the section names in the configuration data to the dataclasses that stores the data."""
    mapping: dict[str, Any] = {
        "scalings": _ScalingsParameters,
        "solver": _SolverParameters,
        "boundary_conditions": _BoundaryConditionsParameters,
        "mesh": _MeshParameters,
        "energy": _EnergyParameters,
        "initial_condition": _InitialConditionParameters,
        "phase_liquid": _PhaseParameters,
        "phase_solid": _PhaseParameters,
        "phase_mixed": _PhaseMixedParameters,
        # radionuclides are dealt with separately
    }

    return mapping


@dataclass
class _ScalingsParameters:
    """Stores parameters in the scalings section in the configuration data. All units are SI.

    Args:
        radius: Radius in metres. Defaults to 1.
        temperature: Temperature in Kelvin. Defaults to 1.
        density: Density in kg/m^3. Defaults to 1.
        time: Time in seconds. Defaults to 1.

    Attributes:
        radius, m
        temperature, K
        density, kg/m^3
        time, s
        area, m^2
        kinetic_energy_per_volume, J/m^3
        gravitational_acceleration, m/s^2
        heat_capacity, J/kg/K
        heat_flux, W/m^2
        latent_heat_per_mass, J/kg
        power_per_mass, W/kg
        power_per_volume, W/m^3
        pressure, Pa
        temperature_gradient, K/m
        thermal_expansivity, 1/K
        thermal_conductivity, W/m/K
        velocity, m/s
        viscosity, Pa s
        time_years, years
        stefan_boltzmann_constant (non-dimensional)
    """

    radius: float = 1
    temperature: float = 1
    density: float = 1
    time: float = 1
    area: float = field(init=False)
    gravitational_acceleration: float = field(init=False)
    temperature_gradient: float = field(init=False)
    thermal_expansivity: float = field(init=False)
    pressure: float = field(init=False)
    velocity: float = field(init=False)
    kinetic_energy_per_volume: float = field(init=False)
    heat_capacity: float = field(init=False)
    latent_heat_per_mass: float = field(init=False)
    power_per_volume: float = field(init=False)
    power_per_mass: float = field(init=False)
    heat_flux: float = field(init=False)
    thermal_conductivity: float = field(init=False)
    viscosity: float = field(init=False)
    time_years: float = field(init=False)
    stefan_boltzmann_constant: float = field(init=False)

    def __post_init__(self) -> None:
        self.area = np.square(self.radius)
        self.gravitational_acceleration = self.radius / np.square(self.time)
        self.temperature_gradient = self.temperature / self.radius
        self.thermal_expansivity = 1 / self.temperature
        self.pressure = self.density * self.gravitational_acceleration * self.radius
        self.velocity = self.radius / self.time
        self.kinetic_energy_per_volume = self.density * np.square(self.velocity)
        self.heat_capacity = self.kinetic_energy_per_volume / self.density / self.temperature
        self.latent_heat_per_mass = self.heat_capacity * self.temperature
        self.power_per_volume = self.kinetic_energy_per_volume / self.time
        self.power_per_mass = self.power_per_volume / self.density
        self.heat_flux = self.power_per_volume * self.radius
        self.thermal_conductivity = self.power_per_volume * self.area / self.temperature
        self.viscosity = self.pressure * self.time
        self.time_years = self.time / constants.Julian_year  # Equivalent to TIMEYRS C code
        # Stefan-Boltzmann units for dimensional are W/m^2/K^4
        self.stefan_boltzmann_constant: float = codata.value("Stefan-Boltzmann constant")
        self.stefan_boltzmann_constant /= (
            self.power_per_volume * self.radius / np.power(self.temperature, 4)
        )
        logger.debug("scalings = %s", self)


@dataclass
class _BoundaryConditionsParameters:
    """Stores parameters in the boundary_conditions section in the configuration data."""

    outer_boundary_condition: int
    outer_boundary_value: float
    inner_boundary_condition: int
    inner_boundary_value: float
    emissivity: float
    equilibrium_temperature: float
    core_density: float
    core_heat_capacity: float
    scalings_: _ScalingsParameters = field(init=False)

    def scale_attributes(self, scalings: _ScalingsParameters) -> None:
        """Scales the attributes.

        Args:
            scalings: scalings
        """
        self.scalings_ = scalings
        self.equilibrium_temperature /= self.scalings_.temperature
        self.core_density /= self.scalings_.density
        self.core_heat_capacity /= self.scalings_.heat_capacity
        self._scale_inner_boundary_condition()
        self._scale_outer_boundary_condition()

    def _scale_inner_boundary_condition(self) -> None:
        """Scales the inner boundary value.

        Equivalent to CORE_BC in C code.
            1: Simple core cooling
            2: Prescribed heat flux
            3: Prescribed temperature
        """
        if self.inner_boundary_condition == 1:
            self.inner_boundary_value = 0
        elif self.inner_boundary_condition == 2:
            self.inner_boundary_value /= self.scalings_.heat_flux
        elif self.inner_boundary_condition == 3:
            self.inner_boundary_value /= self.scalings_.temperature
        else:
            msg: str = f"inner_boundary_condition = {self.inner_boundary_condition} is unknown"
            raise ValueError(msg)

    def _scale_outer_boundary_condition(self) -> None:
        """Scales the outer boundary value.

        Equivalent to SURFACE_BC in C code.
            1: Grey-body atmosphere
            2: Zahnle steam atmosphere
            3: Couple to atmodeller
            4: Prescribed heat flux
            5: Prescribed temperature
        """
        if self.outer_boundary_condition == 1:
            pass
        elif self.outer_boundary_condition == 2:
            pass
        elif self.outer_boundary_condition == 3:
            pass
        elif self.outer_boundary_condition == 4:
            self.outer_boundary_value /= self.scalings_.heat_flux
        elif self.outer_boundary_condition == 5:
            self.outer_boundary_value /= self.scalings_.temperature
        else:
            msg: str = f"outer_boundary_condition = {self.outer_boundary_condition} is unknown"
            raise ValueError(msg)


@dataclass
class _EnergyParameters:
    """Stores parameters in the energy section"""

    conduction: bool
    convection: bool
    gravitational_separation: bool
    mixing: bool
    radionuclides: bool
    tidal: bool


@dataclass
class _InitialConditionParameters:
    """Stores the settings in the initial_condition section in the configuration data."""

    surface_temperature: float = 4000
    basal_temperature: float = 4000
    init_file: str = ""
    from_field: bool = False
    scalings_: _ScalingsParameters = field(init=False)

    def scale_attributes(self, scalings: _ScalingsParameters) -> None:
        """Scales the attributes.

        Args:
            scalings: scalings
        """
        self.scalings_ = scalings
        self.surface_temperature /= self.scalings_.temperature
        self.basal_temperature /= self.scalings_.temperature

        if self.init_file:
            self.from_field = True
            self.init_temperature = np.loadtxt(self.init_file)
            self.init_temperature /= self.scalings_.temperature


@dataclass
class _MeshParameters:
    """Stores parameters in the mesh section in the configuration data."""

    outer_radius: float
    inner_radius: float
    number_of_nodes: int
    mixing_length_profile: str
    # Static pressure profile is derived from the Adams-Williamson equation of state.
    surface_density: float
    gravitational_acceleration: float
    adiabatic_bulk_modulus: float
    scalings_: _ScalingsParameters = field(init=False)

    def scale_attributes(self, scalings: _ScalingsParameters) -> None:
        """Scales the attributes

        Args:
            scalings: scalings
        """
        self.scalings_ = scalings
        self.outer_radius /= self.scalings_.radius
        self.inner_radius /= self.scalings_.radius
        self.surface_density /= self.scalings_.density
        self.gravitational_acceleration /= self.scalings_.gravitational_acceleration
        self.adiabatic_bulk_modulus /= self.scalings_.pressure


@dataclass
class _PhaseMixedParameters:
    """Stores settings in the phase_mixed section in the configuration data."""

    latent_heat_of_fusion: float
    rheological_transition_melt_fraction: float
    rheological_transition_width: float
    solidus: str
    liquidus: str
    phase: str
    phase_transition_width: float
    grain_size: float
    scalings_: _ScalingsParameters = field(init=False)

    def scale_attributes(self, scalings: _ScalingsParameters) -> None:
        """Scales the attributes

        Args:
            scalings: scalings
        """
        self.scalings_ = scalings
        self.latent_heat_of_fusion /= self.scalings_.latent_heat_per_mass
        self.grain_size /= self.scalings_.radius


@dataclass
class _PhaseParameters:
    """Stores settings in a phase section in the configuration data.

    This is used to store settings from phase_liquid and phase_solid.
    """

    density: float | str
    heat_capacity: float | str
    melt_fraction: float
    thermal_conductivity: float | str
    thermal_expansivity: float | str
    viscosity: float | str
    scalings_: _ScalingsParameters = field(init=False)

    def scale_attributes(self, scalings: _ScalingsParameters) -> None:
        """Scales the attributes if they are numbers.

        Args:
            scalings: scalings
        """
        self.scalings_ = scalings
        cls_fields: tuple[Field, ...] = fields(self.__class__)
        for field_ in cls_fields:
            value: Any = getattr(self, field_.name)
            try:
                scaling: float = getattr(self.scalings_, field_.name)
                scaled_value = value / scaling
                setattr(self, field_.name, scaled_value)
                logger.info(
                    "%s is a number (value = %s, scaling = %s, scaled_value = %s)",
                    field_.name,
                    value,
                    scaling,
                    scaled_value,
                )
            except AttributeError:
                logger.info("No scaling found for %s", field_.name)
            except TypeError:
                logger.info(
                    "%s is a string (path to a filename) so the data will be scaled later",
                    field_.name,
                )


@dataclass
class _Radionuclide:
    """Stores the settings in a radionuclide section in the configuration data."""

    name: str
    t0_years: float
    abundance: float
    concentration: float
    heat_production: float
    half_life_years: float
    scalings_: _ScalingsParameters = field(init=False)

    def scale_attributes(self, scalings: _ScalingsParameters) -> None:
        """Scales the attributes.

        Args:
            scalings: scalings
        """
        self.scalings_ = scalings
        self.t0_years /= self.scalings_.time_years
        self.concentration *= 1e-6  # to mass fraction
        self.heat_production /= self.scalings_.power_per_mass
        self.half_life_years /= self.scalings_.time_years

    def get_heating(self, time: npt.NDArray | float) -> npt.NDArray | float:
        """Radiogenic heating

        Args:
            time: Time

        Returns:
            Radiogenic heating as a float if time is a float, otherwise a numpy row array where
                each entry in the row is associated with a single time in the time array.
        """
        arg: npt.NDArray | float = np.log(2) * (self.t0_years - time) / self.half_life_years
        heating: npt.NDArray | float = (
            self.heat_production * self.abundance * self.concentration * np.exp(arg)
        )

        return heating


@dataclass
class _SolverParameters:
    """Stores settings in the solver section in the configuration data."""

    start_time: float
    end_time: float
    atol: float
    rtol: float
    scalings_: _ScalingsParameters = field(init=False)

    def scale_attributes(self, scalings: _ScalingsParameters) -> None:
        self.scalings_ = scalings
        self.start_time /= self.scalings_.time_years
        self.end_time /= self.scalings_.time_years


@dataclass(kw_only=True)
class Parameters:
    """Assembles all the parameters.

    The parameters in each section are scaled here to ensure that all the parameters are scaled
    (non-dimensionalised) consistently with each other.
    """

    boundary_conditions: _BoundaryConditionsParameters
    energy: _EnergyParameters
    initial_condition: _InitialConditionParameters
    mesh: _MeshParameters
    phase_solid: _PhaseParameters
    phase_liquid: _PhaseParameters
    phase_mixed: _PhaseMixedParameters
    radionuclides: list[_Radionuclide]
    scalings: _ScalingsParameters
    solver: _SolverParameters

    def __post_init__(self):
        cls_fields: tuple[Field, ...] = fields(self.__class__)
        for field_ in cls_fields:
            data = getattr(self, field_.name)
            # Dataclass
            if hasattr(data, "scale_attributes"):
                data.scale_attributes(self.scalings)
            # List of dataclasses
            elif isinstance(data, list):
                for entry in data:
                    if hasattr(entry, "scale_attributes"):
                        entry.scale_attributes(self.scalings)

    @classmethod
    def from_file(cls, *filenames) -> Self:
        """Parses the parameters in a configuration file(s)

        Args:
            *filenames: Filenames of the configuration data
        """
        parser: ConfigParser = ConfigParser()
        parser.read(*filenames)

        init_dict: dict[str, Any] = {}
        for section_name, dataclass_ in _get_dataclass_from_section_name().items():
            init_dict[section_name] = parser.parse_section(
                using_dataclass=dataclass_, section_name=section_name
            )
        radionuclides: list[_Radionuclide] = []
        for radionuclide_section in cls.radionuclide_sections(parser):
            radionuclide = parser.parse_section(
                using_dataclass=_Radionuclide, section_name=radionuclide_section
            )
            radionuclides.append(radionuclide)

        init_dict["radionuclides"] = radionuclides

        return cls(**init_dict)  # Unpacking gives required arguments so pylint: disable=E1125

    @staticmethod
    def radionuclide_sections(parser: ConfigParser) -> list[str]:
        """Section names relating to radionuclides

        Sections relating to radionuclides must have the prefix radionuclide_
        """
        return [
            parser[section].name
            for section in parser.sections()
            if section.startswith("radionuclide_")
        ]
