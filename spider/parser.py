#
# Copyright 2024 Dan J. Bower
#
# This file is part of Spider.
#
# Spider is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# Spider is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Spider. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Dataclasses to scale and store the data in a Spider configuration file"""

import logging
from dataclasses import Field, dataclass, fields

import numpy as np
from scipy import constants
from thermochem import codata
from typed_configparser import ConfigParser

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class scalings:
    """Scalings for the numerical problem.

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
    area: float = 0
    gravitational_acceleration: float = 0
    temperature_gradient: float = 0
    thermal_expansivity: float = 0
    pressure: float = 0
    velocity: float = 0
    kinetic_energy_per_volume: float = 0
    heat_capacity: float = 0
    latent_heat_per_mass: float = 0
    power_per_volume: float = 0
    power_per_mass: float = 0
    heat_flux: float = 0
    thermal_conductivity: float = 0
    viscosity: float = 0
    time_years: float = 0
    stefan_boltzmann_constant: float = 0

    def __post_init__(self) -> None:
        """Note that this is not called by typed-configparser and must be called manually"""
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
        # Non-dimensional constants
        self.stefan_boltzmann_constant: float = codata.value(
            "Stefan-Boltzmann constant"
        )  # W/m^2/K^4
        self.stefan_boltzmann_constant /= (
            self.power_per_volume * self.radius / np.power(self.temperature, 4)
        )
        logger.debug("scalings = %s", self)


@dataclass
class boundary_conditions:
    """Boundary conditions"""

    outer_boundary_condition: int
    outer_boundary_value: float
    inner_boundary_condition: int
    inner_boundary_value: float
    emissivity: float
    equilibrium_temperature: float
    core_radius: float
    core_density: float
    core_heat_capacity: float

    def scale_attributes(self, scalings_: scalings) -> None:
        self._scalings: scalings = scalings_
        self.equilibrium_temperature /= self._scalings.temperature
        self.core_radius /= self._scalings.radius
        self.core_density /= self._scalings.density
        self.core_heat_capacity /= self._scalings.heat_capacity
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
            self.inner_boundary_value /= self._scalings.heat_flux
        elif self.inner_boundary_condition == 3:
            self.inner_boundary_value /= self._scalings.temperature
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
            self.outer_boundary_value /= self._scalings.heat_flux
        elif self.outer_boundary_condition == 5:
            self.outer_boundary_value /= self._scalings.temperature
        else:
            msg: str = f"outer_boundary_condition = {self.outer_boundary_condition} is unknown"
            raise ValueError(msg)


@dataclass
class energy:
    """Energy"""

    conduction: bool
    convection: bool
    gravitational_separation: bool
    mixing: bool
    radionuclides: bool
    tidal: bool


@dataclass
class initial_condition:
    """Initial condition"""

    surface_temperature: float
    basal_temperature: float

    def scale_attributes(self, scalings_: scalings) -> None:
        self._scalings: scalings = scalings_
        self.surface_temperature /= self._scalings.temperature
        self.basal_temperature /= self._scalings.temperature


@dataclass
class mesh:
    """Mesh"""

    outer_radius: float
    inner_radius: float
    number_of_nodes: int
    mixing_length_profile: str
    # Static pressure profile is derived from the Adams-Williamson equation of state.
    adams_williamson_surface_density: float
    adams_williamson_beta: float
    gravitational_acceleration: float

    def scale_attributes(self, scalings_: scalings) -> None:
        self._scalings: scalings = scalings_
        self.outer_radius /= self._scalings.radius
        self.inner_radius /= self._scalings.radius
        self.adams_williamson_surface_density /= self._scalings.density
        self.adams_williamson_beta *= self._scalings.radius
        self.gravitational_acceleration /= self._scalings.gravitational_acceleration


@dataclass
class phase_mixed:
    """Mixed phase"""

    latent_heat_of_fusion: float
    rheological_transition_melt_fraction: float
    rheological_transition_width: float
    solidus: str
    liquidus: str

    def scale_attributes(self, scalings_: scalings) -> None:
        self._scalings: scalings = scalings_
        self.latent_heat_of_fusion /= self._scalings.latent_heat_per_mass


@dataclass
class phase:
    """Phase"""

    density: float | str
    viscosity: float | str
    heat_capacity: float | str
    thermal_conductivity: float | str
    thermal_expansivity: float | str

    def scale_attributes(self, scalings_: scalings) -> None:
        """Scales the attributes if they are numbers"""
        self._scalings: scalings = scalings_
        cls_fields: tuple[Field, ...] = fields(self.__class__)
        for field in cls_fields:
            try:
                new_val = getattr(self, field.name) / getattr(self._scalings, field.name)
                logger.info("%s is a number", field.name)
                setattr(self, field.name, new_val)
            except TypeError:
                logger.info("%s is a string (path to a filename)", field.name)


@dataclass
class radionuclide:
    """Radionuclide"""

    name: str
    t0_years: float
    abundance: float
    concentration: float
    heat_production: float
    half_life_years: float

    def scale_attributes(self, scalings_: scalings) -> None:
        self._scalings: scalings = scalings_
        self.t0_years /= self._scalings.time_years
        self.concentration *= 1e-6  # to mass fraction
        self.heat_production /= self._scalings.power_per_mass
        self.half_life_years /= self._scalings.time_years


@dataclass
class solver:
    """Solver"""

    start_time: float
    end_time: float
    atol: float
    rtol: float

    def scale_attributes(self, scalings_: scalings) -> None:
        self._scalings: scalings = scalings_
        self.start_time /= self._scalings.time_years
        self.end_time /= self._scalings.time_years


@dataclass
class Configuration:
    """Configuration"""

    boundary_conditions: boundary_conditions
    energy: energy
    initial_condition: initial_condition
    mesh: mesh
    phase_solid: phase
    phase_liquid: phase
    phase_mixed: phase_mixed
    radionuclides: list[radionuclide]
    scalings: scalings
    solver: solver

    def __post_init__(self):
        # Note typed-configparser redefines __init__ and thus __post_init__ is not called
        self.scalings.__post_init__()
        cls_fields: tuple[Field, ...] = fields(self.__class__)
        for field in cls_fields:
            data = getattr(self, field.name)
            # Dataclass
            if hasattr(data, "scale_attributes"):
                data.scale_attributes(self.scalings)
            # List of dataclasses
            elif isinstance(data, list):
                for entry in data:
                    if hasattr(entry, "scale_attributes"):
                        entry.scale_attributes(self.scalings)


class Parameters:
    """Parser for Spider configuration files"""

    def __init__(self, *filenames):
        self.parser: ConfigParser = ConfigParser()
        self.parser.read(*filenames)

        scalings_ = self.parser.parse_section(using_dataclass=scalings)
        boundary_conditions_ = self.parser.parse_section(using_dataclass=boundary_conditions)
        energy_ = self.parser.parse_section(using_dataclass=energy)
        initial_condition_ = self.parser.parse_section(using_dataclass=initial_condition)
        mesh_ = self.parser.parse_section(using_dataclass=mesh)
        phase_solid_ = self.parser.parse_section(using_dataclass=phase, section_name="phase_solid")
        phase_liquid_ = self.parser.parse_section(
            using_dataclass=phase, section_name="phase_liquid"
        )
        phase_mixed_ = self.parser.parse_section(using_dataclass=phase_mixed)
        solver_ = self.parser.parse_section(using_dataclass=solver)

        radionuclides_: list[radionuclide] = []
        for radionuclide_section in self.radionuclides:
            radionuclide_ = self.parser.parse_section(
                using_dataclass=radionuclide, section_name=radionuclide_section
            )
            radionuclides_.append(radionuclide_)

        self.data: Configuration = Configuration(
            boundary_conditions_,
            energy_,
            initial_condition_,
            mesh_,
            phase_solid_,
            phase_liquid_,
            phase_mixed_,
            radionuclides_,
            scalings_,
            solver_,
        )

    @property
    def radionuclides(self) -> list[str]:
        """Sections relating to radionuclides"""
        return [
            self.parser[section].name
            for section in self.parser.sections()
            if section.startswith("radionuclide_")
        ]


# For testing
test = Parameters("/Users/dan/Documents/academic/explanetology/pyspider/tests/cfg/abe_mixed.cfg")
# test.data.scalings.__post_init__()
print(test.data)
# test.data.scalings.__post_init__()
# print(test.data.scalings)
