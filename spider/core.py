"""Core

See the LICENSE file for licensing information.
"""

from __future__ import annotations

import logging
from ast import literal_eval
from configparser import ConfigParser, SectionProxy
from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Union

import numpy as np
from scipy import constants
from thermochem import codata

from spider.interfaces import ScaledDataclassFromConfiguration
from spider.mesh import StaggeredMesh
from spider.phase import PhaseEvaluator

if TYPE_CHECKING:
    from spider.solver import State

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class BoundaryConditions(ScaledDataclassFromConfiguration):
    """Boundary conditions

    Args:
        scalings: Scalings
        outer_boundary_condition: TODO
        outer_boundary_value: TODO
        inner_boundary_condition: TODO
        inner_boundary_value: TODO
        emissivity: TODO
        equilibrium_temperature: TODO
        core_radius: TODO
        core_density: TODO
        core_heat_capacity: TODO

    Attributes:
        # TODO
    """

    scalings: Scalings
    _: KW_ONLY
    outer_boundary_condition: str
    outer_boundary_value: Union[str, float]
    inner_boundary_condition: str
    inner_boundary_value: Union[str, float]
    emissivity: float
    equilibrium_temperature: float
    core_radius: float
    core_density: float
    core_heat_capacity: float

    def scale_attributes(self):
        self.equilibrium_temperature /= self.scalings.temperature

    def grey_body(self, state: State) -> None:
        """Applies a grey body flux at the surface.

        Args:
            state: The state to apply the boundary conditions to
        """
        state.heat_flux[-1, :] = (
            self.emissivity
            * self.scalings.stefan_boltzmann_constant
            * (np.power(state.top_temperature, 4) - self.equilibrium_temperature**4)
        )

    def apply(self, state: State) -> None:
        """Applies the boundary conditions to the state.

        Args:
            state: The state to apply the boundary conditions to
        """
        self.core_heat_flux(state)
        self.grey_body(state)
        logger.debug("temperature = %s", state.temperature_basic)
        logger.debug("heat_flux = %s", state.heat_flux)

    def core_heat_flux(self, state: State) -> None:
        """Applies the heat flux at the core-mantle boundary.

        Args:
            state: The state to apply the boundary conditions to
        """
        # No heat flux from the core.
        state.heat_flux[0, :] = 0


@dataclass
class InitialCondition(ScaledDataclassFromConfiguration):
    """Initial condition

    Args:
        scalings: Scalings
        mesh: Mesh
        surface_temperature: TODO
        basal_temperature: TODO

    Attributes:
        scalings: Scalings
        mesh.Mesh
        surface_temperature: TODO
        basal_temperature: TODO
    """

    scalings: Scalings
    mesh: StaggeredMesh
    _: KW_ONLY
    surface_temperature: float
    basal_temperature: float
    _temperature: np.ndarray = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self._temperature = self.get_linear()

    def scale_attributes(self) -> None:
        self.surface_temperature /= self.scalings.temperature
        self.basal_temperature /= self.scalings.temperature

    @property
    def temperature(self) -> np.ndarray:
        return self._temperature

    def get_linear(self) -> np.ndarray:
        temperature: np.ndarray = np.linspace(
            self.basal_temperature, self.surface_temperature, self.mesh.staggered.number
        )
        return temperature


@dataclass
class _FixedMesh:
    """A fixed mesh

    Args:
        radii: Radii of the mesh, which could be in non-dimensional units.

    Attributes:
        radii: Radii of the mesh
        inner_radius: Inner radius
        outer_radius: Outer radius
        delta_radii: Delta radii
        depth: Depth below the outer radius
        height: Height above the inner radius
        number: Number of radii
        area: Surface area
        volume: Volume of the spherical shells defined between neighbouring radii.
        mixing_length: Mixing length # TODO: Constant for time being.
        mixing_length_squared: Mixing length squared
        mixing_length_cubed: Mixing length cubed
    """

    radii: np.ndarray
    inner_radius: float = field(init=False)
    outer_radius: float = field(init=False)
    delta_radii: np.ndarray = field(init=False)
    depth: np.ndarray = field(init=False)
    height: np.ndarray = field(init=False)
    number: int = field(init=False)
    area: np.ndarray = field(init=False)
    volume: np.ndarray = field(init=False)
    total_volume: float = field(init=False)
    mixing_length: np.ndarray = field(init=False)
    mixing_length_squared: np.ndarray = field(init=False)
    mixing_length_cubed: np.ndarray = field(init=False)

    def __post_init__(self):
        if not is_monotonic_increasing(self.radii):
            msg: str = "Mesh must be monotonically increasing"
            logger.error(msg)
            raise ValueError(msg)
        self.inner_radius = self.radii[0]
        self.outer_radius = self.radii[-1]
        self.delta_radii = np.diff(self.radii)
        self.depth = self.outer_radius - self.radii
        self.height = self.radii - self.inner_radius
        self.number = len(self.radii)
        # Includes 4*pi factor unlike C-version of SPIDER.
        self.area = 4 * np.pi * np.square(self.radii)
        mesh_cubed: np.ndarray = np.power(self.radii, 3)
        self.volume = 4 / 3 * np.pi * (mesh_cubed[1:] - mesh_cubed[:-1])
        self.total_volume = 4 / 3 * np.pi * (mesh_cubed[-1] - mesh_cubed[0])
        # TODO: To add conventional mixing length as well.
        self.mixing_length = 0.25 * (self.outer_radius - self.inner_radius)
        self.mixing_length_squared = np.square(self.mixing_length)
        self.mixing_length_cubed = np.power(self.mixing_length, 3)


@dataclass
class SpiderMesh(ScaledDataclassFromConfiguration):
    """A staggered mesh.

    The 'basic' mesh is used for the flux calculations and the 'staggered' mesh is used for the
    volume calculations.

    Args:
        radii: Radii of the basic nodes.
        numerical_scalings: Scalings for the numerical problem

    Attributes:
        radii: Radii of the basic nodes
        basic: The basic mesh.
        staggered: The staggered mesh.
    """

    scalings: Scalings
    _: KW_ONLY
    inner_radius: float
    outer_radius: float
    number_of_nodes: int
    basic: _FixedMesh = field(init=False)
    staggered: _FixedMesh = field(init=False)
    _d_dr_transform: np.ndarray = field(init=False)
    _quantity_transform: np.ndarray = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        basic_coordinates: np.ndarray = self.get_linear()
        self.basic = _FixedMesh(basic_coordinates)
        staggered_coordinates: np.ndarray = self.basic.radii[:-1] + 0.5 * self.basic.delta_radii
        self.staggered = _FixedMesh(staggered_coordinates)
        self._d_dr_transform = self.d_dr_transform_matrix()
        self._quantity_transform = self.quantity_transform_matrix()

    def scale_attributes(self) -> None:
        self.inner_radius /= self.scalings.radius
        self.outer_radius /= self.scalings.radius

    def get_linear(self) -> np.ndarray:
        radii: np.ndarray = np.linspace(self.inner_radius, self.outer_radius, self.number_of_nodes)
        return radii

    def d_dr_transform_matrix(self) -> np.ndarray:
        """Transform matrix for determining d/dr of a staggered quantity on the basic mesh.

        Returns:
            The transform matrix
        """
        transform: np.ndarray = np.zeros((self.basic.number, self.staggered.number))
        transform[1:-1, :-1] += np.diag(-1 / self.staggered.delta_radii)  # k=0 diagonal.
        transform[1:-1:, 1:] += np.diag(1 / self.staggered.delta_radii)  # k=1 diagonal.
        # Backward difference at outer radius.
        transform[0, :] = transform[1, :]
        # Forward difference at inner radius.
        transform[-1, :] = transform[-2, :]
        logger.debug("_d_dr_transform = %s", transform)

        return transform

    def d_dr_at_basic_nodes(self, staggered_quantity: np.ndarray) -> np.ndarray:
        """Determines d/dr at the basic nodes of a quantity defined at the staggered nodes.

        Args:
            staggered_quantity: A quantity defined at the staggered nodes.

        Returns:
            d/dr at the basic nodes
        """
        # assert np.size(staggered_quantity) == self.staggered.number

        d_dr_at_basic_nodes: np.ndarray = self._d_dr_transform.dot(staggered_quantity)
        logger.debug("d_dr_at_basic_nodes = %s", d_dr_at_basic_nodes)

        return d_dr_at_basic_nodes

    def quantity_transform_matrix(self) -> np.ndarray:
        """A transform matrix for mapping quantities on the staggered mesh to the basic mesh.

        Returns:
            The transform matrix
        """
        transform: np.ndarray = np.zeros((self.basic.number, self.staggered.number))
        mesh_ratio: np.ndarray = self.basic.delta_radii[:-1] / self.staggered.delta_radii
        transform[1:-1, :-1] += np.diag(1 - 0.5 * mesh_ratio)  # k=0 diagonal.
        transform[1:-1:, 1:] += np.diag(0.5 * mesh_ratio)  # k=1 diagonal.
        # Backward difference at inner radius.
        transform[0, :2] = np.array([1 + 0.5 * mesh_ratio[0], -0.5 * mesh_ratio[0]])
        # Forward difference at outer radius.
        mesh_ratio_outer: np.ndarray = self.basic.delta_radii[-1] / self.staggered.delta_radii[-1]
        transform[-1, -2:] = np.array([-0.5 * mesh_ratio_outer, 1 + 0.5 * mesh_ratio_outer])
        logger.debug("_quantity_transform = %s", transform)

        return transform

    def quantity_at_basic_nodes(self, staggered_quantity: np.ndarray) -> np.ndarray:
        """Determines a quantity at the basic nodes that is defined at the staggered nodes.

        Args:
            staggered_quantity: A quantity defined at the staggered nodes.

        Returns:
            The quantity at the basic nodes
        """
        quantity_at_basic_nodes: np.ndarray = self._quantity_transform.dot(staggered_quantity)
        logger.debug("quantity_at_basic_nodes = %s", quantity_at_basic_nodes)

        return quantity_at_basic_nodes


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


@dataclass(kw_only=True)
class Scalings(ScaledDataclassFromConfiguration):
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
        power_per_mass, W/kg
        power_per_volume, W/m^3
        pressure, Pa
        temperature_gradient, K/m
        thermal_conductivity, W/m/K
        velocity, m/s
        viscosity, Pa s
        time_years, years
        stefan_boltzmann_constant (non-dimensional)
    """

    # Default scalings
    radius: float = 1
    temperature: float = 1
    density: float = 1
    time: float = 1
    # Scalings (dimensional)
    area: float = field(init=False)
    gravitational_acceleration: float = field(init=False)
    heat_capacity: float = field(init=False)
    heat_flux: float = field(init=False)
    kinetic_energy_per_volume: float = field(init=False)
    power_per_mass: float = field(init=False)
    power_per_volume: float = field(init=False)
    pressure: float = field(init=False)
    temperature_gradient: float = field(init=False)
    time_years: float = field(init=False)  # Equivalent to TIMEYRS in C code version
    thermal_conductivity: float = field(init=False)
    velocity: float = field(init=False)
    viscosity: float = field(init=False)
    # Scalings (non-dimensional)
    stefan_boltzmann_constant: float = field(init=False)

    def scale_attributes(self):
        self.area = np.square(self.radius)
        self.gravitational_acceleration = self.radius / np.square(self.time)
        self.temperature_gradient = self.temperature / self.radius
        self.thermal_expansivity = 1 / self.temperature
        self.pressure = self.density * self.gravitational_acceleration * self.radius
        self.velocity = self.radius / self.time
        self.kinetic_energy_per_volume = self.density * np.square(self.velocity)
        self.heat_capacity = self.kinetic_energy_per_volume / self.density / self.temperature
        self.power_per_volume = self.kinetic_energy_per_volume / self.time
        self.power_per_mass = self.power_per_volume / self.density
        self.heat_flux = self.power_per_volume * self.radius
        self.thermal_conductivity = self.power_per_volume * self.area / self.temperature
        self.viscosity = self.pressure * self.time
        self.time_years = self.time / constants.Julian_year
        # Useful non-dimensional constants
        self.stefan_boltzmann_constant = codata.value("Stefan-Boltzmann constant")  # W/m^2/K^4
        self.stefan_boltzmann_constant /= (
            self.power_per_volume * self.radius / np.power(self.temperature, 4)
        )
        logger.debug("scalings = %s", self)


class SpiderConfigParser(ConfigParser):
    """Parser for SPIDER configuration files

    Args:
        *filenames: Filenames of one or several configuration files
    """

    getpath: Callable[..., Path]  # For typing.

    def __init__(self, *filenames):
        kwargs: dict = {
            "comment_prefixes": ("#",),
            "converters": {"path": Path, "any": lambda x: literal_eval(x)},
        }
        super().__init__(**kwargs)
        self.read(filenames)

    @property
    def phases(self) -> dict[str, SectionProxy]:
        """Dictionary of the sections relating to phases"""
        return {
            self[section].name.split("_")[1]: self[section]
            for section in self.sections()
            if section.startswith("phase_")
        }

    @property
    def radionuclides(self) -> dict[str, SectionProxy]:
        """Dictionary of the sections relating to radionuclides"""
        return {
            self[section].name.split("_")[1]: self[section]
            for section in self.sections()
            if section.startswith("radionuclide_")
        }


@dataclass
class SpiderData:
    """Container for the objects necessary to compute the interior evolution.

    This parsers the sections of the configuration data and instantiates the objects.

    Args:
        config_parser: A SpiderConfigParser

    Attributes:
        TODO
    """

    config_parser: SpiderConfigParser
    scalings: Scalings = field(init=False)
    boundary_conditions: BoundaryConditions = field(init=False)
    initial_condition: InitialCondition = field(init=False)
    mesh: StaggeredMesh = field(init=False)
    phases: dict[str, PhaseEvaluator] = field(init=False, default_factory=dict)
    phase: PhaseEvaluator = field(init=False)
    radionuclides: dict[str, Radionuclide] = field(init=False, default_factory=dict)

    def __post_init__(self):
        self.scalings = Scalings.from_configuration(section=self.config_parser["scalings"])
        self.boundary_conditions = BoundaryConditions.from_configuration(
            self.scalings, section=self.config_parser["boundary_conditions"]
        )
        self.mesh = StaggeredMesh.uniform_radii(self.scalings, **self.config_parser["mesh"])
        self.initial_condition = InitialCondition.from_configuration(
            self.scalings, self.mesh, section=self.config_parser["initial_condition"]
        )
        for phase_name, phase_section in self.config_parser.phases.items():
            phase: PhaseEvaluator = PhaseEvaluator.from_configuration(
                self.scalings, phase_name, section=phase_section
            )
            self.phases[phase_name] = phase
        for radionuclide_name, radionuclide_section in self.config_parser.radionuclides.items():
            radionuclide: Radionuclide = Radionuclide.from_configuration(
                self.scalings, radionuclide_name, section=radionuclide_section
            )
            self.radionuclides[radionuclide_name] = radionuclide
        self.set_phase()

    def set_phase(self) -> None:
        """Sets the phase"""

        if len(self.phases) == 1:
            phase_name, phase = next(iter(self.phases.items()))
            logger.info("Only one phase provided: %s", phase_name)
            self.phase = phase

        elif len(self.phases) == 2:
            logger.info("Two phases available, creating composite")
            raise NotImplementedError


def is_monotonic_increasing(some_array: np.ndarray) -> np.bool_:
    """Returns True if an array is monotonically increasing, otherwise returns False."""
    return np.all(np.diff(some_array) > 0)
