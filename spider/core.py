"""Core classes and functions

See the LICENSE file for licensing information.
"""

from __future__ import annotations

import logging
from ast import literal_eval
from configparser import ConfigParser, SectionProxy
from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
from scipy import constants
from thermochem import codata

from spider.interfaces import ScaledDataclassFromConfiguration
from spider.phase import PhaseEvaluator

if TYPE_CHECKING:
    from spider.solver import State

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class BoundaryConditions(ScaledDataclassFromConfiguration):
    """Boundary conditions

    Args:
        scalings: Scalings
        mesh: Mesh
        outer_boundary_condition: Outer boundary condition (flux, temperature, etc.)
        outer_boundary_value: Value of the outer boundary condition (flux, temperature, etc.)
        inner_boundary_condition: Inner boundary condition (flux, temperature. etc.)
        inner_boundary_value: Value of the inner boundary condition (flux, temperature, etc.)
        emissivity: Emissivity of the atmosphere (not necessarily used)
        equilibrium_temperature: Planetary equilibrium temperature (not necessarily used)
        core_radius: Radius of the core (not necessarily used)
        core_density: Density of the core (not necessarily used)
        core_heat_capacity: Heat capacity of the core (not necessarily used)

    Attributes:
        scalings: Scalings
        mesh: Mesh
        outer_boundary_condition: Outer boundary condition (flux, temperature, etc.)
        outer_boundary_value: Value of the outer boundary condition (flux, temperature, etc.)
        inner_boundary_condition: Inner boundary condition (flux, temperature. etc.)
        inner_boundary_value: Value of the inner boundary condition (flux, temperature, etc.)
        emissivity: Emissivity of the atmosphere (not necessarily used)
        equilibrium_temperature: Planetary equilibrium temperature (not necessarily used)
        core_radius: Radius of the core (not necessarily used)
        core_density: Density of the core (not necessarily used)
        core_heat_capacity: Heat capacity of the core (not necessarily used)
    """

    scalings: Scalings
    mesh: SpiderMesh
    _: KW_ONLY
    outer_boundary_condition: int
    outer_boundary_value: float  # Equivalent to surface_bc_value in C code.
    inner_boundary_condition: int
    inner_boundary_value: float  # Equivalent to core_bc_value in C Code.
    emissivity: float
    equilibrium_temperature: float
    core_radius: float
    core_density: float
    core_heat_capacity: float

    def scale_attributes(self):
        self.equilibrium_temperature /= self.scalings.temperature
        self.core_radius /= self.scalings.radius
        self.core_density /= self.scalings.density
        self.core_heat_capacity /= self.core_heat_capacity
        self._scale_inner_boundary_condition()
        self._scale_outer_boundary_condition()

    def _scale_inner_boundary_condition(self) -> None:
        """Scales the inner boundary value.

        Equivalent to CORE_BC in C code.
            1: simple core cooling
            2: prescribed heat flux
            3: prescribed temperature
        """
        if self.inner_boundary_condition == 1:
            self.inner_boundary_value = 0
        elif self.inner_boundary_condition == 2:
            self.inner_boundary_value /= self.scalings.heat_flux
        elif self.inner_boundary_condition == 3:
            self.inner_boundary_value /= self.scalings.temperature
        else:
            msg: str = "inner_boundary_condition = %d is unknown" % self.inner_boundary_condition
            logger.error(msg)
            raise ValueError(msg)

    def _scale_outer_boundary_condition(self) -> None:
        """Scales the outer boundary value.

        Equivalent to SURFACE_BC in C code.
            1: grey-body atmosphere
            2: Zahnle steam atmosphere
            3: self-consistent atmosphere evolution
            4: prescribed heat flux
            5: prescribed temperature
        """
        if self.outer_boundary_condition == 1:
            pass
        elif self.outer_boundary_condition == 2:
            pass
        elif self.outer_boundary_condition == 3:
            pass
        elif self.outer_boundary_condition == 4:
            self.outer_boundary_value /= self.scalings.heat_flux
        elif self.outer_boundary_condition == 5:
            self.outer_boundary_value /= self.scalings.temperature
        else:
            msg: str = "outer_boundary_condition = %d is unknown" % self.outer_boundary_condition
            logger.error(msg)
            raise ValueError(msg)

    def conform_temperature_boundary_conditions(
        self, temperature: np.ndarray, temperature_basic: np.ndarray, dTdr: np.ndarray
    ) -> None:
        """Conforms the temperature and dTdr at the basic nodes to temperature boundary conditions.

        Args:
            temperature: Temperature at the staggered nodes
            temperature_basic: Temperature at the basic nodes
            dTdr: Temperature gradient at the basic nodes
        """
        # Core-mantle boundary
        if self.inner_boundary_condition == 3:
            temperature_basic[0, :] = self.inner_boundary_value
            logger.debug(temperature[0, :])
            logger.debug(temperature_basic[0, :])
            logger.debug(self.mesh.basic.delta_radii[0])
            dTdr[0, :] = (
                2 * (temperature[0, :] - temperature_basic[0, :]) / self.mesh.basic.delta_radii[0]
            )
        # Surface
        if self.outer_boundary_condition == 5:
            temperature_basic[-1, :] = self.outer_boundary_value
            dTdr[-1, :] = (
                2
                * (temperature_basic[-1, :] - temperature[-1, :])
                / self.mesh.basic.delta_radii[-1]
            )

    def apply(self, state: State) -> None:
        """Applies the boundary conditions to the state.

        Args:
            state: The state to apply the boundary conditions to
        """
        self.apply_inner_boundary_condition(state)
        self.apply_outer_boundary_condition(state)
        logger.debug("temperature = %s", state.temperature_basic)
        logger.debug("heat_flux = %s", state.heat_flux)

    def apply_outer_boundary_condition(self, state: State) -> None:
        """Applies the outer boundary condition to the state.

        Args:
            state: The state to apply the boundary conditions to

        Equivalent to SURFACE_BC in C code.
            1: grey-body atmosphere
            2: Zahnle steam atmosphere
            3: self-consistent atmosphere evolution
            4: prescribed heat flux
            5: prescribed temperature
        """
        if self.outer_boundary_condition == 1:
            self.grey_body(state)
        elif self.outer_boundary_condition == 2:
            raise NotImplementedError
        elif self.outer_boundary_condition == 3:
            msg: str = "Requires coupling to atmodeller"
            logger.error(msg)
            raise NotImplementedError(msg)
        elif self.outer_boundary_condition == 4:
            state.heat_flux[-1, :] = self.outer_boundary_value
        elif self.outer_boundary_condition == 5:
            pass
            # raise NotImplementedError
        else:
            msg: str = "outer_boundary_condition = %d is unknown" % self.outer_boundary_condition
            logger.error(msg)
            raise ValueError(msg)

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

    def apply_inner_boundary_condition(self, state: State) -> None:
        """Applies the inner boundary condition to the state.

        Args:
            state: The state to apply the boundary conditions to

        Equivalent to CORE_BC in C code.
            1: simple core cooling
            2: prescribed heat flux
            3: prescribed temperature
        """
        if self.inner_boundary_condition == 1:
            raise NotImplementedError
        elif self.inner_boundary_condition == 2:
            state.heat_flux[0, :] = self.inner_boundary_value
        elif self.inner_boundary_condition == 3:
            pass
            # raise NotImplementedError
        else:
            msg: str = "inner_boundary_condition = %d is unknown" % self.inner_boundary_condition
            logger.error(msg)
            raise ValueError(msg)


@dataclass
class InitialCondition(ScaledDataclassFromConfiguration):
    """Initial condition

    Args:
        scalings: Scalings
        mesh: Mesh
        surface_temperature: Temperature of the "surface" (top staggered node)
        basal_temperature: Temperature of the base of the mantle (bottom staggered node)

    Attributes:
        scalings: Scalings
        mesh: Mesh
        surface_temperature: Temperature of the "surface" (top staggered node)
        basal_temperature: Temperature of the base of the mantle (bottom staggered node)
    """

    scalings: Scalings
    mesh: SpiderMesh
    boundary_conditions: BoundaryConditions
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
            self.basal_temperature, self.surface_temperature, self.mesh.staggered.number_of_nodes
        )
        return temperature


@dataclass
class _FixedMesh:
    """A fixed mesh

    All attributes have the same units as the input argument 'radii', but 'radii' should have
    non-dimensional units for a SPIDER model.

    Some quantities are column vectors (2-D arrays) to allow vectorised calculations
    (see scipy.integrate.solve_ivp)

    Args:
        radii: Radii of the mesh, which could be in dimensional or non-dimensional units.

    Attributes:
        radii: Radii of the mesh
        inner_radius: Inner radius
        outer_radius: Outer radius
        delta_radii: Delta radii
        depth: Depth below the outer radius
        height: Height above the inner radius
        number_of_nodes: Number of nodes
        area: Surface area
        volume: Volume of the spherical shells defined between neighbouring radii.
        mixing_length: Mixing length
        mixing_length_squared: Mixing length squared
        mixing_length_cubed: Mixing length cubed
    """

    radii: np.ndarray
    inner_radius: float = field(init=False)
    outer_radius: float = field(init=False)
    delta_radii: np.ndarray = field(init=False)
    depth: np.ndarray = field(init=False)
    height: np.ndarray = field(init=False)
    number_of_nodes: int = field(init=False)
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
        self.number_of_nodes = len(self.radii)
        # Includes 4*pi factor unlike C-version of SPIDER.
        self.area = 4 * np.pi * np.square(self.radii).reshape(-1, 1)  # 2-D
        mesh_cubed: np.ndarray = np.power(self.radii, 3)
        self.volume = 4 / 3 * np.pi * (mesh_cubed[1:] - mesh_cubed[:-1]).reshape(-1, 1)  # 2-D
        self.total_volume = 4 / 3 * np.pi * (mesh_cubed[-1] - mesh_cubed[0])
        # Average mixing length
        self.mixing_length = 0.25 * (self.outer_radius - self.inner_radius)
        # Conventional mixing length
        # self.mixing_length = np.minimum(
        #     self.outer_radius - self.radii, self.radii - self.inner_radius
        # ).reshape(-1, 1)
        # logger.debug("mixing_length = %s", self.mixing_length)
        self.mixing_length_squared = np.square(self.mixing_length)
        self.mixing_length_cubed = np.power(self.mixing_length, 3)


@dataclass
class SpiderMesh(ScaledDataclassFromConfiguration):
    """A staggered mesh.

    The 'basic' mesh is used for the flux calculations and the 'staggered' mesh is used for the
    volume calculations.

    Args:
        scalings: Scalings

    Attributes:
        scalings: Scalings
        inner_radius: Inner radius
        outer_radius: Outer radius
        number_of_nodes: Number of nodes
        basic: The basic mesh
        staggered: The staggered mesh
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
        """Temperature profile with a constant linear temperature gradient across the mantle

        Returns:
            Linear temperature profile
        """
        radii: np.ndarray = np.linspace(self.inner_radius, self.outer_radius, self.number_of_nodes)
        return radii

    def d_dr_transform_matrix(self) -> np.ndarray:
        """Transform matrix for determining d/dr of a staggered quantity on the basic mesh.

        Returns:
            The transform matrix
        """
        transform: np.ndarray = np.zeros(
            (self.basic.number_of_nodes, self.staggered.number_of_nodes)
        )
        transform[1:-1, :-1] += np.diag(-1 / self.staggered.delta_radii)  # k=0 diagonal.
        transform[1:-1:, 1:] += np.diag(1 / self.staggered.delta_radii)  # k=1 diagonal.
        # Backward difference at outer radius.
        transform[0, :] = transform[1, :]
        # Forward difference at inner radius.
        transform[-1, :] = transform[-2, :]
        logger.debug("_d_dr_transform_matrix = %s", transform)

        return transform

    def d_dr_at_basic_nodes(self, staggered_quantity: np.ndarray) -> np.ndarray:
        """Determines d/dr at the basic nodes of a quantity defined at the staggered nodes.

        Args:
            staggered_quantity: A quantity defined at the staggered nodes.

        Returns:
            d/dr at the basic nodes
        """
        d_dr_at_basic_nodes: np.ndarray = self._d_dr_transform.dot(staggered_quantity)
        logger.debug("d_dr_at_basic_nodes = %s", d_dr_at_basic_nodes)

        return d_dr_at_basic_nodes

    def quantity_transform_matrix(self) -> np.ndarray:
        """A transform matrix for mapping quantities on the staggered mesh to the basic mesh.

        Uses backward and forward differences at the inner and outer radius, respectively, to
        obtain the quantity values of the basic nodes at the innermost and outermost nodes. It may
        be subsequently necessary to conform these outer boundaries to applied boundary conditions.

        Returns:
            The transform matrix
        """
        transform: np.ndarray = np.zeros(
            (self.basic.number_of_nodes, self.staggered.number_of_nodes)
        )
        mesh_ratio: np.ndarray = self.basic.delta_radii[:-1] / self.staggered.delta_radii
        transform[1:-1, :-1] += np.diag(1 - 0.5 * mesh_ratio)  # k=0 diagonal.
        transform[1:-1:, 1:] += np.diag(0.5 * mesh_ratio)  # k=1 diagonal.
        # Backward difference at inner radius.
        transform[0, :2] = np.array([1 + 0.5 * mesh_ratio[0], -0.5 * mesh_ratio[0]]).flatten()
        # Forward difference at outer radius.
        mesh_ratio_outer: np.ndarray = self.basic.delta_radii[-1] / self.staggered.delta_radii[-1]
        transform[-1, -2:] = np.array(
            [-0.5 * mesh_ratio_outer, 1 + 0.5 * mesh_ratio_outer]
        ).flatten()
        logger.debug("_quantity_transform_matrix = %s", transform)

        return transform

    def quantity_at_basic_nodes(self, staggered_quantity: np.ndarray) -> np.ndarray:
        """Determines a quantity at the basic nodes that is defined at the staggered nodes.

        Uses backward and forward differences at the inner and outer radius, respectively, to
        obtain the quantity values of the basic nodes at the innermost and outermost nodes. It may
        be subsequently necessary to conform these outer boundaries to applied boundary conditions.

        Args:
            staggered_quantity: A quantity defined at the staggered nodes

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
    mesh: SpiderMesh = field(init=False)
    phases: dict[str, PhaseEvaluator] = field(init=False, default_factory=dict)
    phase: PhaseEvaluator = field(init=False)
    radionuclides: dict[str, Radionuclide] = field(init=False, default_factory=dict)

    def __post_init__(self):
        self.scalings = Scalings.from_configuration(section=self.config_parser["scalings"])
        self.mesh = SpiderMesh.from_configuration(
            self.scalings, section=self.config_parser["mesh"]
        )
        self.boundary_conditions = BoundaryConditions.from_configuration(
            self.scalings, self.mesh, section=self.config_parser["boundary_conditions"]
        )

        self.initial_condition = InitialCondition.from_configuration(
            self.scalings,
            self.mesh,
            self.boundary_conditions,
            section=self.config_parser["initial_condition"],
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
