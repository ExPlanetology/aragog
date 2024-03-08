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
"""Core classes and functions"""

from __future__ import annotations

import logging
import sys
from ast import literal_eval
from configparser import ConfigParser, SectionProxy
from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
from scipy import constants
from thermochem import codata

from spider.interfaces import ScaledDataclassFromConfiguration
from spider.phase import MixedPhaseEvaluator, PhaseEvaluator, PhaseEvaluatorProtocol

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

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
        See Args
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

    @override
    def scale_attributes(self) -> None:
        """See base class."""
        self.equilibrium_temperature /= self.scalings.temperature
        self.core_radius /= self.scalings.radius
        self.core_density /= self.scalings.density
        self.core_heat_capacity /= self.core_heat_capacity
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
            self.inner_boundary_value /= self.scalings.heat_flux
        elif self.inner_boundary_condition == 3:
            self.inner_boundary_value /= self.scalings.temperature
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
            self.outer_boundary_value /= self.scalings.heat_flux
        elif self.outer_boundary_condition == 5:
            self.outer_boundary_value /= self.scalings.temperature
        else:
            msg: str = f"outer_boundary_condition = {self.outer_boundary_condition} is unknown"
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

    # TODO: Rename to only be associated with flux boundary conditions
    def apply_outer_boundary_condition(self, state: State) -> None:
        """Applies the outer boundary condition to the state.

        Args:
            state: The state to apply the boundary conditions to

        Equivalent to SURFACE_BC in C code.
            1: Grey-body atmosphere
            2: Zahnle steam atmosphere
            3: Couple to atmodeller
            4: Prescribed heat flux
            5: Prescribed temperature
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
        else:
            msg: str = f"outer_boundary_condition = {self.outer_boundary_condition} is unknown"
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

    # TODO: Rename to only be associated with flux boundary conditions
    def apply_inner_boundary_condition(self, state: State) -> None:
        """Applies the inner boundary condition to the state.

        Args:
            state: The state to apply the boundary conditions to

        Equivalent to CORE_BC in C code.
            1: Simple core cooling
            2: Prescribed heat flux
            3: Prescribed temperature
        """
        if self.inner_boundary_condition == 1:
            raise NotImplementedError
        elif self.inner_boundary_condition == 2:
            state.heat_flux[0, :] = self.inner_boundary_value
        elif self.inner_boundary_condition == 3:
            pass
            # raise NotImplementedError
        else:
            msg: str = f"inner_boundary_condition = {self.inner_boundary_condition} is unknown"
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
        See Args
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

    @override
    def scale_attributes(self) -> None:
        """See base class."""
        self.surface_temperature /= self.scalings.temperature
        self.basal_temperature /= self.scalings.temperature

    @property
    def temperature(self) -> np.ndarray:
        return self._temperature

    # TODO: Clunky. Set the staggered and basic temperature together, or be clear which one is
    # being set.
    def get_linear(self) -> np.ndarray:
        """Gets a linear temperature profile

        Returns:
            Linear temperature profile for the staggered nodes
        """
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
        radii: Radii of the mesh, which could be in dimensional or non-dimensional units
        mixing_length_profile: Profile of the mixing length. Can be 'nearest_boundary' or
            'constant'

    Attributes:
        radii: Radii of the mesh
        mixing_length_profile: Profile of the mixing length
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

    # TODO: Pass in initialised EOS object to compute the pressure radius relationship?

    radii: np.ndarray
    mixing_length_profile: str

    def __post_init__(self):
        if not is_monotonic_increasing(self.radii):
            msg: str = "Mesh must be monotonically increasing"
            logger.error(msg)
            raise ValueError(msg)
        self.inner_radius: float = self.radii[0]
        self.outer_radius: float = self.radii[-1]
        self.delta_radii: np.ndarray = np.diff(self.radii)
        self.depth: np.ndarray = self.outer_radius - self.radii
        self.height: np.ndarray = self.radii - self.inner_radius
        self.number_of_nodes: int = len(self.radii)
        # Includes 4*pi factor unlike C-version of SPIDER.
        self.area: np.ndarray = 4 * np.pi * np.square(self.radii).reshape(-1, 1)  # 2-D
        mesh_cubed: np.ndarray = np.power(self.radii, 3)
        self.volume: np.ndarray = (
            4 / 3 * np.pi * (mesh_cubed[1:] - mesh_cubed[:-1]).reshape(-1, 1)
        )  # 2-D
        self.total_volume: float = 4 / 3 * np.pi * (mesh_cubed[-1] - mesh_cubed[0])
        self.set_mixing_length()
        self.mixing_length_squared: np.ndarray = np.square(self.mixing_length)
        self.mixing_length_cubed: np.ndarray = np.power(self.mixing_length, 3)

    def set_mixing_length(self) -> None:
        """Sets the mixing length"""
        if self.mixing_length_profile == "nearest_boundary":
            logger.debug("Set mixing length profile to nearest_boundary")
            self.mixing_length = np.minimum(
                self.outer_radius - self.radii, self.radii - self.inner_radius
            )
        elif self.mixing_length_profile == "constant":
            logger.debug("Set mixing length profile to constant")
            self.mixing_length = (
                np.ones(self.radii.size) * 0.25 * (self.outer_radius - self.inner_radius)
            )
        else:
            msg: str = f"Mixing length profile = {self.mixing_length_profile} is unknown"
            raise ValueError(msg)

        self.mixing_length = self.mixing_length.reshape(-1, 1)  # 2-D


@dataclass
class SpiderMesh(ScaledDataclassFromConfiguration):
    """A staggered mesh.

    The 'basic' mesh is used for the flux calculations and the 'staggered' mesh is used for the
    volume calculations.

    # TODO Update with Adams williamson EOS

    Args:
        scalings: Scalings

    Attributes:
        scalings: Scalings
        inner_radius: Inner radius
        outer_radius: Outer radius
        number_of_nodes: Number of nodes
        mixing_length_profile: The mixing length profile
        basic: The basic mesh
        staggered: The staggered mesh
    """

    scalings: Scalings
    _: KW_ONLY
    inner_radius: float
    outer_radius: float
    number_of_nodes: int
    mixing_length_profile: str
    gravitational_acceleration: float
    adams_williamson_surface_density: float
    adams_williamson_beta: float

    def __post_init__(self):
        super().__post_init__()
        basic_coordinates: np.ndarray = self.get_constant_spacing()
        self.basic: _FixedMesh = _FixedMesh(basic_coordinates, self.mixing_length_profile)
        staggered_coordinates: np.ndarray = self.basic.radii[:-1] + 0.5 * self.basic.delta_radii
        self.staggered: _FixedMesh = _FixedMesh(staggered_coordinates, self.mixing_length_profile)
        self._d_dr_transform: np.ndarray = self.d_dr_transform_matrix()
        self._quantity_transform: np.ndarray = self.quantity_transform_matrix()

    @override
    def scale_attributes(self) -> None:
        """See base class."""
        self.inner_radius /= self.scalings.radius
        self.outer_radius /= self.scalings.radius

    def get_constant_spacing(self) -> np.ndarray:
        """Constant radius spacing across the mantle

        Returns:
            Radii with constant spacing
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
        transform[1:-1, :-1] += np.diag(-1 / self.staggered.delta_radii)  # k=0 diagonal
        transform[1:-1:, 1:] += np.diag(1 / self.staggered.delta_radii)  # k=1 diagonal
        transform[0, :] = transform[1, :]  # Backward difference at outer radius.
        transform[-1, :] = transform[-2, :]  # Forward difference at inner radius.
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

    # TODO: Compatibility with conforming boundary/initial conditions?
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

    # TODO: Compatibility with conforming boundary/initial conditions?
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
        t0_years: Time at which quantities are defined
        abundance: Abundance
        concentration: Concentration
        heat_production: Heat production
        half_life_years: Half life

    Attributes:
        See Args.
    """

    scalings: Scalings
    name: str
    _: KW_ONLY
    t0_years: float
    abundance: float
    concentration: float
    heat_production: float
    half_life_years: float

    @override
    def scale_attributes(self) -> None:
        self.t0_years /= self.scalings.time_years
        self.concentration *= 1e-6  # to mass fraction
        self.heat_production /= self.scalings.power_per_mass
        self.half_life_years /= self.scalings.time_years

    def get_heating(self, time: np.ndarray | float) -> np.ndarray | float:
        """Radiogenic heating

        Args:
            time: Time

        Returns:
            Radiogenic heating as a float if time is a float, otherwise a numpy row array where
                each entry in the row is associated with a single time in the time array.
        """
        arg: np.ndarray | float = np.log(2) * (self.t0_years - time) / self.half_life_years
        heating: np.ndarray | float = (
            self.heat_production * self.abundance * self.concentration * np.exp(arg)
        )

        return heating

    # TODO: Remove eventually, now moved into parser.py
    # @dataclass(kw_only=True)
    # class Scalings(ScaledDataclassFromConfiguration):
    #     """Scalings for the numerical problem.

    #     Args:
    #         radius: Radius in metres. Defaults to 1.
    #         temperature: Temperature in Kelvin. Defaults to 1.
    #         density: Density in kg/m^3. Defaults to 1.
    #         time: Time in seconds. Defaults to 1.

    #     Attributes:
    #         radius, m
    #         temperature, K
    #         density, kg/m^3
    #         time, s
    #         area, m^2
    #         kinetic_energy_per_volume, J/m^3
    #         gravitational_acceleration, m/s^2
    #         heat_capacity, J/kg/K
    #         heat_flux, W/m^2
    #         latent_heat_per_mass, J/kg
    #         power_per_mass, W/kg
    #         power_per_volume, W/m^3
    #         pressure, Pa
    #         temperature_gradient, K/m
    #         thermal_expansivity, 1/K
    #         thermal_conductivity, W/m/K
    #         velocity, m/s
    #         viscosity, Pa s
    #         time_years, years
    #         stefan_boltzmann_constant (non-dimensional)
    #     """

    #     radius: float = 1
    #     temperature: float = 1
    #     density: float = 1
    #     time: float = 1

    #     @override
    #     def scale_attributes(self) -> None:
    #         self.area: float = np.square(self.radius)
    #         self.gravitational_acceleration: float = self.radius / np.square(self.time)
    #         self.temperature_gradient: float = self.temperature / self.radius
    #         self.thermal_expansivity: float = 1 / self.temperature
    #         self.pressure: float = self.density * self.gravitational_acceleration * self.radius
    #         self.velocity: float = self.radius / self.time
    #         self.kinetic_energy_per_volume: float = self.density * np.square(self.velocity)
    #         self.heat_capacity: float = (
    #             self.kinetic_energy_per_volume / self.density / self.temperature
    #         )
    #         self.latent_heat_per_mass: float = self.heat_capacity * self.temperature
    #         self.power_per_volume: float = self.kinetic_energy_per_volume / self.time
    #         self.power_per_mass: float = self.power_per_volume / self.density
    #         self.heat_flux: float = self.power_per_volume * self.radius
    #         self.thermal_conductivity: float = self.power_per_volume * self.area / self.temperature
    #         self.viscosity: float = self.pressure * self.time
    #         self.time_years: float = self.time / constants.Julian_year  # Equivalent to TIMEYRS C code
    #         # Non-dimensional constants
    #         self.stefan_boltzmann_constant: float = codata.value(
    #             "Stefan-Boltzmann constant"
    #         )  # W/m^2/K^4
    #         self.stefan_boltzmann_constant /= (
    #             self.power_per_volume * self.radius / np.power(self.temperature, 4)
    #         )
    #         logger.debug("scalings = %s", self)

    # @property
    # def phase_boundary(self) -> float:
    #     """For scaling phase boundary"""
    #     return self.temperature


class SpiderConfigParser(ConfigParser):
    """Parser for SPIDER configuration files

    Args:
        *filenames: Filenames of one or several configuration files
    """

    getpath: Callable[..., Path]  # For typing.

    def __init__(self, *filenames):
        kwargs: dict = {
            "comment_prefixes": ("#",),
            "converters": {"path": Path, "any": literal_eval},
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
        config_parser: A SpiderConfigParser
        scalings: Scalings
        boundary_conditions: Boundary conditions
        initial_condition: Initial condition
        mesh: Mesh
        phases: Phases
        phase: Active phase for evaluation
        radionuclides: Radionuclides
    """

    config_parser: SpiderConfigParser
    scalings: Scalings = field(init=False)
    boundary_conditions: BoundaryConditions = field(init=False)
    initial_condition: InitialCondition = field(init=False)
    mesh: SpiderMesh = field(init=False)
    phases: dict[str, PhaseEvaluator] = field(init=False, default_factory=dict)
    phase: PhaseEvaluatorProtocol = field(init=False)
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
            logger.info("Two phases found so creating a composite")
            self.phase = MixedPhaseEvaluator.from_configuration(
                self.scalings, self.phases, section=self.config_parser["mixed_phase"]
            )


def is_monotonic_increasing(some_array: np.ndarray) -> np.bool_:
    """Returns True if an array is monotonically increasing, otherwise returns False."""
    return np.all(np.diff(some_array) > 0)


@dataclass
class AdamsWilliamsonEOS:
    """Adams-Williamson equation of state

    Args:
        scalings: Scalings
        mesh: Mesh
        surface_density: Surface density
        beta: Beta parameter
        gravitational_acceleration: Gravitational acceleration

    Attributes:
        TODO
    """

    outer_radius: float
    surface_density: float
    beta: float
    gravitational_acceleration: float

    def density(self) -> np.ndarray:
        """Density

        TODO: Convert math to rst

        Adams-Williamson density is a simple function of depth (radius)
        Sketch derivation:
            dP/dr = dP/drho * drho/dr = -rho g
            dP/drho \sim (dP/drho)_s (adiabatic)
            drho/dr = -rho g / Si
            then integrate to give the form rho(r) = k * exp(-(g*r)/c)
            (g is positive)
            apply the limit that rho = rhos at r=R
            gives:
            rho(z) = rhos * exp( beta * z )
        where z = R-r

        this is arguably the simplest relation to get rho directly from r, but other
        EOSs can be envisaged
        """
        # using pressure, expression is simpler than sketch derivation above
        density: np.ndarray = (
            self.surface_density + self.pressure() * self.beta / self.gravitational_acceleration
        )

        return density

    def mass_element(self) -> np.ndarray:
        """Mass element"""
        # TODO: Check because C Spider does not include 4*pi scaling
        mass_element: np.ndarray = self.mesh.area * self.density()

        return mass_element

    def mass_within_radius(self) -> np.ndarray:
        """Mass contained within radii

        Returns:
            Mass within a radius
        """
        mass: np.ndarray = (
            -2 / self.beta**3
            - np.square(self.mesh.radii) / self.beta
            - 2 * self.mesh.radii / np.square(self.beta)
        )
        # TODO: Check because C Spider does not include 4*pi scaling
        mass *= 4 * np.pi * self.density()

        return mass

    def mass_within_shell(self) -> np.ndarray:
        """Mass within a spherical shell

        Returns:
            Mass within a spherical shell
        """
        # From outer radius to inner
        mass_within_radius: np.ndarray = np.flip(self.mass_within_radius())
        # Return same order as radii
        delta_mass: np.ndarray = np.flip(mass_within_radius[:-1] - mass_within_radius[1:])

        return delta_mass

    def pressure(self) -> np.ndarray:
        """Pressure

        Returns:
            Pressure
        """
        factor: float = self.surface_density * self.gravitational_acceleration / self.beta
        pressure: np.ndarray = factor * (
            np.exp(self.beta * (self.mesh.outer_radius - self.mesh.radii)) - 1
        )

        return pressure

    def pressure_gradient(self) -> np.ndarray:
        """Pressure gradient

        Returns:
            Pressure gradient
        """
        dPdr: np.ndarray = -self.gravitational_acceleration * self.density()

        return dPdr
