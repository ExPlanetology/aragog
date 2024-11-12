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
"""Mesh"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import numpy.typing as npt

from aragog.parser import Parameters, _MeshParameters
from aragog.utilities import FloatOrArray, is_monotonic_increasing

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class FixedMesh:
    """A fixed mesh

    Args:
        settings: Mesh parameters
        radii: Radii of the mesh
        outer_boundary: Outer boundary for computing depth below the surface
        inner_boundary: Inner boundary for computing height above the base

    Attributes:
        settings: Mesh parameters
        radii: Radii of the mesh
        outer_boundary: Outer boundary for computing depth below the surface
        inner_boundary: Inner boundary for computing height above the base
        area: Surface area
        delta_radii: Delta radii
        density: Density
        depth: Depth below the outer boundary
        height: Height above the inner boundary
        mixing_length: Mixing length
        mixing_length_squared: Mixing length squared
        mixing_length_cubed: Mixing length cubed
        number_of_nodes: Number of nodes
        pressure: Pressure
        pressure_gradient: Pressure gradient (dP/dr)
        volume: Volume of the spherical shells defined between neighbouring radii
        total_volume: Total volume
    """

    settings: _MeshParameters
    radii: npt.NDArray
    outer_boundary: float
    inner_boundary: float
    eos: AdamsWilliamsonEOS = field(init=False)

    def __post_init__(self):
        if not is_monotonic_increasing(self.radii):
            msg: str = "Mesh must be monotonically increasing"
            logger.error(msg)
            raise ValueError(msg)
        self.eos = AdamsWilliamsonEOS(
            self.settings, self.radii, self.outer_boundary, self.inner_boundary
        )

    @cached_property
    def area(self) -> npt.NDArray:
        """Area"""
        return 4 * np.pi * np.square(self.radii)

    @cached_property
    def delta_radii(self) -> npt.NDArray:
        return np.diff(self.radii, axis=0)

    @cached_property
    def density(self) -> npt.NDArray:
        return self.eos.density

    @cached_property
    def depth(self) -> npt.NDArray:
        return self.outer_boundary - self.radii

    @cached_property
    def height(self) -> npt.NDArray:
        return self.radii - self.inner_boundary

    @cached_property
    def _mesh_cubed(self) -> npt.NDArray:
        return np.power(self.radii, 3)

    @cached_property
    def mixing_length(self) -> npt.NDArray:
        if self.settings.mixing_length_profile == "nearest_boundary":
            logger.debug("Set mixing length profile to nearest boundary")
            mixing_length = np.minimum(
                self.outer_boundary - self.radii, self.radii - self.inner_boundary
            )
        elif self.settings.mixing_length_profile == "constant":
            logger.debug("Set mixing length profile to constant")
            assert self.outer_boundary is not None
            assert self.inner_boundary is not None
            mixing_length = (
                np.ones_like(self.radii) * 0.25 * (self.outer_boundary - self.inner_boundary)
            )
        else:
            msg: str = f"Mixing length profile = {self.settings.mixing_length_profile} is unknown"
            raise ValueError(msg)

        return mixing_length

    @cached_property
    def mixing_length_cubed(self) -> npt.NDArray:
        return np.power(self.mixing_length, 3)

    @cached_property
    def mixing_length_squared(self) -> npt.NDArray:
        return np.square(self.mixing_length)

    @cached_property
    def number_of_nodes(self) -> int:
        return self.radii.size

    @cached_property
    def pressure(self) -> npt.NDArray:
        return self.eos.pressure

    @cached_property
    def pressure_gradient(self) -> npt.NDArray:
        return self.eos.pressure_gradient

    @cached_property
    def volume(self) -> npt.NDArray:
        volume: npt.NDArray = 4 / 3 * np.pi * (self._mesh_cubed[1:] - self._mesh_cubed[:-1])

        return volume

    @cached_property
    def total_volume(self) -> float:
        return 4 / 3 * np.pi * float(self._mesh_cubed[-1] - self._mesh_cubed[0])


class Mesh:
    """A staggered mesh.

    The basic mesh is used for the flux calculations and the staggered mesh is used for the volume
    calculations.

    Args:
        parameters: Parameters
    """

    def __init__(self, parameters: Parameters):
        self.settings: _MeshParameters = parameters.mesh
        basic_coordinates: npt.NDArray = self.get_constant_spacing()
        self.basic: FixedMesh = FixedMesh(
            self.settings, basic_coordinates, np.max(basic_coordinates), np.min(basic_coordinates)
        )
        staggered_coordinates: npt.NDArray = self.basic.radii[:-1] + 0.5 * self.basic.delta_radii
        self.staggered: FixedMesh = FixedMesh(
            self.settings,
            staggered_coordinates,
            self.basic.outer_boundary,
            self.basic.inner_boundary,
        )
        self._d_dr_transform: npt.NDArray = self._get_d_dr_transform_matrix()
        self._quantity_transform: npt.NDArray = self._get_quantity_transform_matrix()

    def get_constant_spacing(self) -> npt.NDArray:
        """Constant radius spacing across the mantle

        Returns:
            Radii with constant spacing as a column vector
        """
        radii: npt.NDArray = np.linspace(
            self.settings.inner_radius, self.settings.outer_radius, self.settings.number_of_nodes
        )
        radii = np.atleast_2d(radii).T

        return radii

    def _get_d_dr_transform_matrix(self) -> npt.NDArray:
        """Transform matrix for determining d/dr of a staggered quantity on the basic mesh.

        Returns:
            The transform matrix
        """
        transform: npt.NDArray = np.zeros(
            (self.basic.number_of_nodes, self.staggered.number_of_nodes)
        )
        transform[1:-1, :-1] += np.diagflat(-1 / self.staggered.delta_radii)  # k=0 diagonal
        transform[1:-1:, 1:] += np.diagflat(1 / self.staggered.delta_radii)  # k=1 diagonal
        transform[0, :] = transform[1, :]  # Backward difference at outer radius
        transform[-1, :] = transform[-2, :]  # Forward difference at inner radius
        logger.debug("_d_dr_transform_matrix = %s", transform)

        return transform

    def d_dr_at_basic_nodes(self, staggered_quantity: npt.NDArray) -> npt.NDArray:
        """Determines d/dr at the basic nodes of a quantity defined at the staggered nodes.

        Args:
            staggered_quantity: A quantity defined at the staggered nodes.

        Returns:
            d/dr at the basic nodes
        """
        d_dr_at_basic_nodes: npt.NDArray = self._d_dr_transform.dot(staggered_quantity)
        logger.debug("d_dr_at_basic_nodes = %s", d_dr_at_basic_nodes)

        return d_dr_at_basic_nodes

    # TODO: Compatibility with conforming boundary/initial conditions?
    def _get_quantity_transform_matrix(self) -> npt.NDArray:
        """A transform matrix for mapping quantities on the staggered mesh to the basic mesh.

        Uses backward and forward differences at the inner and outer radius, respectively, to
        obtain the quantity values of the basic nodes at the innermost and outermost nodes. It may
        be subsequently necessary to conform these outer boundaries to applied boundary conditions.

        Returns:
            The transform matrix
        """
        transform: npt.NDArray = np.zeros(
            (self.basic.number_of_nodes, self.staggered.number_of_nodes)
        )
        mesh_ratio: npt.NDArray = self.basic.delta_radii[:-1] / self.staggered.delta_radii
        transform[1:-1, :-1] += np.diagflat(1 - 0.5 * mesh_ratio)  # k=0 diagonal
        transform[1:-1:, 1:] += np.diagflat(0.5 * mesh_ratio)  # k=1 diagonal
        # Backward difference at inner radius
        transform[0, :2] = np.array([1 + 0.5 * mesh_ratio[0], -0.5 * mesh_ratio[0]]).flatten()
        # Forward difference at outer radius
        mesh_ratio_outer: npt.NDArray = self.basic.delta_radii[-1] / self.staggered.delta_radii[-1]
        transform[-1, -2:] = np.array(
            [-0.5 * mesh_ratio_outer, 1 + 0.5 * mesh_ratio_outer]
        ).flatten()
        logger.debug("_quantity_transform_matrix = %s", transform)

        return transform

    # TODO: Compatibility with conforming boundary/initial conditions?
    def quantity_at_basic_nodes(self, staggered_quantity: npt.NDArray) -> npt.NDArray:
        """Determines a quantity at the basic nodes that is defined at the staggered nodes.

        Uses backward and forward differences at the inner and outer radius, respectively, to
        obtain the quantity values of the basic nodes at the innermost and outermost nodes. It may
        be subsequently necessary to conform these outer boundaries to applied boundary conditions.

        Args:
            staggered_quantity: A quantity defined at the staggered nodes

        Returns:
            The quantity at the basic nodes
        """
        quantity_at_basic_nodes: npt.NDArray = self._quantity_transform.dot(staggered_quantity)
        logger.debug("quantity_at_basic_nodes = %s", quantity_at_basic_nodes)

        return quantity_at_basic_nodes

    def volume_average(self, staggered_quantity: npt.NDArray) -> float:
        return float(np.dot(staggered_quantity.T, self.basic.volume)) / self.basic.total_volume


class AdamsWilliamsonEOS:
    r"""Adams-Williamson equation of state (EOS).

    EOS due to adiabatic self-compression from the definition of the adiabatic bulk modulus:

    .. math::

        \left( \frac{{d\rho}}{{dP}} \right)_S = \frac{{\rho}}{{K_S}}

    where :math:`\rho` is density, :math:`K_S` the adiabatic bulk modulus, and :math:`S` is
    entropy.
    """

    def __init__(
        self,
        settings: _MeshParameters,
        radii: npt.NDArray,
        outer_boundary: float,
        inner_boundary: float,
    ):
        self._settings: _MeshParameters = settings
        self._radii: npt.NDArray = radii
        self._outer_boundary = outer_boundary
        self._inner_boundary = inner_boundary
        self._surface_density: float = self._settings.surface_density
        self._gravitational_acceleration: float = self._settings.gravitational_acceleration
        self._adiabatic_bulk_modulus: float = self._settings.adiabatic_bulk_modulus
        self._pressure = self.get_pressure_from_radii(radii)
        self._pressure_gradient = self.get_pressure_gradient(self.pressure)
        self._density = self.get_density(self.pressure)

    @property
    def density(self) -> npt.NDArray:
        """Density"""
        return self._density

    @property
    def pressure(self) -> npt.NDArray:
        """Pressure"""
        return self._pressure

    @property
    def pressure_gradient(self) -> npt.NDArray:
        """Pressure gradient"""
        return self.pressure_gradient

    def get_density(self, pressure: FloatOrArray) -> npt.NDArray:
        r"""Computes density from pressure:

        .. math::

            \rho(P) = \rho_s \exp(P/K_S)

        where :math:`\rho` is density, :math:`P` is pressure, :math:`\rho_s` is surface density,
        and :math:`K_S` is adiabatic bulk modulus.

        Args:
            pressure: Pressure

        Returns:
            Density
        """
        density: npt.NDArray = self._surface_density * np.exp(
            pressure / self._adiabatic_bulk_modulus
        )

        return density

    def get_density_from_radii(self, radii: FloatOrArray) -> FloatOrArray:
        r"""Computes density from radii:

        .. math::

            \rho(r) = \frac{\rho_s K_S}{K_S + \rho_s g (r-r_s)}

        where :math:`\rho` is density, :math:`r` is radius, :math:`\rho_s` is surface density,
        :math:`K_S` is adiabatic bulk modulus, and :math:`r_s` is surface radius.

        Args:
            radii: Radii

        Returns
            Density
        """
        density: FloatOrArray = (self._surface_density * self._adiabatic_bulk_modulus) / (
            self._adiabatic_bulk_modulus
            + self._surface_density
            * self._gravitational_acceleration
            * (radii - self._outer_boundary)
        )

        return density

    def get_mass_element(self, radii: FloatOrArray) -> npt.NDArray:
        r"""Computes the mass element:

        .. math::

            \frac{\delta m}{\delta r} = 4 \pi r^2 \rho

        where :math:`\delta m` is the mass element, :math:`r` is radius, and :math:`\rho` is
        density.

        Args:
            radii: Radii

        Returns:
            The mass element at radii
        """
        mass_element: npt.NDArray = (
            4 * np.pi * np.square(radii) * self.get_density_from_radii(radii)
        )

        return mass_element

    def get_mass_within_radii(self, radii: FloatOrArray) -> npt.NDArray:
        r"""Computes mass within radii:

        .. math::

            m(r) = \int 4 \pi r^2 \rho dr

        where :math:`m` is mass, :math:`r` is radius, and :math:`\rho` is density.

        The integral was evaluated using WolframAlpha.

        Args:
            radii: Radii

        Returns:
            Mass within radii
        """
        a: float = self._surface_density
        b: float = self._adiabatic_bulk_modulus
        c: float = self._gravitational_acceleration
        d: float = self._outer_boundary

        def mass_integral(radii_: FloatOrArray) -> npt.NDArray:
            """Mass within radii including arbitrary constant of integration.

            Args:
                radii_: Radii

            Returns:
                Mass within radii
            """

            mass: npt.NDArray = (
                4
                * np.pi
                * (
                    b
                    * (
                        a * c * radii_ * (a * c * (2 * d + radii_) - 2 * b)
                        + 2 * np.square(b - a * c * d) * np.log(a * c * (radii_ - d) + b)
                    )
                    / (2 * np.square(a) + np.power(c, 3))
                )
            )
            # + constant

            return mass

        mass: npt.NDArray = mass_integral(radii) - mass_integral(self._inner_boundary)

        return mass

    def get_mass_within_shell(self, radii: npt.NDArray) -> npt.NDArray:
        """Computes the mass within spherical shells bounded by radii.

        Args:
            radii: Radii

        Returns:
            Mass within the bounded spherical shells
        """
        mass: npt.NDArray = self.get_mass_within_radii(radii[1:]) - self.get_mass_within_radii(
            radii[:-1]
        )

        return mass

    def get_pressure_from_radii(self, radii: FloatOrArray) -> npt.NDArray:
        r"""Computes pressure from radii:

        .. math::

            P(r) = -K_S \ln \left( 1 + \frac{\rho_s g (r-r_s)}{K_S} \right)

        where :math:`r` is radius, :math:`K_S` is adiabatic bulk modulus, :math:`P` is pressure,
        :math:`\rho_s` is surface density, :math:`g` is gravitational acceleration, and
        :math:`r_s` is surface radius.

        Args:
            radii: Radii

        Returns:
            Pressure
        """
        pressure: npt.NDArray = -self._adiabatic_bulk_modulus * np.log(
            (
                self._adiabatic_bulk_modulus
                + self._surface_density
                * self._gravitational_acceleration
                * (radii - self._outer_boundary)
            )
            / self._adiabatic_bulk_modulus
        )

        return pressure

    def get_pressure_gradient(self, pressure: FloatOrArray) -> npt.NDArray:
        r"""Computes the pressure gradient:

        .. math::

            \frac{dP}{dr} = -g \rho

        where :math:`\rho` is density, :math:`P` is pressure, and  :math:`g` is gravitational
        acceleration.

        Args:
            pressure: Pressure

        Returns:
            Pressure gradient
        """
        dPdr: npt.NDArray = -self._gravitational_acceleration * self.get_density(pressure)

        return dPdr

    def get_radii_from_pressure(self, pressure: FloatOrArray) -> npt.NDArray:
        r"""Computes radii from pressure:

        .. math::

            P(r) = \int \frac{dP}{dr} dr = \int -g \rho_s \exp(P/K_S) dr

        And apply the boundary condition :math:`P=0` at :math:`r=r_s` to get:

        .. math::

            r(P) = \frac{K_s \left( \exp(-P/K_S)-1 \right)}{\rho_s g} + r_s

        where :math:`r` is radius, :math:`K_S` is adiabatic bulk modulus, :math:`P` is pressure,
        :math:`\rho_s` is surface density, :math:`g` is gravitational acceleration, and
        :math:`r_s` is surface radius.

        Args:
            pressure: Pressure

        Returns:
            Radii
        """
        radii: npt.NDArray = (
            self._adiabatic_bulk_modulus
            * (np.exp(-pressure / self._adiabatic_bulk_modulus) - 1)
            / (self._surface_density * self._gravitational_acceleration)
            + self._outer_boundary
        )

        return radii
