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
"""Mesh"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from spider.parser import Parameters, _MeshSettings
from spider.utilities import is_monotonic_increasing

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class FixedMesh:
    """A fixed mesh

    Args:
        settings: Mesh settings
        radii: Radii of the mesh
        outer_boundary: Outer boundary for computing depth below the surface
        inner_boundary: Inner boundary for computing height above the base

    Attributes:
        settings: Mesh settings
        radii: Radii of the mesh
        outer_boundary: Outer boundary for computing depth below the surface. Defaults to None, in
            which case the outermost radius is used.
        inner_boundary: Inner boundary for computing height above the base. Defaults to None, in
            which case the innermost radius is used.
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

    settings: _MeshSettings
    radii: np.ndarray
    outer_boundary: float | None = None
    inner_boundary: float | None = None
    _eos: _AdamsWilliamsonEOS = field(init=False)

    def __post_init__(self):
        if not is_monotonic_increasing(self.radii):
            msg: str = "Mesh must be monotonically increasing"
            logger.error(msg)
            raise ValueError(msg)
        if self.outer_boundary is None:
            self.outer_boundary = np.max(self.radii)
        if self.inner_boundary is None:
            self.inner_boundary = np.min(self.radii)
        self._eos: _AdamsWilliamsonEOS = _AdamsWilliamsonEOS(self.settings, self)

    @cached_property
    def area(self) -> np.ndarray:
        """Includes 4*pi factor unlike C-version of SPIDER."""
        return 4 * np.pi * np.square(self.radii)

    @cached_property
    def delta_radii(self) -> np.ndarray:
        return np.diff(self.radii, axis=0)

    @cached_property
    def density(self) -> np.ndarray:
        return self._eos.density

    @cached_property
    def depth(self) -> np.ndarray:
        return self.outer_boundary - self.radii

    @cached_property
    def height(self) -> np.ndarray:
        return self.radii - self.inner_boundary

    @cached_property
    def _mesh_cubed(self) -> np.ndarray:
        return np.power(self.radii, 3)

    @cached_property
    def mixing_length(self) -> np.ndarray:
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
    def mixing_length_cubed(self) -> np.ndarray:
        return np.power(self.mixing_length, 3)

    @cached_property
    def mixing_length_squared(self) -> np.ndarray:
        return np.square(self.mixing_length)

    @cached_property
    def number_of_nodes(self) -> int:
        return self.radii.size

    @cached_property
    def pressure(self) -> np.ndarray:
        return self._eos.pressure

    @cached_property
    def pressure_gradient(self) -> np.ndarray:
        return self._eos.pressure_gradient

    @cached_property
    def volume(self) -> np.ndarray:
        volume: np.ndarray = 4 / 3 * np.pi * (self._mesh_cubed[1:] - self._mesh_cubed[:-1])

        return volume

    @cached_property
    def total_volume(self) -> np.ndarray:
        return 4 / 3 * np.pi * (self._mesh_cubed[-1] - self._mesh_cubed[0])


@dataclass
class Mesh:
    """A staggered mesh.

    The basic mesh is used for the flux calculations and the staggered mesh is used for the volume
    calculations.

    Args:
        parameters: Parameters
    """

    _parameters: Parameters

    def __post_init__(self):
        self.settings: _MeshSettings = self._parameters.mesh
        basic_coordinates: np.ndarray = self.get_constant_spacing()
        self.basic: FixedMesh = FixedMesh(self.settings, basic_coordinates)
        staggered_coordinates: np.ndarray = self.basic.radii[:-1] + 0.5 * self.basic.delta_radii
        self.staggered: FixedMesh = FixedMesh(
            self.settings,
            staggered_coordinates,
            self.basic.outer_boundary,
            self.basic.inner_boundary,
        )
        self._d_dr_transform: np.ndarray = self.d_dr_transform_matrix()
        logger.warning(self._d_dr_transform)
        sys.exit(1)
        self._quantity_transform: np.ndarray = self.quantity_transform_matrix()

    def get_constant_spacing(self) -> np.ndarray:
        """Constant radius spacing across the mantle

        Returns:
            Radii with constant spacing as a column vector
        """
        radii: np.ndarray = np.linspace(
            self.settings.inner_radius, self.settings.outer_radius, self.settings.number_of_nodes
        )
        radii = np.atleast_2d(radii).T

        return radii

    def d_dr_transform_matrix(self) -> np.ndarray:
        """Transform matrix for determining d/dr of a staggered quantity on the basic mesh.

        Returns:
            The transform matrix
        """
        transform: np.ndarray = np.zeros(
            (self.basic.number_of_nodes, self.staggered.number_of_nodes)
        )
        # FIXME: Needed to flatten to return to 1-D array as before
        transform[1:-1, :-1] += np.diag(-1 / self.staggered.delta_radii.flatten())  # k=0 diagonal
        transform[1:-1:, 1:] += np.diag(1 / self.staggered.delta_radii.flatten())  # k=1 diagonal
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


class _AdamsWilliamsonEOS:
    """Adams-Williamson equation of state

    Args:
        settings: Mesh settings
        mesh: A fixed mesh

    Attributes:
        TODO
    """

    def __init__(self, settings: _MeshSettings, mesh: FixedMesh):
        self.settings: _MeshSettings = settings
        self.mesh: FixedMesh = mesh

    @cached_property
    def density(self) -> np.ndarray:
        """Density

        TODO: Convert math to rst

        Adams-Williamson density is a simple function of depth (radius)
        Sketch derivation:
            dP/dr = dP/drho * drho/dr = -rho g
            dP/drho sim (dP/drho)_s (adiabatic)
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
            self.settings.adams_williamson_surface_density
            + self.pressure
            * self.settings.adams_williamson_beta
            / self.settings.gravitational_acceleration
        )

        return density

    # @cached_property
    # def mass_element(self) -> np.ndarray:
    #     """Mass element"""
    #     # TODO: Check because C Spider does not include 4*pi scaling
    #     mass_element: np.ndarray = self.mesh.area * self.density

    #     return mass_element

    # @cached_property
    # def mass_within_radius(self) -> np.ndarray:
    #     """Mass contained within radii

    #     Returns:
    #         Mass within a radius
    #     """
    #     mass: np.ndarray = (
    #         -2 / self.settings.adams_williamson_beta**3
    #         - np.square(self.mesh.radii) / self.settings.adams_williamson_beta
    #         - 2 * self.mesh.radii / np.square(self.settings.adams_williamson_beta)
    #     )
    #     # TODO: Check because C Spider does not include 4*pi scaling
    #     mass *= 4 * np.pi * self.density

    #     return mass

    # @cached_property
    # def mass_within_shell(self) -> np.ndarray:
    #     """Mass within a spherical shell

    #     Returns:
    #         Mass within a spherical shell
    #     """
    #     # TODO: Check because C Spider does not include 4*pi scaling
    #     # From outer radius to inner
    #     mass_within_radius: np.ndarray = np.flip(self.mass_within_radius)
    #     # Return same order as radii
    #     delta_mass: np.ndarray = np.flip(mass_within_radius[:-1] - mass_within_radius[1:])

    #     return delta_mass

    @cached_property
    def pressure(self) -> np.ndarray:
        factor: float = (
            self.settings.adams_williamson_surface_density
            * self.settings.gravitational_acceleration
            / self.settings.adams_williamson_beta
        )
        pressure: np.ndarray = factor * (
            np.exp(self.settings.adams_williamson_beta * (self.mesh.outer_boundary - self.radii))
            - 1
        )
        logger.debug("eos pressure = %s", pressure)

        return pressure

    @cached_property
    def pressure_gradient(self) -> np.ndarray:
        dPdr: np.ndarray = -self.settings.gravitational_acceleration * self.density
        logger.debug("eos dPdr = %s", dPdr)

        return dPdr

    @cached_property
    def radii(self) -> np.ndarray:
        return self.mesh.radii
