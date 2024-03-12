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
from dataclasses import dataclass

import numpy as np

from spider.parser import Parameters, _MeshSettings
from spider.utilities import is_monotonic_increasing

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class FixedMesh:
    """A fixed mesh

    Some quantities are column vectors (2-D arrays) to allow vectorised calculations
    (see scipy.integrate.solve_ivp).

    Args:
        radii: Radii of the mesh
        mixing_length_profile: Profile of the mixing length. Can be nearest_boundary or
            constant.
        outer_boundary: Outer boundary for computing depth below the surface
        inner_boundary: Inner boundary for computing height above the base

    Attributes:
        radii: Radii of the mesh
        mixing_length_profile: Profile of the mixing length
        outer_boundary: Outer boundary for computing depth below the surface. Defaults to None, in
            which case the outermost radius is used.
        inner_boundary: Inner boundary for computing height above the base. Defaults to None, in
            which case the innermost radius is used.
        delta_radii: Delta radii
        depth: Depth below the outer boundary
        height: Height above the inner boundary
        number_of_nodes: Number of nodes
        area: Surface area
        volume: Volume of the spherical shells defined between neighbouring radii
        mixing_length: Mixing length
        mixing_length_squared: Mixing length squared
        mixing_length_cubed: Mixing length cubed
        total_volume: Total volume
    """

    radii: np.ndarray
    mixing_length_profile: str
    outer_boundary: float | None = None
    inner_boundary: float | None = None

    def __post_init__(self):
        if not is_monotonic_increasing(self.radii):
            msg: str = "Mesh must be monotonically increasing"
            logger.error(msg)
            raise ValueError(msg)
        if self.outer_boundary is None:
            self.outer_boundary = self.radii[-1]
        if self.inner_boundary is None:
            self.inner_boundary = self.radii[0]
        self.delta_radii: np.ndarray = np.diff(self.radii)
        self.depth: np.ndarray = self.outer_boundary - self.radii
        self.height: np.ndarray = self.radii - self.inner_boundary
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
            logger.debug("Set mixing length profile to nearest boundary")
            self.mixing_length = np.minimum(
                self.outer_boundary - self.radii, self.radii - self.inner_boundary
            )
        elif self.mixing_length_profile == "constant":
            logger.debug("Set mixing length profile to constant")
            assert self.outer_boundary is not None
            assert self.inner_boundary is not None
            self.mixing_length = (
                np.ones(self.radii.size) * 0.25 * (self.outer_boundary - self.inner_boundary)
            )
        else:
            msg: str = f"Mixing length profile = {self.mixing_length_profile} is unknown"
            raise ValueError(msg)

        self.mixing_length = self.mixing_length.reshape(-1, 1)  # 2-D


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
        self.basic: FixedMesh = FixedMesh(basic_coordinates, self.settings.mixing_length_profile)
        staggered_coordinates: np.ndarray = self.basic.radii[:-1] + 0.5 * self.basic.delta_radii
        self.staggered: FixedMesh = FixedMesh(
            staggered_coordinates,
            self.settings.mixing_length_profile,
            self.basic.outer_boundary,
            self.basic.inner_boundary,
        )
        self._d_dr_transform: np.ndarray = self.d_dr_transform_matrix()
        self._quantity_transform: np.ndarray = self.quantity_transform_matrix()

    def get_constant_spacing(self) -> np.ndarray:
        """Constant radius spacing across the mantle

        Returns:
            Radii with constant spacing
        """
        radii: np.ndarray = np.linspace(
            self.settings.inner_radius, self.settings.outer_radius, self.settings.number_of_nodes
        )
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


class AdamsWilliamsonEOS:
    """Adams-Williamson equation of state

    Args:
        parameters: Parameters
        mesh: A fixed mesh

    Attributes:
        TODO
    """

    def __init__(self, parameters: Parameters, mesh_: FixedMesh):
        self.settings: _MeshSettings = parameters.mesh
        self.mesh: FixedMesh = mesh_

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
            + self.pressure()
            * self.settings.adams_williamson_beta
            / self.settings.gravitational_acceleration
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
            -2 / self.settings.adams_williamson_beta**3
            - np.square(self.mesh.radii) / self.settings.adams_williamson_beta
            - 2 * self.mesh.radii / np.square(self.settings.adams_williamson_beta)
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
        factor: float = (
            self.settings.adams_williamson_surface_density
            * self.settings.gravitational_acceleration
            / self.settings.adams_williamson_beta
        )
        pressure: np.ndarray = factor * (
            np.exp(
                self.settings.adams_williamson_beta * (self.mesh.outer_boundary - self.mesh.radii)
            )
            - 1
        )

        return pressure

    def pressure_gradient(self) -> np.ndarray:
        """Pressure gradient

        Returns:
            Pressure gradient
        """
        dPdr: np.ndarray = -self.settings.gravitational_acceleration * self.density()

        return dPdr
