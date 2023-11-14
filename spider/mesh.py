"""Mesh

See the LICENSE file for licensing information.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Union

import numpy as np

from spider.scalings import Scalings

logger: logging.Logger = logging.getLogger(__name__)


def is_monotonic_increasing(some_array: np.ndarray) -> np.bool_:
    """Returns True if an array is monotonically increasing, otherwise returns False."""
    return np.all(np.diff(some_array) > 0)


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
class StaggeredMesh:
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

    radii: np.ndarray
    scalings: Scalings
    basic: _FixedMesh = field(init=False)
    staggered: _FixedMesh = field(init=False)
    _d_dr_transform: np.ndarray = field(init=False)
    _quantity_transform: np.ndarray = field(init=False)

    def __post_init__(self):
        self.radii /= self.scalings.radius
        self.basic = _FixedMesh(self.radii)
        staggered_coordinates: np.ndarray = self.basic.radii[:-1] + 0.5 * self.basic.delta_radii
        self.staggered = _FixedMesh(staggered_coordinates)
        self._d_dr_transform = self.d_dr_transform_matrix()
        self._quantity_transform = self.quantity_transform_matrix()

    @property
    def inner_radius(self) -> float:
        """Inner radius is given by the basic mesh."""
        return self.basic.inner_radius

    @property
    def outer_radius(self) -> float:
        """Outer radius is given by the basic mesh."""
        return self.basic.outer_radius

    @property
    def total_volume(self) -> float:
        """Total volume is given by the basic mesh."""
        return self.basic.total_volume

    @classmethod
    def uniform_radii(
        cls,
        scalings: Scalings,
        /,
        inner_radius: Union[str, float],
        outer_radius: Union[str, float],
        number_of_nodes: Union[str, int],
        **kwargs,
    ) -> StaggeredMesh:
        """Uniform mesh

        The arguments must allow a string to enable configuration data to be passed in.

        Args:
            scalings: Scalings for the numerical problem
            inner_radius: Inner radius of the basic mesh
            outer_radius: Outer radius of the basic mesh
            number_of_coordinates: Number of basic coordinates
            **kwargs: Catches unused keyword arguments.

        Returns:
            The staggered mesh
        """
        del kwargs
        radii: np.ndarray = np.linspace(
            float(inner_radius), float(outer_radius), int(number_of_nodes)
        )

        return cls(radii, scalings)

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


# PREVIOUS BELOW
# # ====================================================================
# class StructuredGrid(object):

#     """Structured grid of points"""

#     def __init__(self, data_d, const_o):
#         """data_d:  data dictionary for mesh construction"""
#         self.data_d = data_d
#         self.const_o = const_o  # constants
#         # must _set_radius__basic_nodes() first
#         self._set_radius_basic_nodes()  # sets self._radius_basic
#         self._set_geometry()
#         self._set_pressure()
#         self._set_derivative()

#     # -----------------------------
#     # attributes that can be called
#     # -----------------------------

#     def d_dr_at_b(self, YY_a):
#         return self._d_dr_o(YY_a)

#     def depth_basic(self):
#         return self._calc_depth_basic()

#     def depth_staggered(self):
#         return self._calc_depth_staggered()

#     def dPdr_basic(self):
#         return self._calc_dPdr_basic()

#     def dPdr_staggered(self):
#         return self._calc_dPdr_staggered()

#     def dr_basic(self):
#         return self._calc_dr_basic()

#     def dr_staggered(self):
#         return self._calc_dr_staggered()

#     def height_basic(self):
#         return self._calc_height_basic()

#     def height_staggered(self):
#         return self._calc_height_staggered()

#     def mixing_length_basic(self):
#         return self._calc_mixing_length_basic()

#     def num_points_basic(self):
#         return self._calc_num_points_basic()

#     def num_points_staggered(self):
#         return self._calc_num_points_staggered()

#     def pressure_all(self):
#         return self._calc_pressure_all()

#     def pressure_basic(self):
#         return self._calc_pressure_basic()

#     def pressure_staggered(self):
#         return self._calc_pressure_staggered()

#     def qty_at_b(self, YY_a):
#         return self._d_dr_o.qty_at_b(YY_a)

#     def radius_basic(self):
#         return self._radius_basic

#     def radius_staggered(self):
#         return self._calc_radius_staggered()

#     # --------------------
#     # calculate properties
#     # --------------------

#     def _calc_depth_basic(self):
#         """Depth for basic nodes"""
#         return self._calc_radius_max() - self._calc_radius_basic()

#     def _calc_depth_staggered(self):
#         """Depth for staggered nodes"""
#         return self._calc_radius_max() - self._calc_radius_staggered()

#     def _calc_dr_basic(self):
#         """dr for basic nodes"""
#         return np.diff(self._calc_radius_basic())

#     def _calc_dr_staggered(self):
#         """dr for staggered nodes"""
#         return np.diff(self._calc_radius_staggered())

#     def _calc_height_basic(self):
#         """Calculate height for basic nodes"""
#         return self._calc_radius_basic() - self._calc_radius_min()

#     def _calc_height_staggered(self):
#         """Calculate height for staggered nodes"""
#         return self._calc_radius_staggered() - self._calc_radius_min()

#     def _calc_mixing_length_basic(self):
#         """Calculate mixing length for basic nodes"""
#         return np.minimum(self._calc_depth_basic(), self._calc_height_basic())

#     def _calc_num_points_basic(self):
#         """Calculate number of points for basic nodes"""
#         return len(self._calc_radius_basic())

#     def _calc_num_points_staggered(self):
#         """Calculate number of points for staggered nodes"""
#         return len(self._calc_radius_staggered())

#     def _calc_pressure_all(self):
#         P_b = self._calc_pressure_basic()
#         P_s = self._calc_pressure_staggered()
#         P_a = np.concatenate((P_b, P_s))
#         P_a.sort()
#         return P_a

#     def _calc_radius_basic(self):
#         """Return radius of basic nodes"""
#         return self._radius_basic

#     def _calc_radius_max(self):
#         """Maximum radius of mesh"""
#         return self._calc_radius_basic().max()

#     def _calc_radius_min(self):
#         """Minimum radius of mesh"""
#         return self._calc_radius_basic().min()

#     def _calc_radius_staggered(self):
#         """Calculate radius of staggered nodes"""
#         return self._calc_radius_basic()[:-1] + 0.5 * self._calc_dr_basic()

#     # -------------------
#     # other set functions
#     # -------------------

#     def _set_derivative(self):
#         """order of spatial derivative"""
#         # TODO: this was originally in the input cfg file, but now we
#         # are solving for the derivative quantity (dS/dr) we only
#         # use this derivative operator to compute derivatives of the
#         # melting curve quantities.  These are time-independent, and
#         # a higher order method should make no difference.
#         set_deriv = 1
#         if set_deriv == 1:
#             d_dr_o = d_dr_linear
#         elif set_deriv == 2:
#             d_dr_o = d_dr_quadratic
#         else:
#             msg = "set_derivative={0} is undefined".format(set_deriv)
#             logging.critical(msg)
#             sys.exit(1)
#         setattr(self, "_d_dr_o", d_dr_o(self))

#     def _set_geometry(self):
#         """spherical geometry factors for FVM"""
#         radius_b = self._calc_radius_basic()
#         radius_s = self._calc_radius_staggered()
#         self.data_d["geom_area_b"] = radius_b**2.0
#         # control volume
#         rcub = radius_b**3.0
#         self.data_d["geom_vol_s"] = rcub[None:-1] - rcub[1:None]
#         self.data_d["geom_vol_s"] *= 1.0 / 3.0
#         geom_vol = self.data_d["geom_vol_s"]

#     def _set_pressure(self):
#         """pressure from an eos"""
#         set_pressure = self.data_d["set_pressure"]
#         # Adams-Williamson equation of state
#         if set_pressure == 2:
#             pres_o = adams_williamson_eos(self)
#         # linear
#         elif set_pressure == 3:
#             pres_o = linear_eos(self)
#         # quadratic fit to PREM from Monteux et al. (2016)
#         elif set_pressure == 4:
#             pres_o = prem_eos(self)
#         else:
#             msg = "set_pressure={0} is undefined".format(set_pressure)
#             logging.critical(msg)

#         self._calc_pressure = pres_o._calc_pressure
#         self._calc_pressure_basic = pres_o.pressure_basic
#         self._calc_pressure_staggered = pres_o.pressure_staggered
#         self._calc_dPdr_basic = pres_o.dPdr_basic
#         self._calc_dPdr_staggered = pres_o.dPdr_staggered

#     # ----------------------------------------
#     # radial mesh construction for basic nodes
#     # ----------------------------------------

#     def _set_radius_basic_nodes(self):
#         """Set radius mesh for basic nodes"""
#         set_mesh = self.data_d["set_mesh"]
#         # read mesh from file
#         if set_mesh == 1:
#             radius_basic = self._calc_radius_basic_nodes_file()
#         # constant mesh spacing
#         elif set_mesh == 2:
#             radius_basic = self._calc_radius_basic_nodes_even()
#         # geometric mesh spacing
#         elif set_mesh == 3:
#             radius_basic = self._calc_radius_basic_nodes_geometric()
#         elif set_mesh == 4:
#             # geometric mesh spacing about a pin radius
#             radius_basic = self._calc_radius_basic_nodes_geometric_pin()
#         else:
#             msg = "set_mesh={0} is undefined".format(set_mesh)
#             logging.critical(msg)
#         # radius always from outer to inner (required for pressure
#         # interpolation)
#         radius_basic.sort()
#         self._radius_basic = radius_basic[::-1]  # outer to inner
#         logging.info("radius_basic= {0}".format(self._radius_basic))
#         logging.info("size(radius_basic) = {0}".format(np.size(self._radius_basic)))

#     # set_mesh == 1
#     def _calc_radius_basic_nodes_file(self):
#         """Read radial grid points from file for basic nodes"""
#         mesh_dir = self.data_d["mesh_dir"]
#         mesh_file = self.data_d["mesh_file"]
#         infile = os.path.join(mesh_dir, mesh_file)
#         try:
#             return load_column_data(infile)
#         except Exception:
#             msg = "cannot read file: {0}".format(infile)
#             logging.critical(msg)
#             sys.exit(1)

#     # set_mesh == 2
#     def _calc_radius_basic_nodes_even(self):
#         """Create evenly spaced grid for basic nodes"""
#         dd = self.data_d
#         return np.linspace(dd["core_radius"], 1.0, dd["num_points"])

#     # set_mesh == 3
#     def _calc_radius_basic_nodes_geometric(self):
#         """Create geometric grid for basic nodes"""
#         dd = self.data_d
#         maxdepth = 1.0 - dd["core_radius"]
#         msg = "geometric grid, number of points: {0}"
#         TOP = dd["set_top_refine"]
#         BOT = dd["set_bot_refine"]
#         MID = dd["set_mid_refine"]
#         # upper thermal boundary layer refinement
#         if TOP:
#             rad_top = self._calc_radius_top_refine()
#             if not BOT and not MID:
#                 return rad_top
#         # lower thermal boundary layer refinement
#         if BOT:
#             rad_bot = self._calc_radius_bot_refine()
#             if not TOP and not MID:
#                 return rad_bot
#         # middle refinement for first crystal formation
#         if MID:
#             rad_mid = self._calc_radius_mid_refine()
#             if not TOP and not BOT:
#                 return rad_mid

#         # merge arrays
#         if BOT and MID and not TOP:
#             pin = dd["radius_pin"]
#             lim1 = 0.5 * (pin - dd["core_radius"]) + dd["core_radius"]
#             cond1 = rad_bot < lim1
#             cond2 = rad_mid >= lim1
#             rad_bot = rad_bot[cond1]
#             rad_mid = rad_mid[cond2]
#             rad_all = np.concatenate((rad_mid, rad_bot))

#         if TOP and BOT and not MID:
#             # upper and lower thermal boundary layer refinement
#             cond1 = rad_top > dd["core_radius"] + 0.5 * maxdepth
#             cond2 = rad_bot < dd["core_radius"] + 0.5 * maxdepth
#             rad_top = rad_top[cond1]
#             rad_top = np.append(rad_top, 1.0 - 0.5 * maxdepth)
#             rad_bot = rad_bot[cond2]
#             rad_all = np.concatenate((rad_top, rad_bot))

#         msg = "geometric grid, number of points: {0}".format(len(rad_all))
#         logging.info(msg)
#         return rad_all

#     def _calc_radius_mid_refine(self):
#         dd = self.data_d
#         depth = 1.0 - dd["core_radius"]
#         dr_min = dd["dr_mid_min"]
#         dist_a = self._calc_radius_geometric_distance(dr_min)
#         core_radius = dd["core_radius"]
#         pin = dd["radius_pin"]
#         # coarsen away from pin point
#         rad_bot = pin - dist_a
#         rad_bot = rad_bot[rad_bot > core_radius]
#         rad_bot = np.append(rad_bot, core_radius)
#         rad_top = pin + dist_a
#         rad_top = rad_top[rad_top < 1.0]
#         rad_top = np.append(rad_top, 1.0)
#         rad_mid = np.concatenate((rad_bot[::-1], rad_top[1:None]))
#         rad_mid = rad_mid[::-1]  # outer to inner
#         return rad_mid

#     def _calc_radius_top_refine(self):
#         dd = self.data_d
#         dr_min = dd["dr_min"]
#         dist_a = self._calc_radius_geometric_distance(dr_min)
#         rad_top = 1.0 - dist_a
#         return rad_top

#     def _calc_radius_bot_refine(self):
#         dd = self.data_d
#         dr_min = dd["dr_min"]
#         dist_a = self._calc_radius_geometric_distance(dr_min)
#         rad_bot = dd["core_radius"] + dist_a
#         rad_bot = rad_bot[::-1]  # outer to inner
#         return rad_bot

#     def _calc_radius_geometric_distance(self, dr_min):
#         """Position of nodes"""
#         dd = self.data_d
#         depth = 1.0 - dd["core_radius"]
#         dist = 0.0
#         dist_l = [dist]

#         # algorithm that avoids OverflowError
#         dr_prev = dr_min / dd["geom_factor"]
#         while dist < depth:
#             dr = dr_prev * dd["geom_factor"]
#             if dr > dd["dr_max"]:
#                 dr = dd["dr_max"]
#             dr_prev = dr  # update for next iteration
#             dist_up = dist + dr
#             dist_l.append(dist_up)
#             dist = dist_up

#         # ensure core_radius is last point
#         if dist_l[-1] < depth:
#             dist_l.append(depth)
#         else:
#             dist_l[-1] = depth
#         return np.array(dist_l)


# # ====================================================================
# class StructuredGridStatic(StructuredGrid):

#     """A mesh that does not change with time"""

#     def __init__(self, *args, **kwargs):
#         super(StructuredGridStatic, self).__init__(*args, **kwargs)
#         # preset all these attributes (do not change with time)
#         self._depth_basic = self._calc_depth_basic()
#         self._depth_staggered = self._calc_depth_staggered()
#         self._dPdr_basic = self._calc_dPdr_basic()
#         self._dPdr_staggered = self._calc_dPdr_staggered()
#         self._dr_basic = self._calc_dr_basic()
#         self._dr_staggered = self._calc_dr_staggered()
#         self._height_basic = self._calc_height_basic()
#         self._height_staggered = self._calc_height_staggered()
#         self._mixing_length_basic = self._calc_mixing_length_basic()
#         self._num_points_basic = self._calc_num_points_basic()
#         self._num_points_staggered = self._calc_num_points_staggered()
#         self._pressure_all = self._calc_pressure_all()
#         self._pressure_basic = self._calc_pressure_basic()
#         self._pressure_staggered = self._calc_pressure_staggered()
#         self._radius_basic = self._calc_radius_basic()
#         self._radius_staggered = self._calc_radius_staggered()
#         # check monotonic behaviour
#         util.mono_check(self._depth_basic, check="increasing")
#         util.mono_check(self._depth_staggered, check="increasing")
#         util.mono_check(self._pressure_basic, check="increasing")
#         util.mono_check(self._pressure_staggered, check="increasing")
#         util.mono_check(self._radius_basic, check="decreasing")
#         util.mono_check(self._radius_staggered, check="decreasing")

#     def depth_basic(self):
#         return self._depth_basic

#     def depth_staggered(self):
#         return self._depth_staggered

#     def dPdr_basic(self):
#         return self._dPdr_basic

#     def dPdr_staggered(self):
#         return self._dPdr_staggered

#     # commented out otherwise inheritance breaks
#     # def dr_basic( self ):
#     #    return self._dr_basic

#     # def dr_staggered( self ):
#     #    return self._dr_staggered

#     def height_basic(self):
#         return self._height_basic

#     def height_staggered(self):
#         return self._height_staggered

#     def mixing_length_basic(self):
#         return self._mixing_length_basic

#     # commented out otherwise inheritance breaks
#     # def num_points_basic( self ):
#     #    return self._num_points_basic

#     # def num_points_staggered( self ):
#     #    return self._num_points_staggered

#     def pressure_all(self):
#         return self._pressure_all

#     def pressure_basic(self):
#         return self._pressure_basic

#     def pressure_staggered(self):
#         return self._pressure_staggered

#     def radius_basic(self):
#         return self._radius_basic

#     def radius_staggered(self):
#         return self._radius_staggered


# # ====================================================================
# class linear_eos(object):

#     """Linearised EOS"""

#     def __init__(self, mesh_o):
#         self.mesh_o = mesh_o

#     def _calc_dPdr(self, depth):
#         """dPdr"""
#         cc = self.mesh_o.const_o.data_d
#         dd = self.mesh_o.data_d
#         dp_a = dd["rho_s"] * cc["gravity"]
#         dp_a *= np.ones(depth.size)
#         return dp_a

#     def dPdr_basic(self):
#         depth_b = self.mesh_o._calc_depth_basic()
#         return self._calc_dPdr(depth_b)

#     def dPdr_staggered(self):
#         depth_s = self.mesh_o._calc_depth_staggered()
#         return self._calc_dPdr(depth_s)

#     def _calc_pressure(self, depth):
#         """Pressure"""
#         cc = self.mesh_o.const_o.data_d
#         dd = self.mesh_o.data_d
#         p_a = -dd["rho_s"] * cc["gravity"] * depth
#         return p_a

#     def pressure_basic(self):
#         depth_b = self.mesh_o._calc_depth_basic()
#         return self._calc_pressure(depth_b)

#     def pressure_staggered(self):
#         depth_s = self.mesh_o._calc_depth_staggered()
#         return self._calc_pressure(depth_s)


# # ====================================================================
# class prem_eos(object):

#     """Quadratic fit to PREM model of pressure, from
#     Monteux et al. (2016)"""

#     def __init__(self, mesh_o):
#         self.mesh_o = mesh_o
#         self.AA = 4.0074e11
#         self.BB = -91862
#         self.CC = 0.0045483

#     def _calc_dPdr(self, radius):
#         """dPdr"""
#         cc = self.mesh_o.const_o.data_d
#         BB = self.BB
#         CC = 2.0 * self.CC * radius * cc["radius0"]
#         dPdr_a = cc["radius0"] / cc["pressure0"] * (BB + CC)
#         return dPdr_a

#     def dPdr_basic(self):
#         radius_b = self.mesh_o._calc_radius_basic()
#         return self._calc_dPdr(radius_b)

#     def dPdr_staggered(self):
#         radius_s = self.mesh_o._calc_radius_staggered()
#         return self._calc_dPdr(radius_s)

#     def _calc_pressure(self, radius):
#         """Pressure"""
#         cc = self.mesh_o.const_o.data_d
#         AA = self.AA
#         BB = self.BB * (radius * cc["radius0"])
#         CC = self.CC * (radius * cc["radius0"]) ** 2.0
#         P_a = AA + BB + CC  # this is dimensional P
#         P_a /= cc["pressure0"]  # non-dim
#         P_a -= P_a[0]  # outer pressure must be zero (so shift here)
#         return P_a

#     def pressure_basic(self):
#         radius_b = self.mesh_o._calc_radius_basic()
#         return self._calc_pressure(radius_b)

#     def pressure_staggered(self):
#         radius_s = self.mesh_o._calc_radius_staggered()
#         return self._calc_pressure(radius_s)


# # ====================================================================
# class adams_williamson_eos(object):

#     """Adams-Williamson EOS"""

#     def __init__(self, mesh_o):
#         self.mesh_o = mesh_o

#     def _calc_dPdr(self, depth):
#         """dPdr"""
#         cc = self.mesh_o.const_o.data_d
#         dd = self.mesh_o.data_d
#         dp_a = dd["rho_s"] * cc["gravity"]
#         dp_a *= np.exp(dd["beta"] * depth)
#         return dp_a

#     def dPdr_basic(self):
#         depth_b = self.mesh_o._calc_depth_basic()
#         return self._calc_dPdr(depth_b)

#     def dPdr_staggered(self):
#         depth_s = self.mesh_o._calc_depth_staggered()
#         return self._calc_dPdr(depth_s)

#     def _calc_pressure(self, depth):
#         """Pressure"""
#         cc = self.mesh_o.const_o.data_d
#         dd = self.mesh_o.data_d
#         p_a = -dd["rho_s"] * cc["gravity"] / dd["beta"]
#         p_a *= np.exp(dd["beta"] * depth) - 1
#         return p_a

#     def pressure_basic(self):
#         depth_b = self.mesh_o._calc_depth_basic()
#         return self._calc_pressure(depth_b)

#     def pressure_staggered(self):
#         depth_s = self.mesh_o._calc_depth_staggered()
#         return self._calc_pressure(depth_s)


# # ====================================================================
# # Note about numerical derivative estimators:
# # radius arrays are always ordered from surface to CMB, therefore:
# # deriv is POSITIVE if
# #    - quantity decreases with decreasing radius
# # deriv is NEGATIVE if:
# #    - quantity increases with decreasing radius
# # ====================================================================
# class d_dr_linear(object):

#     """Compute radial derivative of a quantity at the basic nodes
#     using values at staggered nodes"""

#     # basically assumes that the derivative between neighbouring
#     # staggered nodes is constant.  If the mesh has constant
#     # spacing then this is formally 2nd order accurate since the
#     # derivative estimate coincides with the basic node.  But for
#     # an uneven mesh this is only approximate.  A 2nd order
#     # method for an uneven mesh is given by d_dr_quadratic()

#     def __init__(self, mesh_o):
#         dr_s = mesh_o.dr_staggered()
#         rad_b = mesh_o.radius_basic()
#         rad_s = mesh_o._calc_radius_staggered()
#         num_staggered = mesh_o.num_points_staggered()
#         num_basic = mesh_o.num_points_basic()
#         # assemble coefficient matrix
#         # remember: ( rows, columns )
#         AA_a = np.zeros((num_basic, num_staggered))
#         AA_a[1:-1, :-1] += np.diag(-1.0 / dr_s)  # k=0 diagonal
#         AA_a[1:-1:, 1:] += np.diag(1.0 / dr_s)  # k=1 diagonal

#         BB_a = np.zeros((num_basic, num_staggered))
#         dh = rad_b[1:] - rad_s
#         BB_a[1:-1, :-1] += np.diag(1.0 - dh[:-1] / dr_s)
#         BB_a[1:-1:, 1:] += np.diag(dh[:-1] / dr_s)

#         self._d_dr_at_b = AA_a
#         self._qty_at_b = BB_a

#     def __call__(self, YY_a):
#         return self.d_dr_at_b(YY_a)

#     def d_dr_at_b(self, YY_a):
#         return self._d_dr_at_b.dot(YY_a)

#     def qty_at_b(self, YY_a):
#         return self._qty_at_b.dot(YY_a)


# # ====================================================================
# class d_dr_linear_stag(object):

#     """Compute radial derivative of a quantity at the staggered nodes
#     using values at staggered nodes.  2nd order accurate central
#     for interior points and 2nd order accurate forward and backward
#     for first and last point, respectively"""

#     def __init__(self, mesh_o):
#         dr = mesh_o.dr_staggered()
#         num_staggered = mesh_o.num_points_staggered()

#         # assemble coefficient matrix
#         # remember: ( rows, columns )
#         # central difference
#         AA_a = np.zeros((num_staggered, num_staggered))
#         AA_a[1:-1, 0:-2] = np.diag(-1.0 / dr[1:])  # k=-1 diagonal
#         AA_a[1:-1, 2:] += np.diag(1.0 / dr[1:])  # k=1 diagonal
#         # forward difference
#         AA_a[0, 0] = -3.0 / dr[0]
#         AA_a[0, 1] = 4.0 / dr[0]
#         AA_a[0, 2] = -1.0 / dr[0]
#         # backward difference
#         AA_a[-1, -3] = 1.0 / dr[-1]
#         AA_a[-1, -2] = -4.0 / dr[-1]
#         AA_a[-1, -1] = 3.0 / dr[-1]

#         self.AA_a = 0.5 * AA_a

#     def __call__(self, YY_a):
#         return self.AA_a.dot(YY_a)


# # ====================================================================
# class d_dr_quadratic(object):

#     """Compute radial derivative of a quantity at the internal basic
#     nodes using a polynomial fit between neighbouring staggered
#     nodes:  Y = ax^2 + bx + c,  dY/dx = 2*a*x + b"""

#     def __init__(self, mesh_o):
#         dr_b = mesh_o.dr_basic()
#         dr_s = mesh_o.dr_staggered()
#         num_staggered = mesh_o.num_points_staggered()
#         num_basic = mesh_o.num_points_basic()

#         h1 = dr_s[:-1]
#         h2 = dr_s[1:]

#         # assemble coefficient matrix AA
#         # remember: ( rows, columns )
#         AA = np.zeros((num_basic, num_staggered))
#         AA[1:-2, :-2] += np.diag(1.0 / (h1 * (h1 + h2)))  # k=0 diagonal
#         AA[1:-2, 1:-1] += np.diag(-1.0 / (h1 * h2))  # k=1 diagonal
#         AA[1:-2, 2:] += np.diag(1.0 / (h2 * (h1 + h2)))  # k=2 diagonal
#         # last row same as penultimate (but derived using backward
#         # difference - see below for modification to BB matrix)
#         AA[-2, :] = AA[-3, :]

#         # assemble coefficient matrix BB
#         BB = np.zeros((num_basic, num_staggered))
#         BB[1:-2, :-2] += np.diag((-2.0 * h1 - h2) / (h1 * (h1 + h2)))
#         BB[1:-2, 1:-1] += np.diag((h1 + h2) / (h1 * h2))
#         BB[1:-2, 2:] += np.diag(-h1 / (h2 * (h1 + h2)))
#         # need backward difference for last point, to retain ability
#         # to dot product with half the basic node spacing when
#         # sequentially stepping forward
#         BB[-2, -3] += -h2[-1] / (h1[-1] * (h1[-1] + h2[-1]))
#         BB[-2, -2] += (h2[-1] - h1[-1]) / (h1[-1] * h2[-1])
#         BB[-2, -1] += h1[-1] / (h2[-1] * (h1[-1] + h2[-1]))

#         # assemble coefficient matrix CC
#         CC = np.zeros((num_basic, num_staggered))
#         # TODO: clean up, hacky to use h1 to give length of array
#         CC[1:-2, :-2] += np.diag(1.0 / h1 * h1)
#         # for last point
#         CC[-2, -2] = 1.0

#         # (forward) distance of basic node relative to staggered node
#         dx1 = 0.5 * dr_b[:-1]

#         # for overlapping regions can get two estimates of derivative
#         # at a given point.  Need half step + one more cell dr forward
#         # first value in this array corresponds to 3rd basic node
#         dx2 = 0.5 * dr_b[:-3] + dr_b[1:-2]

#         # compare with (final) S1a matrix in Petsc code
#         S1a = np.zeros((num_basic, num_staggered))
#         # below does a row-wise multiplication with dx1**2
#         S1a[1:-1] += AA[1:-1] * dx1[:, np.newaxis] ** 2.0
#         S1a[1:-1] += BB[1:-1] * dx1[:, np.newaxis]
#         S1a[1:-1] += CC[1:-1]
#         # print 'S1a=', S1a

#         # compare with dS1 matrix in Petsc code
#         dS1 = np.zeros((num_basic, num_staggered))
#         dS1[1:-1] += 2.0 * AA[1:-1] * dx1[:, np.newaxis] + BB[1:-1]
#         # print 'dS1=', dS1

#         # compare with (final) S2a matrix in Petsc code
#         S2a = np.zeros((num_basic, num_staggered))
#         S2a[2:-2] += AA[1:-3] * dx2[:, np.newaxis] ** 2
#         S2a[2:-2] += BB[1:-3] * dx2[:, np.newaxis]
#         S2a[2:-2] += CC[1:-3]
#         # print 'S2a=', S2a

#         # compare with dS2 matrix in Petsc code
#         dS2 = np.zeros((num_basic, num_staggered))
#         dS2[2:-2] += 2.0 * AA[1:-3] * dx2[:, np.newaxis] + BB[1:-3]
#         # print 'dS2=', dS2

#         # combine by arithmetic average
#         Sc = S1a + S2a
#         Sc[2:-2] *= 0.5
#         # print 'Sc=', Sc
#         dSc = dS1 + dS2
#         dSc[2:-2] *= 0.5
#         # print 'dSc=', dSc
#         self._qty_at_b = Sc
#         self._d_dr_at_b = dSc
#         # print '_qty_at_b=', Sc
#         # print '_d_dr_at_b=', dSc

#     def __call__(self, YY_a):
#         return self.d_dr_at_b(YY_a)

#     def d_dr_at_b(self, YY_a):
#         return self._d_dr_at_b.dot(YY_a)

#     def qty_at_b(self, YY_a):
#         return self._qty_at_b.dot(YY_a)
