#!/usr/bin/env python

"""The original SPIDER C code uses lookup data for the material properties of MgSiO3 for both a 
solid and liquid phase, and then combines these data to construct the properties in the mixed
phase region. The SPIDER C code uses entropy as an independent variable, but this Python version
requires temperature instead.

Hence this script converts the original lookup data in (pressure, entropy) coordinates to 
(pressure, temperature) coordinates. The lookup data file format is also slightly different for
this Python version.
"""
from __future__ import annotations

import logging
from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes._axes import Axes
from scipy import interpolate

from aragog import debug_logger

# Paths should not usually be hard-coded, but the data conversion is only done once
entropy_data: Path = Path("/Users/dan/Programs/pyspider/data/1TPa-dK09-elec-free/entropy")
temperature_data: Path = Path("/Users/dan/Programs/pyspider/data/1TPa-dK09-elec-free/temperature")

logger: logging.Logger = debug_logger()


@dataclass
class DataFile:
    name: str
    phase: str
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray
    _: KW_ONLY
    ylabel: str = "Entropy (J/kg/K)"
    contour_start: float | None = None
    contour_step: float | None = None
    temperature_datafile: DataFile | None = None
    log: bool = False
    _interpolate: interpolate.RectBivariateSpline = field(init=False)

    def __post_init__(self):
        self.name = self.name.capitalize()
        if self.log:
            self.Z = np.log10(self.Z)
        self.set_interpolate()
        if self.temperature_datafile is None:
            # No temperature datafile is assumed to mean it is the temperature datafile.
            self.temperature_datafile = self

    @property
    def number_coordinate_points(self) -> int:
        return self.X.shape[0]

    @property
    def number_pressure_points(self) -> int:
        return self.X.shape[1]

    @property
    def aspect(self) -> float:
        """Aspect ratio to ensure that the plot figure is square by changing the pixel size"""
        aspect: float = (self.xmax - self.xmin) / (self.ymax - self.ymin)
        return aspect

    def ev(self, pressure: np.ndarray, coordinate: np.ndarray) -> np.ndarray:
        """Evaluate the interpolation function

        Args:
            pressure: Pressure in GPa
            coordinate: Entropy in J/kg/K or temperature in K

        Returns:
            Quantity at pressure and coordinate
        """
        return self._interpolate.ev(pressure, coordinate)

    @property
    def is_contour(self) -> bool:
        return self.contour_start is not None and self.contour_step is not None

    def set_interpolate(self) -> None:
        """Sets the interpolation"""
        # The data is gridded, although due to floating point precision the values may be slightly
        # different. So force a consistent coordinate.
        pressure_unique: np.ndarray = np.unique(self.X[0, :])
        coordinate_unique: np.ndarray = np.unique(self.Y[:, 0])
        self._interpolate = interpolate.RectBivariateSpline(
            pressure_unique, coordinate_unique, self.Z.T
        )

    def to_temperature(self) -> DataFile:
        """Creates a new DataFile with temperature the independent variable"""
        assert self.temperature_datafile is not None
        temperature_at_entropy: np.ndarray = self.temperature_datafile.ev(
            self.X.flatten(), self.Y.flatten()
        )
        Y_temperature: np.ndarray = temperature_at_entropy.reshape(self.Y.shape)

        # FIXME: Add procedures also here to map temperature (and resample pressure) to a regular
        # grid.

        return DataFile(
            self.name,
            self.phase,
            self.X,
            Y_temperature,
            self.Z,
            ylabel="Temperature (K)",
            contour_start=self.contour_start,
            contour_step=self.contour_step,
            # Quantity will have Z data already converted
        )

    @property
    def contour_levels(self) -> np.ndarray:
        assert self.is_contour
        return np.arange(self.contour_start, self.zmax, self.contour_step)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """Extent of plot"""
        return (self.xmin, self.xmax, self.ymin, self.ymax)

    @property
    def title(self) -> str:
        """Plot title"""
        title: str = f"{self.name} ({self.phase})"
        if self.log:
            title += ", log10"
        return title

    @property
    def xmin(self) -> float:
        return self.X.min()

    @property
    def xmax(self) -> float:
        return self.X.max()

    @property
    def ymin(self) -> float:
        return self.Y.min()

    @property
    def ymax(self) -> float:
        return self.Y.max()

    @property
    def zmin(self) -> float:
        return self.Z.min()

    @property
    def zmax(self) -> float:
        return self.Z.max()

    @staticmethod
    def to_GPa(pressure_pascals: float) -> float:
        return pressure_pascals * 1e-9

    @classmethod
    def load_data(
        cls,
        file_path: Path,
        name: str,
        phase: str,
        number_pressure_points: int,
        number_coordinate_points: int,
        pressure_scaling: float = 1,
        coordinate_scaling: float = 1,
        quantity_scaling: float = 1,
        **kwargs,
    ) -> DataFile:
        logger.info("Loading data from %s", file_path)
        xs, ys, zs = np.loadtxt(file_path, unpack=True)
        reshape: tuple[int, int] = (number_coordinate_points, number_pressure_points)
        X: np.ndarray = xs.reshape(reshape) * cls.to_GPa(pressure_scaling)
        Y: np.ndarray = ys.reshape(reshape) * coordinate_scaling
        Z: np.ndarray = zs.reshape(reshape) * quantity_scaling

        return cls(name, phase, X, Y, Z, **kwargs)

    def plot(self, ax: Axes | None = None, show: bool = False) -> None:
        logger.info("Plotting %s (%s)", self.name, self.phase)
        if ax is None:
            fig, ax = plt.subplots()
        assert ax is not None

        im = ax.imshow(
            self.Z,
            interpolation="bilinear",
            cmap="plasma",
            origin="lower",
            extent=self.extent,
            vmax=self.zmax,
            vmin=self.zmin,
            aspect=self.aspect,
        )
        if self.is_contour:
            contour = ax.contour(
                self.X, self.Y, self.Z, levels=self.contour_levels, colors="white", linewidths=0.5
            )
            ax.clabel(contour, inline=True, fontsize=8, colors="white")
        ax.set_title(self.title)
        ax.set_xlabel("Pressure (GPa)")
        ax.set_ylabel(self.ylabel)
        plt.colorbar(im, ax=ax)
        if show:
            plt.show()


# Melt data
NUMBER_PRESSURE_POINTS: int = 2020
NUMBER_COORDINATE_POINTS: int = 95
PRESSURE_SCALING: float = 1e9  # Pa
ENTROPY_SCALING: float = 4805046.659407042  # J/kg/K

# The temperature data is used to interpolate from entropy to temperature
temperature_melt: DataFile = DataFile.load_data(
    Path(entropy_data, "temperature_melt.dat"),
    "temperature",
    "melt",
    NUMBER_PRESSURE_POINTS,
    NUMBER_COORDINATE_POINTS,
    PRESSURE_SCALING,
    ENTROPY_SCALING,
    1.0,
    contour_start=0,
    contour_step=1000,
)

adiabat_temp_grad_melt: DataFile = DataFile.load_data(
    Path(entropy_data, "adiabat_temp_grad_melt.dat"),
    "adiabatic temperature gradient",
    "melt",
    NUMBER_PRESSURE_POINTS,
    NUMBER_COORDINATE_POINTS,
    PRESSURE_SCALING,
    ENTROPY_SCALING,
    1e-9,
    log=True,
    temperature_datafile=temperature_melt,
)

density_melt: DataFile = DataFile.load_data(
    Path(entropy_data, "density_melt.dat"),
    "density",
    "melt",
    NUMBER_PRESSURE_POINTS,
    NUMBER_COORDINATE_POINTS,
    PRESSURE_SCALING,
    ENTROPY_SCALING,
    1000.0,
    contour_start=0,
    contour_step=1000,
    temperature_datafile=temperature_melt,
)

heat_capacity_melt: DataFile = DataFile.load_data(
    Path(entropy_data, "heat_capacity_melt.dat"),
    "heat capacity",
    "melt",
    NUMBER_PRESSURE_POINTS,
    NUMBER_COORDINATE_POINTS,
    PRESSURE_SCALING,
    ENTROPY_SCALING,
    ENTROPY_SCALING,
    contour_start=0,
    contour_step=100,
    temperature_datafile=temperature_melt,
)

thermal_exp_melt: DataFile = DataFile.load_data(
    Path(entropy_data, "thermal_exp_melt.dat"),
    "thermal expansion",
    "melt",
    NUMBER_PRESSURE_POINTS,
    NUMBER_COORDINATE_POINTS,
    PRESSURE_SCALING,
    ENTROPY_SCALING,
    1.0,
    contour_start=0,
    contour_step=1e-5,
    log=True,
    temperature_datafile=temperature_melt,
)

# Solid data
NUMBER_PRESSURE_POINTS: int = 2020
NUMBER_COORDINATE_POINTS: int = 125
PRESSURE_SCALING: float = 1e9  # Pa
ENTROPY_SCALING: float = 4824266.84604467  # J/kg/K

# The temperature data is used to interpolate from entropy to temperature
temperature_solid: DataFile = DataFile.load_data(
    Path(entropy_data, "temperature_solid.dat"),
    "temperature",
    "solid",
    NUMBER_PRESSURE_POINTS,
    NUMBER_COORDINATE_POINTS,
    PRESSURE_SCALING,
    ENTROPY_SCALING,
    1.0,
    contour_start=0,
    contour_step=1000,
)

adiabat_temp_grad_solid: DataFile = DataFile.load_data(
    Path(entropy_data, "adiabat_temp_grad_solid.dat"),
    "adiabatic temperature gradient",
    "solid",
    NUMBER_PRESSURE_POINTS,
    NUMBER_COORDINATE_POINTS,
    PRESSURE_SCALING,
    ENTROPY_SCALING,
    1e-09,
    log=True,
    temperature_datafile=temperature_solid,
)

density_solid: DataFile = DataFile.load_data(
    Path(entropy_data, "density_solid.dat"),
    "density",
    "solid",
    NUMBER_PRESSURE_POINTS,
    NUMBER_COORDINATE_POINTS,
    PRESSURE_SCALING,
    ENTROPY_SCALING,
    1000.0,
    contour_start=0,
    contour_step=1000,
    temperature_datafile=temperature_solid,
)

heat_capacity_solid: DataFile = DataFile.load_data(
    Path(entropy_data, "heat_capacity_solid.dat"),
    "heat capacity",
    "solid",
    NUMBER_PRESSURE_POINTS,
    NUMBER_COORDINATE_POINTS,
    PRESSURE_SCALING,
    ENTROPY_SCALING,
    ENTROPY_SCALING,
    contour_start=0,
    contour_step=50,
    temperature_datafile=temperature_solid,
)

thermal_exp_solid: DataFile = DataFile.load_data(
    Path(entropy_data, "thermal_exp_solid.dat"),
    "thermal expansion",
    "solid",
    NUMBER_PRESSURE_POINTS,
    NUMBER_COORDINATE_POINTS,
    PRESSURE_SCALING,
    ENTROPY_SCALING,
    1.0,
    contour_start=0,
    contour_step=1.0e-5,
    log=True,
    temperature_datafile=temperature_solid,
)


def plot_melt_eos_entropy() -> None:
    """Plot MgSiO3 RTPress melt EOS"""
    fig, ax = plt.subplots(3, 2, figsize=(11, 13), gridspec_kw={"hspace": 0.3})
    adiabat_temp_grad_melt.plot(ax[0][0])
    density_melt.plot(ax[0][1])
    heat_capacity_melt.plot(ax[1][0])
    temperature_melt.plot(ax[1][1])
    thermal_exp_melt.plot(ax[2][0])
    ax[2][1].remove()
    fig.suptitle("MgSiO3 melt (RTPress) EOS, entropy space")


def plot_melt_eos_temperature() -> None:
    """Plot MgSiO3 RTPress melt EOS"""
    fig, ax = plt.subplots(3, 2, figsize=(11, 13), gridspec_kw={"hspace": 0.3})
    adiabat_temp_grad_melt_temp = adiabat_temp_grad_melt.to_temperature()
    adiabat_temp_grad_melt_temp.plot(ax[0][0])
    density_melt_temp = density_melt.to_temperature()
    density_melt_temp.plot(ax[0][1])
    heat_capacity_melt_temp = heat_capacity_melt.to_temperature()
    heat_capacity_melt_temp.plot(ax[1][0])
    temperature_melt_temp = temperature_melt.to_temperature()
    temperature_melt_temp.plot(ax[1][1])
    thermal_exp_melt_temp = thermal_exp_melt.to_temperature()
    thermal_exp_melt_temp.plot(ax[2][0])
    ax[2][1].remove()
    fig.suptitle("MgSiO3 melt (RTPress) EOS, temperature space")


def plot_solid_eos_entropy() -> None:
    """Plot MgSiO3 solid EOS"""
    fig, ax = plt.subplots(3, 2, figsize=(11, 13), gridspec_kw={"hspace": 0.3})
    adiabat_temp_grad_solid.plot(ax[0][0])
    density_solid.plot(ax[0][1])
    heat_capacity_solid.plot(ax[1][0])
    temperature_solid.plot(ax[1][1])
    thermal_exp_solid.plot(ax[2][0])
    ax[2][1].remove()
    fig.suptitle("MgSiO3 solid EOS, entropy space")


def main():
    plot_melt_eos_entropy()
    plot_melt_eos_temperature()
    # plot_solid_eos_entropy()
    # plt.show()

    # new = density_melt.to_temperature()
    # new.plot()

    plt.show()


if __name__ == "__main__":
    main()


# def plot(DataFile: DataFile):
#     fig, axes = plt.subplots()

#     # Solid temperature.
#     xs, ys, zs = np.loadtxt("temperature_solid.dat", unpack=True)
#     xs *= pressure_scale
#     ys *= entropy_scale_solid
#     zs *= 1.0
#     xs2 = xs.reshape((nrows, ncols))
#     ys2 = ys.reshape((nrows, ncols))
#     grids = zs.reshape((nrows, ncols))
#     # print(np.min(ys2), np.max(ys2), np.max(grids))
#     ax1.pcolor(xs2, ys2, grids, cmap="RdBu", vmin=500, vmax=13000)
#     ax1.axis([xs.min(), xs.max(), ys.min(), ys.max()])
#     ax1.set_title("Temperature, K")
#     ax1.set_xlabel("Pressure, Pa")
#     ax1.set_ylabel("Entropy, J/kg/K")
#     # Interpolation.
#     xgs = np.unique(xs2[0, :])
#     ygs = np.unique(ys2[:, 0])
#     # NOTE: y before x.
#     fms = interpolate.RectBivariateSpline(ygs, xgs, grids)

# Read in some other quantity and interpolate to P, T, quantity from P, S, quantity.
# Uncomment for solid.
# in_quantities = {'adiabat_temp_grad': 1.0E-9,'density':1000.0,'heat_capacity': entropy_scale_solid,'thermal_exp':1.0}
# Uncomment for liquid.
# in_quantities = {'adiabat_temp_grad': 1.0E-9,'density':1000.0,'heat_capacity': entropy_scale_liquid,'thermal_exp':1.0}

# for quantity, quantity_scale in in_quantities.items():
#     filename = quantity + '_melt.dat'
#     # filename = quantity + '_solid.dat'
#     xq,yq,zq = np.loadtxt(filename,unpack=True)
#     xq *= pressure_scale
#     yq *= entropy_scale_liquid
#     zq *= quantity_scale
#     quantity_T = fms.ev(yq, xq)
#     out_data = np.column_stack((xq, quantity_T, zq))
#     np.savetxt(Path('temperature',filename), out_data)

# xs, ys = np.loadtxt("solidus_A11_H13.dat", unpack=True)
# xs *= 1000000000.0
# ys *= 4824266.84604467
# vms = fms.ev(ys, xs)
# out_data = np.column_stack((xs, vms))
# filename = "solidus_A11_H13.dat"
# np.savetxt(Path("temperature", filename), out_data)

# xm,ym,zm = np.loadtxt('temperature_melt.dat',unpack=True)
# xm *= 1000000000.0*1.0E-9
# ym *= 4805046.659407042
# zm *= 1.0
# nrows = 95
# ncols = 2020
# xm2 = xm.reshape((nrows,ncols))#[:,:290]
# ym2 = ym.reshape((nrows,ncols))#[:,:290]
# gridm = zm.reshape((nrows,ncols))#[:,:290]

# im = ax2.pcolor(xm2,ym2,gridm,cmap='RdBu', vmin=500, vmax=13000)
# ax2.axis([xm.min(), xm.max(), ym.min(), ym.max()])

# x,y = np.loadtxt('liquidus_A11_H13.dat',unpack=True)
# x *= 1000000000.0*1.0E-9
# y *= 4805046.659407042
# ax2.plot(x,y,'k-')

# ax2.set_title('Melt')
# ax2.set_xlabel('Pressure, GPa')
# ax2.set_ylabel('Entropy, J/kg/K')

# fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes([0.85,0.15,0.05,0.7])
# fig.colorbar(im, cax=cbar_ax)

# # # test interpolation
# # xgs = np.unique(xs2[0,:])
# # ygs = np.unique(ys2[:,0])
# # fms = interpolate.RectBivariateSpline(ygs,xgs,grids)
# # vms = fms.ev(ys,xs)

# # test interpolation
# xg = np.unique(xm2[0,:])
# yg = np.unique(ym2[:,0])
# fm = interpolate.RectBivariateSpline(yg,xg,gridm)
# vm = fm.ev(y,x)

# ax3.plot( x, vm )
# ax3.plot( xs, vms )
# ax3.set_title( 'Temperature' )
# ax3.set_xlabel('Pressure, GPa')
# ax3.set_ylabel('Temperature, K')
# ax3.set_xlim( [0,1000])
# ax3.set_ylim( [1200,13000] )

# # Fei et al. (2021) MgSiO3 melting curve
# P_lin = np.linspace(0,1E4,1000)
# # lower bound
# T_lower = 6000*(P_lin/140)**0.26
# ax3.plot( P_lin, T_lower )
# # upper bound
# T_upper = 6295*(P_lin/140)**0.317
# ax3.plot( P_lin, T_upper )

# # now get temperature along 0.4 melt fraction contour
# #phi = 0.4
# #vmphi = vm * phi + vms * (1.0-phi)
# #ax3.plot( x, vmphi )

# # Pr and Tr
# #tp, Pr = np.loadtxt('Pr.dat', unpack=True )
# #tt, Tr = np.loadtxt('Tr.dat', unpack=True )
# #ax3.plot( Pr, Tr, 'r--' )

# plt.colorbar()
# plt.show()
