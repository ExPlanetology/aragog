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
"""A phase defines the equation of state (EOS) and transport properties."""

from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import Field, dataclass, field, fields
from typing import Protocol

import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d

from spider.parser import _MeshSettings, _PhaseMixedSettings, _PhaseSettings
from spider.utilities import combine_properties, is_file, is_number, tanh_weight

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class PropertyABC(ABC):
    """A property whose value is to be evaluated at temperature and pressure.

    Args:
        name: Name of the property

    Attributes:
        name: Name of the property
    """

    name: str

    @abstractmethod
    def _get_value(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray | float:
        """Computes the property value at temperature and pressure.

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            The property value evaluated at temperature and pressure.
        """

    def __call__(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray | float:
        """Returns an array with the same shape as pressure"""
        return self._get_value(temperature, pressure)


@dataclass(kw_only=True)
class ConstantProperty(PropertyABC):
    """A property with a constant value

    Args:
        name: Name of the property
        value: The constant value

    Attributes:
        name: Name of the property
        value: The constant value
        ndim: Number of dimensions, which is equal to zero
    """

    value: float
    ndim: int = field(init=False, default=0)

    @override
    def _get_value(self, temperature: np.ndarray, pressure: np.ndarray) -> float:
        """See base class."""
        del temperature
        del pressure
        return self.value


@dataclass(kw_only=True)
class LookupProperty1D(PropertyABC):
    """A property from a 1-D lookup

    Args:
        name: Name of the property
        value: The 1-D array

    Attributes:
        name: Name of the property
        value: The 1-D array
        ndim: Number of dimensions, which is equal to one
    """

    value: np.ndarray
    ndim: int = field(init=False, default=1)
    _lookup: interp1d = field(init=False)

    def __post_init__(self):
        # Sort the data to ensure x is increasing
        data: np.ndarray = self.value[self.value[:, 0].argsort()]
        self._lookup = interp1d(data[:, 0], data[:, 1])

    @override
    def _get_value(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """See base class."""
        del temperature
        # TODO: Must return a 1-D array for liquidus and solidus
        return self._lookup(pressure).reshape(-1, 1)


@dataclass(kw_only=True)
class LookupProperty2D(PropertyABC):
    """A property from a 2-D lookup

    Args:
        name: Name of the property
        value: The 2-D array

    Attributes:
        name: Name of the property
        value: The 2-D array
        ndim: Number of dimensions, which is equal to two
    """

    value: np.ndarray
    ndim: int = field(init=False, default=2)
    _lookup: RectBivariateSpline = field(init=False)

    def __post_init__(self):
        # x and y must be increasing otherwise the interpolation might give unexpected behaviour
        x_values = self.value[:, 0].round(decimals=0)
        logger.debug("x_values round = %s", x_values)
        logger.debug("self.value.shape = %s", self.value.shape)
        x_values: np.ndarray = np.unique(self.value[:, 0])
        logger.debug("x_values.shape = %s", x_values.shape)
        y_values: np.ndarray = np.unique(self.value[:, 1])
        logger.debug("y_values.shape = %s", y_values.shape)
        z_values: np.ndarray = self.value[:, 2]
        logger.debug("z_values = %s", z_values)
        z_values = z_values.reshape((x_values.size, y_values.size), order="F")
        self._lookup = RectBivariateSpline(x_values, y_values, z_values, kx=1, ky=1, s=0)

    @override
    def _get_value(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """See base class."""
        return self._lookup(pressure, temperature, grid=False)


class PhaseEvaluatorProtocol(Protocol):
    """Phase evaluator protocol

    raise NotImplementedError() is to prevent pylint from reporting assignment-from-no-return /
    E1111.
    """

    def density(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray | float:
        raise NotImplementedError()

    def dTdPs(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def gravitational_acceleration(self, temperature: np.ndarray, pressure: np.ndarray) -> float:
        """To compute dT/dr at constant entropy."""
        raise NotImplementedError()

    def heat_capacity(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray | float:
        raise NotImplementedError()

    def melt_fraction(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray | float:
        raise NotImplementedError()

    def thermal_conductivity(
        self, temperature: np.ndarray, pressure: np.ndarray
    ) -> np.ndarray | float:
        raise NotImplementedError()

    def thermal_expansivity(
        self, temperature: np.ndarray, pressure: np.ndarray
    ) -> np.ndarray | float:
        raise NotImplementedError()

    def viscosity(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray | float:
        raise NotImplementedError()


class SinglePhaseEvaluator:
    """Contains the objects to evaluate the EOS and transport properties of a phase.

    Args:
        settings: phase
        mesh: mesh
    """

    # For typing
    density: PropertyABC
    gravitational_acceleration: PropertyABC
    heat_capacity: PropertyABC
    melt_fraction: PropertyABC
    thermal_conductivity: PropertyABC
    thermal_expansivity: PropertyABC
    viscosity: PropertyABC

    def __init__(self, settings: _PhaseSettings, mesh: _MeshSettings):
        self._settings: _PhaseSettings = settings
        self._mesh: _MeshSettings = mesh
        cls_fields: tuple[Field, ...] = fields(self._settings)
        for field_ in cls_fields:
            name: str = field_.name
            value = getattr(self._settings, field_.name)

            if is_number(value):
                # Numbers have already been scaled
                setattr(self, name, ConstantProperty(name=name, value=value))

            elif is_file(value):
                with open(value, encoding="utf-8") as infile:
                    logger.debug("%s is a file = %s", name, value)
                    header = infile.readline()
                    col_names = header[1:].split()
                value_array: np.ndarray = np.loadtxt(value, ndmin=2)
                logger.debug("before scaling, value_array = %s", value_array)
                # Must scale lookup data
                for nn, col_name in enumerate(col_names):
                    logger.info("Scaling %s from %s", col_name, value)
                    value_array[:, nn] /= getattr(self._settings.scalings_, col_name)
                logger.debug("after scaling, value_array = %s", value_array)
                ndim = value_array.shape[1]
                logger.debug("ndim = %d", ndim)
                if ndim == 2:
                    setattr(self, name, LookupProperty1D(name=name, value=value_array))
                elif ndim == 3:
                    setattr(self, name, LookupProperty2D(name=name, value=value_array))
                else:
                    raise ValueError(f"Lookup data must have 2 or 3 dimensions, not {ndim}")
            else:
                msg: str = f"Cannot interpret value ({value}): not a number or a file"
                raise ValueError(msg)

        self.gravitational_acceleration = ConstantProperty(
            "gravitational_acceleration", value=self._mesh.gravitational_acceleration
        )

    def dTdPs(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """TODO: Update reference to sphinx: Solomatov (2007), Treatise on Geophysics, Eq. 3.2"""
        dTdPs: np.ndarray = (
            self.thermal_expansivity(temperature, pressure)
            * temperature
            / (self.density(temperature, pressure) * self.heat_capacity(temperature, pressure))
        )

        return dTdPs


class _MixedPhaseEvaluator:
    """Evaluates the EOS and transport properties of a mixed phase.

    This only computes quantities within the mixed phase region where 0 < melt fraction < 1.

    Args:
        settings: Mixed phase settings
        solid: Solid phase evaluator
        liquid: Liquid phase evaluator
    """

    def __init__(
        self,
        settings: _PhaseMixedSettings,
        solid: PhaseEvaluatorProtocol,
        liquid: PhaseEvaluatorProtocol,
    ):
        self.settings: _PhaseMixedSettings = settings
        self.solid: PhaseEvaluatorProtocol = solid
        self.liquid: PhaseEvaluatorProtocol = liquid
        self.solidus: LookupProperty1D = self._get_melting_curve_lookup(
            "solidus", self.settings.solidus
        )
        self.liquidus: LookupProperty1D = self._get_melting_curve_lookup(
            "liquidus", self.settings.liquidus
        )

    def density(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """Density of the mixed phase computed by volume additivity"""
        liquidus_temperature: np.ndarray = self.liquidus(temperature, pressure)
        solidus_temperature: np.ndarray = self.solidus(temperature, pressure)
        melt_fraction: np.ndarray = self.melt_fraction(temperature, pressure)
        density_inverse: np.ndarray = melt_fraction / self.liquid.density(
            liquidus_temperature, pressure
        )
        density_inverse += (1 - melt_fraction) / self.solid.density(solidus_temperature, pressure)
        density: np.ndarray = 1 / density_inverse

        return density

    def dTdPs(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """TODO: Update reference to sphinx: Solomatov (2007), Treatise on Geophysics, Eq. 3.2"""
        dTdPs: np.ndarray = (
            self.thermal_expansivity(temperature, pressure)
            * temperature
            / (self.density(temperature, pressure) * self.heat_capacity(temperature, pressure))
        )

        return dTdPs

    def gravitational_acceleration(
        self, temperature: np.ndarray, pressure: np.ndarray
    ) -> np.ndarray:
        """Gravitational acceleration

        Same for both the solid and liquid phase so just pick one.
        """
        return self.solid.gravitational_acceleration(temperature, pressure)

    def heat_capacity(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """Heat capacity of the mixed phase :cite:p:`{Equation 4,}SOLO07`"""
        liquidus_temperature: np.ndarray = self.liquidus(temperature, pressure)
        solidus_temperature: np.ndarray = self.solidus(temperature, pressure)
        delta_fusion_temperature: np.ndarray = liquidus_temperature - solidus_temperature
        heat_capacity: np.ndarray = self.settings.latent_heat_of_fusion / delta_fusion_temperature

        return heat_capacity

    def melt_fraction_no_clip(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """Melt fraction without clipping.

        This is the melt fraction (phi) without clipping, which effectively determines the phase
        because phi<0 for the solid, 0<phi<1 for the mixed phase, and phi>1 for the melt.
        """
        liquidus_temperature: np.ndarray = self.liquidus(temperature, pressure)
        logger.debug("liquidus_temperature.shape = %s", liquidus_temperature.shape)
        solidus_temperature: np.ndarray = self.solidus(temperature, pressure)
        logger.debug("solidus_temperature.shape = %s", solidus_temperature.shape)
        delta_fusion_temperature: np.ndarray = liquidus_temperature - solidus_temperature
        logger.debug("delta_fusion_temperature.shape = %s", delta_fusion_temperature.shape)
        logger.debug("temperature.shape = %s", temperature.shape)
        melt_fraction: np.ndarray = (temperature - solidus_temperature) / delta_fusion_temperature
        logger.debug("melt_fraction_no_clip.shape = %s", melt_fraction.shape)

        return melt_fraction

    def melt_fraction(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """Melt fraction of the mixed phase

        The melt fraction is always between zero and one.
        """
        melt_fraction_no_clip: np.ndarray = self.melt_fraction_no_clip(temperature, pressure)
        melt_fraction = np.clip(melt_fraction_no_clip, 0, 1)
        logger.debug("melt_fraction.shape = %s", melt_fraction.shape)

        return melt_fraction

    def porosity(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """Porosity of the mixed phase, that is the volume fraction occupied by the melt"""
        liquidus_temperature: np.ndarray = self.liquidus(temperature, pressure)
        solidus_temperature: np.ndarray = self.solidus(temperature, pressure)
        density: np.ndarray = self.density(temperature, pressure)
        liquidus_density: np.ndarray = self.liquid.density(liquidus_temperature, pressure)
        solidus_density: np.ndarray = self.solid.density(solidus_temperature, pressure)
        porosity: np.ndarray = (solidus_density - density) / (solidus_density - liquidus_density)

        return porosity

    def thermal_conductivity(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """Thermal conductivity of the mixed phase by linear mixing"""
        melt_fraction: np.ndarray = self.melt_fraction(temperature, pressure)
        conductivity: np.ndarray = melt_fraction * self.liquid.thermal_conductivity(
            temperature, pressure
        )
        conductivity += (1 - melt_fraction) * self.solid.thermal_conductivity(
            temperature, pressure
        )
        logger.debug("thermal_conductivity.shape = %s", conductivity.shape)

        return conductivity

    def thermal_expansivity(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """Thermal expansivity of the mixed phase :cite:p:`{Equation 3,}SOLO07`

        The first term in :cite:t:`{Equation 3,}SOLO07` is not included because it is small
        compared to the latent heat term :cite:p:`{Equation 33,}SS93`.
        """
        liquidus_temperature: np.ndarray = self.liquidus(temperature, pressure)
        solidus_temperature: np.ndarray = self.solidus(temperature, pressure)
        density: np.ndarray = self.density(temperature, pressure)
        liquidus_density: np.ndarray = self.liquid.density(liquidus_temperature, pressure)
        solidus_density: np.ndarray = self.solid.density(solidus_temperature, pressure)
        delta_fusion_temperature: np.ndarray = liquidus_temperature - solidus_temperature
        thermal_expansivity: np.ndarray = (
            (solidus_density - liquidus_density) / delta_fusion_temperature / density
        )

        return thermal_expansivity

    def viscosity(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """Viscosity of the mixed phase"""
        liquidus_temperature: np.ndarray = self.liquidus(temperature, pressure)
        solidus_temperature: np.ndarray = self.solidus(temperature, pressure)
        melt_fraction: np.ndarray = self.melt_fraction(temperature, pressure)
        weight: np.ndarray = tanh_weight(
            melt_fraction,
            self.settings.rheological_transition_melt_fraction,
            self.settings.rheological_transition_width,
        )
        liquidus_viscosity: np.ndarray = self.liquid.viscosity(liquidus_temperature, pressure)
        solidus_viscosity: np.ndarray = self.solid.viscosity(solidus_temperature, pressure)
        log10_viscosity: np.ndarray = weight * np.log10(liquidus_viscosity) + (
            1 - weight
        ) * np.log10(solidus_viscosity)
        viscosity: np.ndarray = 10**log10_viscosity

        return viscosity

    def _get_melting_curve_lookup(self, name: str, value: str) -> LookupProperty1D:
        with open(value, encoding="utf-8") as infile:
            header = infile.readline()
            col_names = header[1:].split()
        value_array: np.ndarray = np.loadtxt(value, ndmin=2)
        logger.debug("before scaling, value_array = %s", value_array)
        for nn, col_name in enumerate(col_names):
            logger.info("Scaling %s from %s", col_name, value)
            value_array[:, nn] /= getattr(self.settings.scalings_, col_name)

        return LookupProperty1D(name=name, value=value_array)


class CompositePhaseEvaluator:
    """Evaluates the EOS and transport properties of a composite phase.

    This combines the single phase evaluators for the liquid and solid regions with the mixed phase
    evaluator for the mixed phase region. This ensure that the phase properties are computed
    correctly for all temperatures and pressures.

    Args:
        solid: Solid phase evaluator
        liquid: Liquid phase evaluator
        mixed: Mixed phase evaluator
    """

    def __init__(
        self,
        solid: PhaseEvaluatorProtocol,
        liquid: PhaseEvaluatorProtocol,
        mixed: _MixedPhaseEvaluator,
    ):
        self.solid: PhaseEvaluatorProtocol = solid
        self.liquid: PhaseEvaluatorProtocol = liquid
        self.mixed: _MixedPhaseEvaluator = mixed

    def density(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """Density"""
        return self._get_composite(temperature, pressure, "density")

    def dTdPs(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """dTdPs"""
        return self._get_composite(temperature, pressure, "dTdPs")

    def gravitational_acceleration(
        self, temperature: np.ndarray, pressure: np.ndarray
    ) -> np.ndarray:
        """Gravitational acceleration"""
        return self.mixed.gravitational_acceleration(temperature, pressure)

    def liquidus(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        return self.mixed.liquidus(temperature, pressure)

    def heat_capacity(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """Heat capacity"""
        return self._get_composite(temperature, pressure, "heat_capacity")

    def melt_fraction(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """Melt fraction"""
        return self.mixed.melt_fraction(temperature, pressure)

    def solidus(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        return self.mixed.solidus(temperature, pressure)

    def thermal_conductivity(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """Thermal conductivity"""
        return self._get_composite(temperature, pressure, "thermal_conductivity")

    def thermal_expansivity(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """Thermal expansivity"""
        return self._get_composite(temperature, pressure, "thermal_expansivity")

    def viscosity(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """Viscosity"""
        smoothing: np.ndarray | float = self._get_smoothing(temperature, pressure)
        mixed: np.ndarray = np.log10(self.mixed.viscosity(temperature, pressure))
        single: np.ndarray = np.log10(
            self._get_single_phase_to_blend(temperature, pressure, "viscosity")
        )
        combined: np.ndarray = 10 ** combine_properties(smoothing, mixed, single)

        return combined

    def _get_smoothing(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray | float:
        """Blends phases across phase boundaries."""

        phase_transition_width: float = self.mixed.settings.phase_transition_width
        melt_fraction_no_clip: np.ndarray = self.mixed.melt_fraction_no_clip(temperature, pressure)

        # no smoothing
        if phase_transition_width == 0.0:
            smoothing_factor: np.ndarray = np.where(
                ((melt_fraction_no_clip < 0.0) | (melt_fraction_no_clip > 1.0)), 0, 1
            )

        # tanh smoothing
        else:
            smoothing_liquid: np.ndarray = 1.0 - tanh_weight(
                melt_fraction_no_clip, 1.0, phase_transition_width
            )
            smoothing_solid: np.ndarray = tanh_weight(
                melt_fraction_no_clip, 0.0, phase_transition_width
            )
            smoothing_factor = np.where(
                melt_fraction_no_clip > 0.5, smoothing_liquid, smoothing_solid
            )

        logger.debug("smoothing_factor.shape = %s", smoothing_factor.shape)

        return smoothing_factor

    def _get_single_phase_to_blend(
        self, temperature: np.ndarray, pressure: np.ndarray, property_name: str
    ) -> np.ndarray:
        """Evaluates the single phase (liquid or solid) to blend with the mixed phase."""

        melt_fraction_no_clip: np.ndarray = self.mixed.melt_fraction_no_clip(temperature, pressure)

        liquid_value: np.ndarray = getattr(self.liquid, property_name)(temperature, pressure)
        solid_value: np.ndarray = getattr(self.solid, property_name)(temperature, pressure)

        single_phase_to_blend: np.ndarray = np.where(
            melt_fraction_no_clip > 0.5, liquid_value, solid_value
        )

        return single_phase_to_blend

    def _get_composite(
        self, temperature: np.ndarray, pressure: np.ndarray, property_name: str
    ) -> np.ndarray:
        """Evaluates the composite property"""

        smoothing: np.ndarray | float = self._get_smoothing(temperature, pressure)
        mixed: np.ndarray = getattr(self.mixed, property_name)(temperature, pressure)
        single: np.ndarray = self._get_single_phase_to_blend(temperature, pressure, property_name)
        combined: np.ndarray = combine_properties(smoothing, mixed, single)

        return combined
