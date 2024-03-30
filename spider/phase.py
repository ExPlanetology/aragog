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
from dataclasses import KW_ONLY, Field, dataclass, field, fields

import numpy as np
from scipy.interpolate import RectBivariateSpline

from spider.interfaces import PhaseEvaluatorProtocol, PropertyProtocol
from spider.parser import _MeshSettings, _PhaseMixedSettings, _PhaseSettings
from spider.utilities import (
    FloatOrArray,
    combine_properties,
    is_file,
    is_number,
    tanh_weight,
)

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class ConstantProperty(PropertyProtocol):
    """A property with a constant value

    Args:
        name: Name of the property
        value: The constant value

    Attributes:
        name: Name of the property
        value: The constant value
        ndim: Number of dimensions, which is equal to zero
    """

    name: str
    _: KW_ONLY
    value: float
    ndim: int = field(init=False, default=0)

    def __call__(self, temperature: np.ndarray, pressure: np.ndarray) -> float:
        del temperature
        del pressure
        return self.value


@dataclass
class LookupProperty1D(PropertyProtocol):
    """A property from a 1-D lookup

    Args:
        name: Name of the property
        value: The 1-D array

    Attributes:
        name: Name of the property
        value: The 1-D array
        ndim: Number of dimensions, which is equal to one
    """

    name: str
    _: KW_ONLY
    value: np.ndarray
    ndim: int = field(init=False, default=1)
    _gradient: np.ndarray = field(init=False)

    def __post_init__(self):
        # Sort the data to ensure x is increasing
        self.value = self.value[self.value[:, 0].argsort()]
        self._gradient = np.gradient(self.value[:, 1], self.value[:, 0])

    def gradient(self, pressure: np.ndarray) -> np.ndarray:
        """Computes the gradient"""
        return np.interp(pressure, self.value[:, 0], self._gradient)

    def __call__(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        del temperature
        # TODO: Will this break?  Assumes always one column
        return np.interp(pressure, self.value[:, 0], self.value[:, 1]).reshape(-1, 1)


@dataclass
class LookupProperty2D(PropertyProtocol):
    """A property from a 2-D lookup

    Args:
        name: Name of the property
        value: The 2-D array

    Attributes:
        name: Name of the property
        value: The 2-D array
        ndim: Number of dimensions, which is equal to two
    """

    name: str
    _: KW_ONLY
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

    def __call__(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        return self._lookup(pressure, temperature, grid=False)


class SinglePhaseEvaluator(PhaseEvaluatorProtocol):
    """Contains the objects to evaluate the EOS and transport properties of a phase.

    Args:
        settings: phase settings
        mesh: mesh settings
    """

    # For typing
    density: PropertyProtocol
    gravitational_acceleration: PropertyProtocol
    heat_capacity: PropertyProtocol
    melt_fraction: ConstantProperty
    thermal_conductivity: PropertyProtocol
    thermal_expansivity: PropertyProtocol
    viscosity: PropertyProtocol

    def __init__(self, settings: _PhaseSettings, mesh: _MeshSettings):
        self._settings: _PhaseSettings = settings
        self._mesh: _MeshSettings = mesh
        cls_fields: tuple[Field, ...] = fields(self._settings)
        for field_ in cls_fields:
            name: str = field_.name
            value = getattr(self._settings, field_.name)

            if is_number(value):
                # Numbers have already been scaled by the parser
                setattr(self, name, ConstantProperty(name=name, value=value))

            elif is_file(value):
                with open(value, encoding="utf-8") as infile:
                    logger.debug("%s is a file = %s", name, value)
                    header = infile.readline()
                    col_names = header[1:].split()
                value_array: np.ndarray = np.loadtxt(value, ndmin=2)
                logger.debug("before scaling, value_array = %s", value_array)
                # Scale lookup data
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


class MixedPhaseEvaluator(PhaseEvaluatorProtocol):
    """Evaluates the EOS and transport properties of a mixed phase.

    This only computes quantities within the mixed phase region between the solidus and the
    liquidus. Computing quantities outside of this region will give incorrect results.

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
    ) -> FloatOrArray:
        """Gravitational acceleration

        Same for both the solid and liquid phase so just pick one.
        """
        return self.solid.gravitational_acceleration(temperature, pressure)

    def heat_capacity(self, temperature: np.ndarray, pressure: np.ndarray) -> FloatOrArray:
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
        melt_fraction: np.ndarray = np.clip(melt_fraction_no_clip, 0, 1)
        # logger.debug("melt_fraction.shape = %s", melt_fraction.shape)

        return melt_fraction

    def porosity(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """Porosity of the mixed phase, that is the volume fraction occupied by the melt"""
        liquidus_temperature: np.ndarray = self.liquidus(temperature, pressure)
        solidus_temperature: np.ndarray = self.solidus(temperature, pressure)
        density: np.ndarray = self.density(temperature, pressure)
        liquidus_density: FloatOrArray = self.liquid.density(liquidus_temperature, pressure)
        solidus_density: FloatOrArray = self.solid.density(solidus_temperature, pressure)
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
        # logger.debug("thermal_conductivity.shape = %s", conductivity.shape)

        return conductivity

    def thermal_expansivity(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """Thermal expansivity of the mixed phase :cite:p:`{Equation 3,}SOLO07`

        The first term in :cite:t:`{Equation 3,}SOLO07` is not included because it is small
        compared to the latent heat term :cite:p:`{Equation 33,}SS93`.
        """
        liquidus_temperature: np.ndarray = self.liquidus(temperature, pressure)
        solidus_temperature: np.ndarray = self.solidus(temperature, pressure)
        density: np.ndarray = self.density(temperature, pressure)
        liquidus_density: FloatOrArray = self.liquid.density(liquidus_temperature, pressure)
        solidus_density: FloatOrArray = self.solid.density(solidus_temperature, pressure)
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
        liquidus_viscosity: FloatOrArray = self.liquid.viscosity(liquidus_temperature, pressure)
        solidus_viscosity: FloatOrArray = self.solid.viscosity(solidus_temperature, pressure)
        log10_viscosity: np.ndarray = combine_properties(
            weight, np.log10(liquidus_viscosity), np.log10(solidus_viscosity)
        )
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


class CompositePhaseEvaluator(PhaseEvaluatorProtocol):
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
        mixed: MixedPhaseEvaluator,
    ):
        self.solid: PhaseEvaluatorProtocol = solid
        self.liquid: PhaseEvaluatorProtocol = liquid
        self.mixed: MixedPhaseEvaluator = mixed

    def density(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """Density"""
        return self._get_composite(temperature, pressure, "density")

    def dTdPs(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """dTdPs"""
        return self._get_composite(temperature, pressure, "dTdPs")

    def gravitational_acceleration(
        self, temperature: np.ndarray, pressure: np.ndarray
    ) -> FloatOrArray:
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

        # logger.debug("smoothing_factor.shape = %s", smoothing_factor.shape)

        return smoothing_factor

    def _get_single_phase_to_blend(
        self, temperature: np.ndarray, pressure: np.ndarray, property_name: str
    ) -> np.ndarray:
        """Evaluates the single phase (liquid or solid) to blend with the mixed phase."""

        melt_fraction_no_clip: np.ndarray = self.mixed.melt_fraction_no_clip(temperature, pressure)

        # Initialize array for blended property values
        single_phase_to_blend: np.ndarray = np.empty_like(melt_fraction_no_clip)

        # Evaluate properties only where needed based on melt fraction
        liquid_mask: np.ndarray = melt_fraction_no_clip > 0.5
        solid_mask: np.ndarray = ~liquid_mask

        single_phase_to_blend[liquid_mask] = getattr(self.liquid, property_name)(
            temperature[liquid_mask], pressure
        )
        single_phase_to_blend[solid_mask] = getattr(self.solid, property_name)(
            temperature[solid_mask], pressure
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
