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
"""A phase defines the equation of state (EOS) and transport properties."""

from __future__ import annotations

import logging
import sys
from dataclasses import KW_ONLY, Field, InitVar, dataclass, field, fields

import numpy as np
import numpy.typing as npt
from scipy.interpolate import RectBivariateSpline, PchipInterpolator

from aragog.interfaces import (
    MixedPhaseEvaluatorProtocol,
    PhaseEvaluatorABC,
    PhaseEvaluatorProtocol,
    PropertyProtocol,
)
from aragog.parser import Parameters, _PhaseMixedParameters, _PhaseParameters
from aragog.utilities import (
    FloatOrArray,
    combine_properties,
    is_file,
    is_number,
    tanh_weight,
)

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

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
        ndim: Number of dimensions, which is equal to zero for a constant property
    """

    name: str
    _: KW_ONLY
    value: float
    ndim: int = field(init=False, default=0)

    def eval(self) -> float:
        return self.value

    def __call__(self, temperature: npt.NDArray, pressure: npt.NDArray) -> float:
        """Evaluates the property.

        Args:
            temperature: Temperature
            pressure: Pressure
        """
        del temperature
        del pressure
        return self.eval()


@dataclass
class LookupProperty1D(PropertyProtocol):
    """A property from a 1-D lookup

    Args:
        name: Name of the property
        value: A 2-D array, with x values in the first column and y values in the second column.

    Attributes:
        name: Name of the property
        value: A 2-D array
        ndim: Number of dimensions, which is equal to one for a 1-D lookup
    """

    name: str
    _: KW_ONLY
    value: npt.NDArray
    ndim: int = field(init=False, default=1)
    _gradient: npt.NDArray = field(init=False)

    def __post_init__(self):
        # Sort the data to ensure x is increasing
        self.value = self.value[self.value[:, 0].argsort()]
        self._gradient = np.gradient(self.value[:, 1], self.value[:, 0])

    def eval(self, pressure: npt.NDArray) -> npt.NDArray:
        return np.interp(pressure, self.value[:, 0], self.value[:, 1])

    def gradient(self, pressure: npt.NDArray) -> npt.NDArray:
        """Computes the gradient"""
        return np.interp(pressure, self.value[:, 0], self._gradient)

    def __call__(self, temperature: npt.NDArray, pressure: npt.NDArray) -> npt.NDArray:
        del temperature
        return self.eval(pressure)


@dataclass
class LookupProperty2D(PropertyProtocol):
    """A property from a 2-D lookup

    Args:
        name: Name of the property
        value: The 2-D array

    Attributes:
        name: Name of the property
        value: The 2-D array
        ndim: Number of dimensions, which is equal to two for a 2-D lookup
    """

    name: str
    _: KW_ONLY
    value: npt.NDArray
    ndim: int = field(init=False, default=2)
    _lookup: RectBivariateSpline = field(init=False)

    def __post_init__(self):
        # Prepare data for spline
        x_values, y_values, z_values = self.prepare_data_for_spline(self.value)
        self._lookup = RectBivariateSpline(x_values, y_values, z_values, kx=1, ky=1, s=0)

    def prepare_data_for_spline(self, data):
        """Ensure your data is on a regular grid for RectBivariateSpline"""
        # Extract x, y, and z values
        x_values = np.unique(data[:, 0])  # Unique pressure values
        y_values = np.unique(data[:, 1])  # Unique temperature values

        # Create a grid for z values
        z_values = np.full((x_values.size, y_values.size), np.nan)

        # Find the indices of the x and y values in the unique arrays
        x_indices = np.searchsorted(x_values, data[:, 0])
        y_indices = np.searchsorted(y_values, data[:, 1])

        # Fill the z_values grid
        z_values[x_indices, y_indices] = data[:, 2]

        return x_values, y_values, z_values

    def eval(self, temperature: npt.NDArray, pressure: npt.NDArray) -> npt.NDArray:
        return self._lookup(pressure, temperature, grid=False)

    def __call__(self, temperature: npt.NDArray, pressure: npt.NDArray) -> npt.NDArray:
        return self.eval(temperature, pressure)


class SinglePhaseEvaluator(PhaseEvaluatorABC):
    """Contains the objects to evaluate the EOS and transport properties of a phase.

    Args:
        settings: Phase parameters
        gravitational_acceleration: PropertyProtocol
    """

    # For typing
    _density: PropertyProtocol
    _gravitational_acceleration: PropertyProtocol
    _heat_capacity: PropertyProtocol
    _melt_fraction: ConstantProperty
    _thermal_conductivity: PropertyProtocol
    _thermal_expansivity: PropertyProtocol
    _viscosity: PropertyProtocol

    def __init__(self, settings: _PhaseParameters, gravitational_acceleration: PropertyProtocol):
        self._settings: _PhaseParameters = settings
        cls_fields: tuple[Field, ...] = fields(self._settings)
        for field_ in cls_fields:
            name: str = field_.name
            private_name: str = f"_{name}"
            value = getattr(self._settings, field_.name)

            if is_number(value):
                # Numbers have already been scaled by the parser
                setattr(self, private_name, ConstantProperty(name=name, value=value))

            elif is_file(value):
                with open(value, encoding="utf-8") as infile:
                    logger.debug("%s is a file = %s", name, value)
                    header = infile.readline()
                    col_names = header[1:].split()
                value_array: npt.NDArray = np.loadtxt(value, ndmin=2)
                logger.debug("before scaling, value_array = %s", value_array)
                # Scale lookup data
                for nn, col_name in enumerate(col_names):
                    logger.info("Scaling %s from %s", col_name, value)
                    value_array[:, nn] /= getattr(self._settings.scalings_, col_name)
                logger.debug("after scaling, value_array = %s", value_array)
                ndim = value_array.shape[1]
                logger.debug("ndim = %d", ndim)
                if ndim == 2:
                    setattr(self, private_name, LookupProperty1D(name=name, value=value_array))
                elif ndim == 3:
                    setattr(self, private_name, LookupProperty2D(name=name, value=value_array))
                else:
                    raise ValueError(f"Lookup data must have 2 or 3 dimensions, not {ndim}")
            else:
                logger.info("Cannot interpret value (%s): not a number or a file", value)

        self._gravitational_acceleration = gravitational_acceleration

    @override
    def density(self) -> FloatOrArray:
        return self._density(self.temperature, self.pressure)

    @override
    def gravitational_acceleration(self) -> FloatOrArray:
        return self._gravitational_acceleration(self.temperature, self.pressure)

    @override
    def heat_capacity(self) -> FloatOrArray:
        return self._heat_capacity(self.temperature, self.pressure)

    @override
    def melt_fraction(self) -> float:
        return self._melt_fraction(self.temperature, self.pressure)
    
    @override
    def latent_heat(self) -> float:
        return 0.

    @override
    def thermal_conductivity(self) -> FloatOrArray:
        return self._thermal_conductivity(self.temperature, self.pressure)

    @override
    def thermal_expansivity(self) -> FloatOrArray:
        return self._thermal_expansivity(self.temperature, self.pressure)

    @override
    def viscosity(self) -> FloatOrArray:
        return self._viscosity(self.temperature, self.pressure)

    @override
    def relative_velocity(self) -> float:
        return 0.

    @override
    def delta_specific_volume(self) -> FloatOrArray:
        return 0.0

class MixedPhaseEvaluator(PhaseEvaluatorABC):
    """Evaluates the EOS and transport properties of a mixed phase.

    This only computes quantities within the mixed phase region between the solidus and the
    liquidus. Computing quantities outside of this region will give incorrect results.

    Args:
        parameters: Parameters

    Attributes:
        settings: Mixed phase parameters
    """

    def __init__(self, parameters: Parameters):
        gravitational_acceleration: PropertyProtocol = setup_gravitational_acceleration(parameters)
        self.settings: _PhaseMixedParameters = parameters.phase_mixed
        self._liquid: PhaseEvaluatorProtocol = SinglePhaseEvaluator(
            parameters.phase_liquid, gravitational_acceleration
        )
        self._solid: PhaseEvaluatorProtocol = SinglePhaseEvaluator(
            parameters.phase_solid, gravitational_acceleration
        )
        self._solidus: LookupProperty1D = self._get_melting_curve_lookup(
            "solidus", self.settings.solidus
        )
        self._liquidus: LookupProperty1D = self._get_melting_curve_lookup(
            "liquidus", self.settings.liquidus
        )
        self._grain_size: float = self.settings.grain_size
        self._latent_heat: float = self.settings.latent_heat_of_fusion

    @override
    def set_pressure(self, pressure: npt.NDArray) -> None:
        """Sets pressure and updates quantities that only depend on pressure"""
        super().set_pressure(pressure)
        # Sets the temperature of the solid and liquid phases to the appropriate melting curve in
        # order to evaluate mixed properties
        self._solid.set_temperature(self.solidus())
        self._solid.set_pressure(pressure)
        self._liquid.set_temperature(self.liquidus())
        self._liquid.set_pressure(pressure)
        self._delta_density = self._solid.density() - self._liquid.density()
        self._delta_specific_volume = 1.0/self._liquid.density() - 1.0/self._solid.density()
        self._delta_fusion = self.liquidus() - self.solidus()
        # Heat capacity of the mixed phase :cite:p:`{Equation 4,}SOLO07`
        self._heat_capacity = self.latent_heat() / self.delta_fusion()

    @override
    def update(self):
        # Melt fraction without clipping
        # phi<0 for the solid, 0<phi<1 for the mixed phase, and phi>1 for the melt
        self._melt_fraction_no_clip = (self.temperature - self.solidus()) / self.delta_fusion()
        logger.debug("_melt_fraction_no_clip = %s", self.melt_fraction_no_clip())
        self._melt_fraction = np.clip(self._melt_fraction_no_clip, 0, 1)

        # Mixed density by volume additivity
        self._density = combine_properties(
            self.melt_fraction(), 1 / self._liquid.density(), 1 / self._solid.density()
        )
        self._density = 1 / self._density

        # Porosity
        epsilon = 1e-10  # Used to avoid division by zero when dividing by the delta density for higher mass planets
        self._porosity = (self._solid.density() - self.density()) / (self.delta_density() + epsilon)

        # Relative velocity between melt and solid
        self._relative_velocity = self._get_relative_velocity()

        # Thermal conductivity
        self._thermal_conductivity = combine_properties(
            self.melt_fraction(),
            self._liquid.thermal_conductivity(),
            self._solid.thermal_conductivity(),
        )

        # Thermal expansivity :cite:p:`{Equation 3,}SOLO07`
        # The first term in :cite:t:`{Equation 3,}SOLO07` is not included because it is small
        # compared to the latent heat term :cite:p:`{Equation 33,}SS93`.
        self._thermal_expansivity = self.delta_density() / self.delta_fusion() / self.density()

        # Viscosity
        weight: npt.NDArray = tanh_weight(
            self.melt_fraction(),
            self.settings.rheological_transition_melt_fraction,
            self.settings.rheological_transition_width,
        )
        log10_viscosity: npt.NDArray = combine_properties(
            weight, np.log10(self._liquid.viscosity()), np.log10(self._solid.viscosity())
        )
        self._viscosity = 10**log10_viscosity

    def delta_density(self) -> FloatOrArray:
        return self._delta_density

    def delta_fusion(self) -> npt.NDArray:
        return self._delta_fusion

    @override
    def density(self) -> npt.NDArray:
        return self._density

    @override
    def gravitational_acceleration(self) -> FloatOrArray:
        return self._solid.gravitational_acceleration()

    @override
    def heat_capacity(self) -> FloatOrArray:
        """Heat capacity of the mixed phase :cite:p:`{Equation 4,}SOLO07`"""
        return self._heat_capacity

    def liquidus(self) -> npt.NDArray:
        """Liquidus"""
        liquidus: npt.NDArray = self._liquidus.eval(self.pressure)
        logger.debug("liquidus = %s", liquidus)

        return liquidus

    def liquidus_gradient(self) -> npt.NDArray:
        """Liquidus gradient"""
        return self._liquidus.gradient(self.pressure)

    def melt_fraction_no_clip(self) -> npt.NDArray:
        """Melt fraction without clipping"""
        return self._melt_fraction_no_clip

    @override
    def melt_fraction(self) -> npt.NDArray:
        """Melt fraction of the mixed phase

        The melt fraction is always between zero and one.
        """
        return self._melt_fraction

    @override
    def latent_heat(self) -> float:
        """Latent heat of fusion"""
        return self._latent_heat

    def porosity(self) -> npt.NDArray:
        """Porosity of the mixed phase, that is the volume fraction occupied by the melt"""
        return self._porosity

    def solidus(self) -> npt.NDArray:
        """Solidus"""
        solidus: npt.NDArray = self._solidus.eval(self.pressure)
        logger.debug("solidus = %s", solidus)

        return solidus

    def solidus_gradient(self) -> npt.NDArray:
        """Solidus gradient"""
        return self._solidus.gradient(self.pressure)

    @override
    def thermal_conductivity(self) -> npt.NDArray:
        return self._thermal_conductivity

    @override
    def thermal_expansivity(self) -> npt.NDArray:
        return self._thermal_expansivity

    @override
    def viscosity(self) -> npt.NDArray:
        return self._viscosity

    @override
    def relative_velocity(self) -> npt.NDArray:
        """Relative velocity between melt and solid"""
        return self._relative_velocity

    def _get_relative_velocity(self) -> npt.NDArray:
        """Compute relative velocity"""
        dv = (
            - self.delta_density()
            * self.gravitational_acceleration()
            * self._permeability()
            / self._liquid.viscosity()
        )
        return dv

    @override
    def delta_specific_volume(self) -> npt.NDArray:
        """Difference of specific volume between melt and solid"""
        return self._delta_specific_volume

    def _permeability(self) -> npt.NDArray:

        # RumpfGupte regime (default)
        permeability = self._permeability_rumpf_gupte()

        # Stokes regime
        permeability = np.where(
            self._porosity > 0.771462,
            self._permeability_stokes(),
            permeability
            )

        # Blake-Kozeny-Carman regime
        permeability = np.where(
            self._porosity < 0.0769452,
            self._permeability_blake_kozeny_carman(),
            permeability
            )
        
        return permeability

    def _permeability_stokes(self) -> npt.NDArray:
        """Permeability for Stokes flow in the mixed phase"""
        permeability = 2./9.*self._grain_size**2
        return permeability

    def _permeability_blake_kozeny_carman(self) -> npt.NDArray:
        """Permeability for Blake-Kozeny-Carman flow in the mixed phase"""
        permeability = (
            0.001
            * self._grain_size**2
            * self._porosity**2
            / (1-self._porosity)**2
        )
        return permeability

    def _permeability_rumpf_gupte(self) -> npt.NDArray:
        """Permeability for Rumpf-Gupte flow in the mixed phase"""
        permeability = (
            5./7.
            * self._grain_size**2
            * self._porosity**4.5
        )
        return permeability

    def _get_melting_curve_lookup(self, name: str, value: str) -> LookupProperty1D:
        with open(value, encoding="utf-8") as infile:
            header = infile.readline()
            col_names = header[1:].split()
        value_array: npt.NDArray = np.loadtxt(value, ndmin=2)
        logger.debug("before scaling, value_array = %s", value_array)
        for nn, col_name in enumerate(col_names):
            logger.info("Scaling %s from %s", col_name, value)
            value_array[:, nn] /= getattr(self.settings.scalings_, col_name)

        return LookupProperty1D(name=name, value=value_array)


class CompositePhaseEvaluator(PhaseEvaluatorABC):
    """Evaluates the EOS and transport properties of a composite phase.

    This combines the single phase evaluators for the liquid and solid regions with the mixed phase
    evaluator for the mixed phase region. This ensure that the phase properties are computed
    correctly for all temperatures and pressures.

    Args:
        parameters: Parameters
    """

    def __init__(self, parameters: Parameters):
        gravitational_acceleration: PropertyProtocol = setup_gravitational_acceleration(parameters)
        self._liquid: PhaseEvaluatorProtocol = SinglePhaseEvaluator(
            parameters.phase_liquid, gravitational_acceleration
        )
        self._solid: PhaseEvaluatorProtocol = SinglePhaseEvaluator(
            parameters.phase_solid, gravitational_acceleration
        )
        self._mixed: MixedPhaseEvaluator = MixedPhaseEvaluator(parameters)

    @override
    def set_temperature(self, temperature: npt.NDArray) -> None:
        super().set_temperature(temperature)
        self._solid.set_temperature(temperature)
        self._liquid.set_temperature(temperature)
        self._mixed.set_temperature(temperature)

    @override
    def set_pressure(self, pressure: npt.NDArray) -> None:
        """Sets pressure and updates quantities that only depend on pressure"""
        super().set_pressure(pressure)
        self._solid.set_pressure(pressure)
        self._liquid.set_pressure(pressure)
        self._mixed.set_pressure(pressure)

    @override
    def update(self) -> None:
        self._mixed.update()
        self._set_blending_and_masks()
        self._density = self._get_composite("density")
        self._heat_capacity = self._get_composite("heat_capacity")
        self._thermal_conductivity = self._get_composite("thermal_conductivity")
        self._thermal_expansivity = self._get_composite("thermal_expansivity")
        self._dTdPs = self._get_composite("dTdPs")
        self._dTdrs = self._get_composite("dTdrs")
        self._relative_velocity = self._get_composite("relative_velocity")
        self._delta_specific_volume = self._get_composite("delta_specific_volume")

        name: str = "viscosity"
        log10_mixed_phase: npt.NDArray = np.log10(getattr(self._mixed, name)())
        single_phase: npt.NDArray = np.empty_like(self._blending_factor)
        try:
            single_phase[self._liquid_mask] = getattr(self._liquid, name)()[self._liquid_mask]
        except (IndexError, TypeError):
            single_phase[self._liquid_mask] = getattr(self._liquid, name)()
        try:
            single_phase[self._solid_mask] = getattr(self._solid, name)()[self._solid_mask]
        except (IndexError, TypeError):
            single_phase[self._solid_mask] = getattr(self._solid, name)()
        log10_single_phase: npt.NDArray = np.log10(single_phase)
        self._viscosity = combine_properties(
            self._blending_factor, log10_mixed_phase, log10_single_phase
        )
        self._viscosity = 10**self._viscosity

    @override
    def density(self) -> npt.NDArray:
        return self._density

    @override
    def dTdPs(self) -> npt.NDArray:
        return self._dTdPs

    @override
    def dTdrs(self) -> npt.NDArray:
        return self._dTdrs

    @override
    def gravitational_acceleration(self) -> FloatOrArray:
        return self._mixed.gravitational_acceleration()

    @override
    def heat_capacity(self) -> npt.NDArray:
        """Heat capacity"""
        return self._heat_capacity

    def liquidus(self) -> npt.NDArray:
        return self._mixed.liquidus()

    def liquidus_gradient(self) -> npt.NDArray:
        return self._mixed.liquidus_gradient()

    @override
    def melt_fraction(self) -> npt.NDArray:
        """Melt fraction"""
        return self._mixed.melt_fraction()

    @override
    def latent_heat(self) -> FloatOrArray:
        """Latent heat of fusion"""
        return self._mixed.latent_heat()

    def solidus(self) -> npt.NDArray:
        return self._mixed.solidus()

    def solidus_gradient(self) -> npt.NDArray:
        return self._mixed.solidus_gradient()

    @override
    def thermal_conductivity(self) -> npt.NDArray:
        """Thermal conductivity"""
        return self._thermal_conductivity

    @override
    def thermal_expansivity(self) -> npt.NDArray:
        """Thermal expansivity"""
        return self._thermal_expansivity

    def viscosity(self) -> npt.NDArray:
        """Viscosity"""
        return self._viscosity

    @override
    def relative_velocity(self) -> npt.NDArray:
        """Relative velocity between melt and solid"""
        return self._relative_velocity

    @override
    def delta_specific_volume(self) -> npt.NDArray:
        """Difference of specific volume between melt and solid"""
        return self._delta_specific_volume

    def _set_blending_and_masks(self) -> None:
        """Sets blending and masks."""

        phase_transition_width: float = self._mixed.settings.phase_transition_width
        melt_fraction_no_clip: npt.NDArray = self._mixed.melt_fraction_no_clip()

        if phase_transition_width == 0.0:
            blending_factor: npt.NDArray = np.where(
                ((melt_fraction_no_clip < 0.0) | (melt_fraction_no_clip > 1.0)),
                0,
                1,
            )
        else:
            blending_liquid: npt.NDArray = 1.0 - tanh_weight(
                melt_fraction_no_clip, 1.0, phase_transition_width
            )
            blending_solid: npt.NDArray = tanh_weight(
                melt_fraction_no_clip, 0.0, phase_transition_width
            )
            blending_factor = np.where(
                melt_fraction_no_clip > 0.5, blending_liquid, blending_solid
            )

        self._blending_factor = blending_factor
        logger.debug("_blending_factor = %s", self._blending_factor)
        self._liquid_mask = melt_fraction_no_clip > 0.5
        logger.debug("_liquid_mask = %s", self._liquid_mask)
        self._solid_mask = ~self._liquid_mask
        logger.debug("_solid_mask = %s", self._solid_mask)

    def _get_composite(self, property_name: str) -> npt.NDArray:
        """Evaluates the composite property"""
        mixed_phase: npt.NDArray = getattr(self._mixed, property_name)()
        single_phase: npt.NDArray = np.empty_like(self._blending_factor)
        logger.debug("single_phase = %s", single_phase)
        logger.debug("_liquid_mask = %s", self._liquid_mask)
        logger.debug("_solid_mask = %s", self._solid_mask)
        test = getattr(self._liquid, property_name)()
        logger.debug("test = %s", test)

        # logger.debug(self.temperature.shape)
        # logger.debug(self.pressure.shape)
        # logger.debug(mixed_phase.shape)
        # logger.debug(single_phase.shape)

        # TODO: This is ugly.  Clean up logic.
        try:
            single_phase[self._liquid_mask] = getattr(self._liquid, property_name)()[
                self._liquid_mask
            ]
        except (IndexError, TypeError):
            single_phase[self._liquid_mask] = getattr(self._liquid, property_name)()
        try:
            single_phase[self._solid_mask] = getattr(self._solid, property_name)()[
                self._solid_mask
            ]
        except (IndexError, TypeError):
            single_phase[self._solid_mask] = getattr(self._solid, property_name)()

        combined: npt.NDArray = combine_properties(
            self._blending_factor, mixed_phase, single_phase
        )

        return combined


@dataclass
class PhaseEvaluatorCollection:
    """A collection of phase evaluators

    Creates the phase evaluators and selects the active phase based on configuration data.

    Args:
        parameters: Parameters

    Attributes:
        liquid: Liquid evaluator
        solid: Solid evaluator
        mixed: Mixed evaluator
        composite: Composite evaluator
        active: The active evaluator, which is defined by configuration data
    """

    parameters: InitVar[Parameters]
    liquid: PhaseEvaluatorProtocol = field(init=False)
    solid: PhaseEvaluatorProtocol = field(init=False)
    mixed: MixedPhaseEvaluatorProtocol = field(init=False)
    composite: MixedPhaseEvaluatorProtocol = field(init=False)
    active: PhaseEvaluatorProtocol = field(init=False)

    def __post_init__(self, parameters: Parameters):
        gravitational_acceleration: PropertyProtocol = setup_gravitational_acceleration(parameters)
        self.liquid = SinglePhaseEvaluator(parameters.phase_liquid, gravitational_acceleration)
        self.solid = SinglePhaseEvaluator(parameters.phase_solid, gravitational_acceleration)
        self.mixed = MixedPhaseEvaluator(parameters)
        self.composite = CompositePhaseEvaluator(parameters)

        # Configuration data defines which phase to use for the model.
        phase_to_use: str = parameters.phase_mixed.phase

        if phase_to_use == "liquid":
            self.active = self.liquid
        elif phase_to_use == "solid":
            self.active = self.solid
        # Allowing selection of self.mixed doesn't really make sense because it will probably give
        # crazy results outside the mixed phase region. Hence just use composite.
        elif phase_to_use == "mixed" or phase_to_use == "composite":
            self.active = self.composite
        else:
            raise ValueError(f"Phase = {phase_to_use} is not a valid selection")

def setup_gravitational_acceleration(parameters: Parameters):
    """Sets up the gravitational acceleration property.

    Args:
        parameters: Parameters

    Returns:
        gravitational_acceleration: PropertyProtocol
    """
    if parameters.mesh.eos_method == 1:
        # AdamsWilliamson EOS method = constant gravitational acceleration
        gravitational_acceleration = ConstantProperty(
            "gravitational_acceleration", value=parameters.mesh.gravitational_acceleration
        )
    elif parameters.mesh.eos_method == 2:
        # User defined EOS method = space dependent gravitational acceleration (1D lookup)
        # Re-interpolate EOS-data to match standard lookup pressure grid
        interp_gravity = PchipInterpolator(
            np.flip(parameters.mesh.eos_pressure),
            np.flip(parameters.mesh.eos_gravity))
        num_points: int = 138
        max_pressure: float = 1.37e11 / parameters.scalings.pressure
        lookup_data: npt.NDArray = np.zeros((num_points, 2), dtype=float)
        lookup_data[:,0] = np.linspace(0.0, max_pressure, num=num_points)
        lookup_data[:,1] = interp_gravity(lookup_data[:,0])
        gravitational_acceleration = LookupProperty1D(
            "gravitational_acceleration",
            value=lookup_data,
        )
    else:
        raise ValueError(f"EOS method = {parameters.mesh.eos_method} is not a valid selection")

    return gravitational_acceleration