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
from spider.utilities import is_file, is_number, tanh_weight

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

    def __call__(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """Returns an array with the same shape as pressure"""
        return self._get_value(temperature, pressure) * np.ones_like(pressure)


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
    def _get_value(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """See base class."""
        del pressure
        return self.value * np.ones_like(temperature)


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
        return self._lookup(pressure)


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


@dataclass
class PhaseStateStaggered:
    """Stores the state (material properties) of a phase at the staggered nodes.

    Args:
        phase_evaluator: A PhaseEvaluatorProtocol

    Attributes:
        capacitance: Thermal capacitance
        density: Density
        heat_capacity: Heat capacity
    """

    phase_evaluator: PhaseEvaluatorProtocol
    capacitance: np.ndarray = field(init=False)
    density: np.ndarray = field(init=False)
    heat_capacity: np.ndarray = field(init=False)

    def update(self, temperature: np.ndarray, pressure: np.ndarray) -> None:
        """Updates the state.

        The order of evaluation matters.

        Args:
            temperature: Temperature at the staggered nodes
            pressure: Pressure at the staggered nodes
        """
        logger.debug("Updating the state of %s", self.__class__.__name__)
        self.density = self.phase_evaluator.density(temperature, pressure)
        logger.debug("density = %s", self.density)
        self.heat_capacity = self.phase_evaluator.heat_capacity(temperature, pressure)
        logger.debug("heat_capacity = %s", self.heat_capacity)
        self.capacitance = self.density * self.heat_capacity
        logger.debug("capacitance = %s", self.capacitance)


@dataclass
class PhaseStateBasic:
    """Stores the state (material properties) of a phase at the basic nodes.

    Args:
        phase_evaluator: A PhaseEvaluatorProtocol

    Attributes:
        density: Density
        dTdrs: Adiabatic temperature gradient with respect to radius
        gravitational_acceleration: Gravitational acceleration
        heat_capacity: Heat capacity
        kinematic_viscosity: Kinematic viscosity
        thermal_conductivity: Thermal conductivity
        thermal_expansivity: Thermal expansivity
        viscosity: Dynamic viscosity
    """

    phase_evaluator: PhaseEvaluatorProtocol
    density: np.ndarray = field(init=False)
    gravitational_acceleration: np.ndarray = field(init=False)
    heat_capacity: np.ndarray = field(init=False)
    thermal_conductivity: np.ndarray = field(init=False)
    thermal_expansivity: np.ndarray = field(init=False)
    viscosity: np.ndarray = field(init=False)
    dTdrs: np.ndarray = field(init=False)
    kinematic_viscosity: np.ndarray = field(init=False)

    def update(self, temperature: np.ndarray, pressure: np.ndarray) -> None:
        """Updates the state.

        This minimises the number of function evaluations to avoid slowing down the code, hence the
        order of evaluation matters.

        Args:
            temperature: Temperature at the basic nodes
            pressure: Pressure at the basic nodes
        """
        logger.debug("Updating the state of %s", self.__class__.__name__)
        self.density = self.phase_evaluator.density(temperature, pressure)
        self.gravitational_acceleration = self.phase_evaluator.gravitational_acceleration(
            temperature, pressure
        )
        self.heat_capacity = self.phase_evaluator.heat_capacity(temperature, pressure)
        self.thermal_conductivity = self.phase_evaluator.thermal_conductivity(
            temperature, pressure
        )
        self.thermal_expansivity = self.phase_evaluator.thermal_expansivity(temperature, pressure)
        self.viscosity = self.phase_evaluator.viscosity(temperature, pressure)
        self.dTdrs = (
            -self.gravitational_acceleration
            * self.thermal_expansivity
            * temperature
            / self.heat_capacity
        )
        self.kinematic_viscosity = self.viscosity / self.density


class PhaseEvaluatorProtocol(Protocol):
    """Phase evaluator protocol

    raise NotImplementedError() is to prevent pylint from reporting assignment-from-no-return /
    E1111.
    """

    def density(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def dTdPs(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def gravitational_acceleration(
        self, temperature: np.ndarray, pressure: np.ndarray
    ) -> np.ndarray:
        """To compute dT/dr at constant entropy."""
        raise NotImplementedError()

    def heat_capacity(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def thermal_conductivity(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def thermal_expansivity(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def viscosity(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
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
                    value_array[:, nn] /= getattr(self._settings.scalings, col_name)
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
        self._settings: _PhaseMixedSettings = settings
        self.solid: PhaseEvaluatorProtocol = solid
        self.liquid: PhaseEvaluatorProtocol = liquid
        self.solidus: LookupProperty1D = self._get_melting_curve_lookup(
            "solidus", self._settings.solidus
        )
        self.liquidus: LookupProperty1D = self._get_melting_curve_lookup(
            "liquidus", self._settings.liquidus
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
        heat_capacity: np.ndarray = self._settings.latent_heat_of_fusion / delta_fusion_temperature

        return heat_capacity

    def melt_fraction(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """Melt fraction of the mixed phase

        The melt fraction is always between zero and one.
        """
        liquidus_temperature: np.ndarray = self.liquidus(temperature, pressure)
        solidus_temperature: np.ndarray = self.solidus(temperature, pressure)
        delta_fusion_temperature: np.ndarray = liquidus_temperature - solidus_temperature
        melt_fraction: np.ndarray = (temperature - solidus_temperature) / delta_fusion_temperature
        melt_fraction = np.clip(melt_fraction, 0, 1)

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
            self._settings.rheological_transition_melt_fraction,
            self._settings.rheological_transition_width,
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
            value_array[:, nn] /= getattr(self._settings.scalings_, col_name)

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
        self.mixed: PhaseEvaluatorProtocol = mixed

    def density(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """Density"""

    def dTdPs(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """dTdPs"""

    def gravitational_acceleration(
        self, temperature: np.ndarray, pressure: np.ndarray
    ) -> np.ndarray:
        """Gravitational acceleration"""

    def heat_capacity(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """Heat capacity"""

    def thermal_conductivity(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """Thermal conductivity"""

    def thermal_expansivity(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """Thermal expansivity"""

    def viscosity(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """Viscosity"""

    def _get_smoothing(self, smooth_width, gphi):
        # get smoothing across phase boundaries for a two-phase composite

        # no smoothing
        if smooth_width == 0.0:
            smth = 1.0  # mixed phase only
            if gphi < 0.0 or gphi > 1.0:
                smth = 0.0  # single phase only

        # tanh smoothing
        else:
            if gphi > 0.5:
                smth = 1.0 - tanh_weight(gphi, 1.0, smooth_width)
            else:
                smth = tanh_weight(gphi, 0.0, smooth_width)

        return smth

    def _tanh_weight(self, qty, threshold, width):
        # tanh weight for viscosity profile and smoothing

        z = (qty - threshold) / width
        fwt = 0.5 * (1.0 + np.tanh(z))
        return fwt


# Below copied from cspider. To implement above to get (optional) smoothing across the phase
# boundaries

#   smth = get_smoothing(composite->matprop_smooth_width, gphi);

#   /* now blend mixed phase EOS with single phase EOS across the phase boundary */
#   if (gphi > 0.5)
#   {
#     /* melt only properties */
#     ierr = EOSEval(composite->eos[composite->melt_slot], P, T, &eval2);
#     CHKERRQ(ierr);
#   }
#   else
#   {
#     /* solid only properties */
#     ierr = EOSEval(composite->eos[composite->solid_slot], P, T, &eval2);
#     CHKERRQ(ierr);
#   }

#   /* blend mixed phase with single phase, across phase boundary */
#   eval->alpha = combine_matprop(smth, eval->alpha, eval2.alpha);
#   eval->rho = combine_matprop(smth, eval->rho, eval2.rho);
#   eval->T = combine_matprop(smth, eval->T, eval2.T);
#   eval->Cp = combine_matprop(smth, eval->Cp, eval2.Cp);
#   eval->dTdPs = combine_matprop(smth, eval->dTdPs, eval2.dTdPs);
#   eval->cond = combine_matprop(smth, eval->cond, eval2.cond);
#   eval->log10visc = combine_matprop(smth, eval->log10visc, eval2.log10visc);

#   PetscFunctionReturn(0);
# }


# PetscScalar get_smoothing(PetscScalar smooth_width, PetscScalar gphi)
# {
#     /* get smoothing across phase boundaries for a two phase composite */

#     PetscScalar smth;

#     /* no smoothing */
#     if (smooth_width == 0.0)
#     {
#         smth = 1.0; // mixed phase only
#         if ((gphi < 0.0) || (gphi > 1.0))
#         {
#             smth = 0.0; // single phase only
#         }
#     }

#     /* tanh smoothing */
#     else
#     {
#         if (gphi > 0.5)
#         {
#             smth = 1.0 - tanh_weight(gphi, 1.0, smooth_width);
#         }
#         else
#         {
#             smth = tanh_weight(gphi, 0.0, smooth_width);
#         }
#     }

#     return smth;
# }

# PetscScalar tanh_weight(PetscScalar qty, PetscScalar threshold, PetscScalar width)
# {
#     /* tanh weight for viscosity profile and smoothing */

#     PetscScalar fwt, z;

#     z = (qty - threshold) / width;
#     fwt = 0.5 * (1.0 + PetscTanhScalar(z));
#     return fwt;
# }
