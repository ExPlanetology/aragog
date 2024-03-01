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
from configparser import SectionProxy
from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from spider.interfaces import (
    ConstantProperty,
    LookupProperty1D,
    LookupProperty2D,
    PhaseEvaluatorProtocol,
    PropertyABC,
)
from spider.utilities import is_file, is_number

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

if TYPE_CHECKING:
    from spider.core import Scalings

logger: logging.Logger = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class PhaseEvaluator:
    """Contains the objects to evaluate the EOS and transport properties of a phase.

    Args:
        scalings: Scalings
        name: Name of the phase
        density: Object to evaluate density
        gravitational_acceleration: Object to evaluate gravitational acceleration
        heat_capacity: Object to evaluate heat capacity
        thermal_conductivity: Object to evaluate thermal conductivity
        thermal_expansivity: Object to evaluate thermal expansivity
        viscosity: Object to evaluate viscosity
        phase_boundary: Object to evaluate phase boundary

    Attributes:
        scalings: Scalings
        name: Name of the phase
        density: Object to evaluate density at temperature and pressure
        gravitational_acceleration: Object to evaluate gravitational acceleration
        heat_capacity: Object to evaluate heat capacity
        thermal_conductivity: Object to evaluate thermal conductivity
        thermal_expansivity: Object to evaluate thermal expansivity
        viscosity: Object to evaluate viscosity
        phase_boundary: Object to evaluate phase boundary
    """

    scalings: Scalings
    name: str
    _: KW_ONLY
    density: PropertyABC
    gravitational_acceleration: PropertyABC
    heat_capacity: PropertyABC
    thermal_conductivity: PropertyABC
    thermal_expansivity: PropertyABC
    viscosity: PropertyABC
    phase_boundary: PropertyABC

    @classmethod
    def from_configuration(cls, scalings: Scalings, name: str, *, section: SectionProxy) -> Self:
        """Creates a class instance from a configuration section.

        Args:
            name: Name of the phase
            scalings: Scalings
            config: A configuration section with phase data

        Returns:
            A PhaseEvaluator
        """
        init_dict: dict[str, PropertyABC] = {}
        for key, value in section.items():

            if is_number(value):
                value_float: float = float(value)
                value_float /= getattr(scalings, key)
                logger.debug("%s (%s) is a number = %f", key, section.name, value_float)
                init_dict[key] = ConstantProperty(name=key, value=value_float)

            elif is_file(value):
                with open(value, encoding="utf-8") as infile:
                    logger.debug("%s (%s) is a file = %s", key, section.name, value)
                    header = infile.readline()
                    col_names = header[1:].split()
                value_array: np.ndarray = np.loadtxt(value, ndmin=2)
                logger.debug("before scaling, value_array = %s", value_array)
                for nn, col_name in enumerate(col_names):
                    logger.info("Scaling %s from %s", col_name, value)
                    value_array[:, nn] /= getattr(scalings, key)
                logger.debug("after scaling, value_array = %s", value_array)
                ndim = value_array.shape[1]
                logger.debug("ndim = %d", ndim)
                if ndim == 2:
                    init_dict[key] = LookupProperty1D(name=key, value=value_array)
                elif ndim == 3:
                    init_dict[key] = LookupProperty2D(name=key, value=value_array)
                else:
                    raise ValueError(f"Lookup data must have 2 or 3 dimensions, not {ndim}")
            else:
                msg: str = f"Cannot interpret value ({value}): not a number or a file"
                raise ValueError(msg)

        return cls(scalings, name, **init_dict)


# region Composite phase


@dataclass
class CompositeMeltFraction(PropertyABC):
    """Melt fraction of the composite

    The melt fraction is always between zero and one.
    """

    _liquid: PhaseEvaluator
    _solid: PhaseEvaluator
    name: str = field(init=False, default="melt_fraction")

    @override
    def _get_value(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        liquidus_temperature: np.ndarray = self._liquid.phase_boundary(temperature, pressure)
        solidus_temperature: np.ndarray = self._solid.phase_boundary(temperature, pressure)
        delta_fusion_temperature: np.ndarray = liquidus_temperature - solidus_temperature
        melt_fraction: np.ndarray = (temperature - solidus_temperature) / delta_fusion_temperature
        melt_fraction = np.clip(melt_fraction, 0, 1)

        return melt_fraction


@dataclass
class CompositeConductivity(PropertyABC):
    """Thermal conductivity of the composite by linear mixing"""

    _liquid: PhaseEvaluator
    _solid: PhaseEvaluator
    _: KW_ONLY
    _melt_fraction: PropertyABC
    name: str = field(init=False, default="conductivity")

    @override
    def _get_value(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        melt_fraction: np.ndarray = self._melt_fraction(temperature, pressure)
        conductivity: np.ndarray = melt_fraction * self._liquid.thermal_conductivity(
            temperature, pressure
        )
        conductivity += (1 - melt_fraction) * self._solid.thermal_conductivity(
            temperature, pressure
        )

        return conductivity


@dataclass
class CompositeDensity(PropertyABC):
    """Density of the composite computed by volume additivity"""

    _liquid: PhaseEvaluator
    _solid: PhaseEvaluator
    _: KW_ONLY
    _melt_fraction: PropertyABC
    name: str = field(init=False, default="density")

    @override
    def _get_value(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        liquidus_temperature: np.ndarray = self._liquid.phase_boundary(temperature, pressure)
        solidus_temperature: np.ndarray = self._solid.phase_boundary(temperature, pressure)
        melt_fraction: np.ndarray = self._melt_fraction(temperature, pressure)
        density_inverse: np.ndarray = melt_fraction / self._liquid.density(
            liquidus_temperature, pressure
        )
        density_inverse += (1 - melt_fraction) / self._solid.density(solidus_temperature, pressure)
        density: np.ndarray = 1 / density_inverse

        return density


@dataclass
class CompositePorosity(PropertyABC):
    """Porosity of the composite, that is the volume fraction occupied by the melt"""

    _liquid: PhaseEvaluator
    _solid: PhaseEvaluator
    _: KW_ONLY
    _density: PropertyABC
    name: str = field(init=False, default="porosity")

    @override
    def _get_value(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        liquidus_temperature: np.ndarray = self._liquid.phase_boundary(temperature, pressure)
        solidus_temperature: np.ndarray = self._solid.phase_boundary(temperature, pressure)
        density: np.ndarray = self._density(temperature, pressure)
        liquidus_density: np.ndarray = self._liquid.density(liquidus_temperature, pressure)
        solidus_density: np.ndarray = self._solid.density(solidus_temperature, pressure)
        porosity: np.ndarray = (solidus_density - density) / (solidus_density - liquidus_density)

        return porosity


@dataclass
class CompositeThermalExpansivity(PropertyABC):
    """Thermal expansivity of the composite :cite:p:`{Equation 3.3,}SOLO07`

    The first term is not included because it is small compared to the latent heat term
    """

    _liquid: PhaseEvaluator
    _solid: PhaseEvaluator
    _: KW_ONLY
    _density: PropertyABC
    name: str = field(init=False, default="density")

    @override
    def _get_value(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        liquidus_temperature: np.ndarray = self._liquid.phase_boundary(temperature, pressure)
        solidus_temperature: np.ndarray = self._solid.phase_boundary(temperature, pressure)
        density: np.ndarray = self._density(temperature, pressure)
        liquidus_density: np.ndarray = self._liquid.density(liquidus_temperature, pressure)
        solidus_density: np.ndarray = self._solid.density(solidus_temperature, pressure)
        delta_fusion_temperature: np.ndarray = liquidus_temperature - solidus_temperature
        thermal_expansivity: np.ndarray = (
            (solidus_density - liquidus_density) / delta_fusion_temperature / density
        )

        return thermal_expansivity


@dataclass
class CompositePhaseEvaluator:
    """Contains the objects to evaluate the EOS and transport properties of a two-phase mixture

    TODO: Need to finish this.
    """

    phases: dict[str, PhaseEvaluator]
    density: PropertyABC = field(init=False)
    melt_fraction: PropertyABC = field(init=False)
    porosity: PropertyABC = field(init=False)
    name: str = field(init=False, default="composite")

    def __post_init__(self):
        self.melt_fraction = CompositeMeltFraction(self.liquid, self.solid)
        self.density = CompositeDensity(self.liquid, self.solid, _melt_fraction=self.melt_fraction)
        self.porosity = CompositePorosity(self.liquid, self.solid, _density=self.density)

    @property
    def liquid(self) -> PhaseEvaluator:
        """Liquid phase evaluator"""
        return self.phases["liquid"]

    @property
    def solid(self) -> PhaseEvaluator:
        """Solid phase evaluator"""
        return self.phases["solid"]


# endregion

# Copied from C SPIDER

#   ierr = EOSCompositeGetTwoPhaseLiquidus(eos, P, &liquidus);
#   CHKERRQ(ierr);
#   ierr = EOSCompositeGetTwoPhaseSolidus(eos, P, &solidus);
#   CHKERRQ(ierr);
#   eval->fusion = liquidus - solidus;
#   eval->fusion_curve = solidus + 0.5 * eval->fusion;
#   gphi = (T - solidus) / eval->fusion;


#   /* properties along melting curves */
#   ierr = EOSEval(composite->eos[composite->liquidus_slot], P, liquidus, &eval_melt);
#   CHKERRQ(ierr);
#   ierr = EOSEval(composite->eos[composite->solidus_slot], P, solidus, &eval_solid);
#   CHKERRQ(ierr);

#   /* enthalpy of fusion */
#   eval->enthalpy_of_fusion = eval->fusion_curve * composite->entropy_of_fusion;

#   /* Cp */
#   /* Solomatov (2007), Treatise on Geophysics, Eq. 3.4 */
#   /* The first term is not included because it is small compared to the latent heat term */
#   eval->Cp = eval->enthalpy_of_fusion; // enthalpy change upon melting.
#   eval->Cp /= eval->fusion;

#   /* dTdPs */
#   /* Solomatov (2007), Treatise on Geophysics, Eq. 3.2 */
#   eval->dTdPs = eval->alpha * eval->T / (eval->rho * eval->Cp);

#   /* Viscosity */
#   ierr = EOSEvalSetViscosity(composite->eos[composite->liquidus_slot], &eval_melt);
#   CHKERRQ(ierr);
#   ierr = EOSEvalSetViscosity(composite->eos[composite->solidus_slot], &eval_solid);
#   CHKERRQ(ierr);
#   fwt = tanh_weight(eval->phase_fraction, composite->phi_critical, composite->phi_width);
#   eval->log10visc = fwt * eval_melt.log10visc + (1.0 - fwt) * eval_solid.log10visc;
