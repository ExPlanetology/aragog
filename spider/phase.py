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
from dataclasses import Field, dataclass, field, fields
from typing import Protocol

import numpy as np

from spider.interfaces import (
    ConstantProperty,
    LookupProperty1D,
    LookupProperty2D,
    PropertyABC,
)
from spider.parser import mesh, phase, phase_mixed
from spider.utilities import is_file, is_number, tanh_weight

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
    gravitational_acceleration: float = field(init=False)
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

    def gravitational_acceleration(self, temperature: np.ndarray, pressure: np.ndarray) -> float:
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


class SinglePhaseEvaluator(PhaseEvaluatorProtocol):
    """Contains the objects to evaluate the EOS and transport properties of a phase.

    Args:
        settings: phase
        mesh: mesh
    """

    # For typing
    _density: PropertyABC
    _gravitational_acceleration: PropertyABC
    _heat_capacity: PropertyABC
    _thermal_conductivity: PropertyABC
    _thermal_expansivity: PropertyABC
    _viscosity: PropertyABC

    def __init__(self, settings: phase, mesh_: mesh):
        self._settings: phase = settings
        self._mesh: mesh = mesh_
        cls_fields: tuple[Field, ...] = fields(self._settings)
        for field_ in cls_fields:
            name: str = field_.name
            private_name: str = f"_{name}"
            value = getattr(self, field_.name)

            if is_number(value):
                # Numbers have already been scaled
                setattr(self, private_name, ConstantProperty(name=name, value=value))

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
                    setattr(self, private_name, LookupProperty1D(name=name, value=value_array))
                elif ndim == 3:
                    setattr(self, private_name, LookupProperty2D(name=name, value=value_array))
                else:
                    raise ValueError(f"Lookup data must have 2 or 3 dimensions, not {ndim}")
            else:
                msg: str = f"Cannot interpret value ({value}): not a number or a file"
                raise ValueError(msg)

        self._gravitational_acceleration = ConstantProperty(
            "gravitational_acceleration", value=self._mesh.gravitational_acceleration
        )

    def density(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        return self._density(temperature, pressure)

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
        return self._gravitational_acceleration(temperature, pressure)

    def heat_capacity(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        return self._heat_capacity(temperature, pressure)

    def thermal_conductivity(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        return self._thermal_conductivity(temperature, pressure)

    def thermal_expansivity(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        return self._thermal_expansivity(temperature, pressure)

    def viscosity(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        return self._viscosity(temperature, pressure)


class MixedPhaseEvaluator(PhaseEvaluatorProtocol):
    """Evaluates the EOS and transport properties of a mixed phase.

    TODO: This is only correct for the mixed phase region, since properties are evaluated at the
    liquidus and solidus values. Need a composite phase evaluator to correctly merge liquid, solid,
    and mixed at all temperature and pressure conditions.

    Args:
        settings: Mixed phase settings
        solid: Solid phase evaluator
        liquid: Liquid phase evaluator
    """

    def __init__(
        self,
        settings: phase_mixed,
        solid: PhaseEvaluatorProtocol,
        liquid: PhaseEvaluatorProtocol,
    ):
        self._settings: phase_mixed = settings
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

    def gravitational_acceleration(self, temperature: np.ndarray, pressure: np.ndarray) -> float:
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
            value_array[:, nn] /= getattr(self._settings.scalings, col_name)

        return LookupProperty1D(name=name, value=value_array)
