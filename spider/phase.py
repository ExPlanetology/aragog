"""A phase defines EOS and transport properties.

See the LICENSE file for licensing information.
"""

from __future__ import annotations

import logging
from configparser import SectionProxy
from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING, Callable, Protocol, Self

import numpy as np

from spider.interfaces import PropertyABC

if TYPE_CHECKING:
    from spider.core import Scalings

logger: logging.Logger = logging.getLogger(__name__)


def ensure_size_equal_to_temperature(
    func: Callable[[ConstantProperty, np.ndarray, np.ndarray], float]
) -> Callable:
    """A decorator to ensure that the returned array is the same size as the temperature array.

    This is necessary when a phase is specified with constant properties that should be applied
    across the entire temperature and pressure range.
    """

    def wrapper(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """Wrapper

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            The quantity as an array with the same length as the temperature array.
        """
        result: np.ndarray = func(self, temperature, pressure) * np.ones_like(temperature)

        return result

    return wrapper


@dataclass(kw_only=True, frozen=True)
class ConstantProperty(PropertyABC):
    """A property with a constant value

    Args:
        name: Name of the property
        value: The constant value

    Attributes:
        See Args
    """

    value: float

    @ensure_size_equal_to_temperature
    def get_value(self, temperature: np.ndarray, pressure: np.ndarray) -> float:
        """Returns the constant value. See base class."""
        del temperature
        del pressure
        return self.value  # The decorator ensures return type is np.ndarray.


@dataclass
class PhaseStateStaggered:
    """Stores the state (material properties) of a phase at the staggered nodes.

    This only evaluates the necessary quantities to solve the system of equations to avoid
    unnecessary function calls that may slow down the code.

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

    This minimises the number of function evaluations to avoid slowing down the code.

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
    _dTdrs: np.ndarray = field(init=False)
    _kinematic_viscosity: np.ndarray = field(init=False)

    def update(self, temperature: np.ndarray, pressure: np.ndarray) -> None:
        """Updates the state.

        The order of evaluation matters.

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
        self._dTdrs = (
            -self.gravitational_acceleration
            * self.thermal_expansivity
            * temperature
            / self.heat_capacity
        )
        self._kinematic_viscosity = self.viscosity / self.density

    @property
    def dTdrs(self) -> np.ndarray:
        return self._dTdrs

    @property
    def kinematic_viscosity(self) -> np.ndarray:
        return self._kinematic_viscosity


class PhaseEvaluatorProtocol(Protocol):
    def density(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        ...

    def gravitational_acceleration(
        self, temperature: np.ndarray, pressure: np.ndarray
    ) -> np.ndarray:
        ...

    def heat_capacity(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        ...

    def thermal_conductivity(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        ...

    def thermal_expansivity(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        ...

    def viscosity(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        ...


@dataclass(frozen=True)
class PhaseEvaluator(PhaseEvaluatorProtocol):
    """Contains the objects to evaluate the EOS and transport properties of a phase.

    Args:
        scalings: Scalings
        name: Name of the phase
        density: To evaluate density at temperature and pressure
        gravitational_acceleration: To evaluate gravitational acceleration
        heat_capacity: To evaluate heat capacity
        thermal_conductivity: To evaluate thermal conductivity
        thermal_expansivity: To evaluate thermal expansivity
        viscosity: To evaluate viscosity
        phase_boundary: To evaluate phase boundary

    Attributes:
        See Args.
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
            scalings: Scalings for the numerical problem
            config: A configuration section with phase data

        Returns:
            A PhaseEvaluator
        """
        init_dict: dict[str, PropertyABC] = {}
        for key, value in section.items():
            try:
                value_float: float = float(value)
                value_float /= getattr(scalings, key)
                logger.debug("%s (%s) is a number = %f", key, section.name, value_float)
                init_dict[key] = ConstantProperty(name=key, value=value_float)

            # TODO: Add other tries to identify 1-D or 2-D lookup data.

            except TypeError:
                raise

        return cls(scalings, name, **init_dict)


# TODO: Define properties along melting curves


@dataclass(frozen=True)
class CompositeMeltFraction(PropertyABC):
    """Melt fraction for the composite"""

    _liquid: PhaseEvaluator
    _solid: PhaseEvaluator
    name: str = field(init=False, default="melt_fraction")

    def get_value(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        liquidus: np.ndarray = self._liquid.phase_boundary.get_value(temperature, pressure)
        solidus: np.ndarray = self._solid.phase_boundary.get_value(temperature, pressure)
        fusion: np.ndarray = liquidus - solidus
        melt_fraction: np.ndarray = (temperature - solidus) / fusion
        melt_fraction = np.clip(melt_fraction, 0, 1)

        return melt_fraction


@dataclass(frozen=True)
class CompositeDensity(PropertyABC):
    """Density for the composite

    Volume additivity
    """

    _liquid: PhaseEvaluator
    _solid: PhaseEvaluator
    _: KW_ONLY
    _melt_fraction: PropertyABC
    name: str = field(init=False, default="density")

    def get_value(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        melt_fraction: np.ndarray = self._melt_fraction.get_value(temperature, pressure)
        density: np.ndarray = melt_fraction / self._liquid.density.get_value(temperature, pressure)
        density += (1 - melt_fraction) / self._solid.density.get_value(temperature, pressure)

        return 1 / density


@dataclass(frozen=True)
class CompositePorosity(PropertyABC):
    """Porosity of the composite"""

    _liquid: PhaseEvaluator
    _solid: PhaseEvaluator
    _: KW_ONLY
    _density: PropertyABC
    name: str = field(init=False, default="porosity")

    def get_value(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        density: np.ndarray = self._density.get_value(temperature, pressure)
        solid_density: np.ndarray = self._solid.density.get_value(temperature, pressure)
        liquid_density: np.ndarray = self._liquid.density.get_value(temperature, pressure)
        porosity: np.ndarray = (solid_density - density) / (solid_density - liquid_density)

        return porosity


@dataclass(frozen=True)
class CompositeThermalExpansivity(PropertyABC):
    """Thermal expansivity of the composite

    Solomatov (2007), Treatise on Geophysics, Eq. 3.3

    The first term is not included because it is small compared to the latent heat term
    """

    _liquid: PhaseEvaluator
    _solid: PhaseEvaluator
    _: KW_ONLY
    _density: PropertyABC
    name: str = field(init=False, default="density")

    def get_value(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        density: np.ndarray = self._density.get_value(temperature, pressure)
        solid_density: np.ndarray = self._solid.density.get_value(temperature, pressure)
        liquid_density: np.ndarray = self._liquid.density.get_value(temperature, pressure)
        liquidus: np.ndarray = self._liquid.phase_boundary.get_value(temperature, pressure)
        solidus: np.ndarray = self._solid.phase_boundary.get_value(temperature, pressure)
        fusion: np.ndarray = liquidus - solidus
        thermal_expansivity: np.ndarray = (solid_density - liquid_density) / fusion / density

        return thermal_expansivity


@dataclass
class CompositePhaseEvaluator(PhaseEvaluatorProtocol):
    """Contains the objects to evaluate the EOS and transport properties of a two-phase mixture"""

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


# Copied from C SPIDER
#   eval->P = P;
#   eval->T = T;

#   /* these are strictly only valid for the mixed phase region, and not for general P and T
#      conditions */
#   /* unsure what the best approach is here.  The following functions are highly modular,
#      but I think it slows the code down a lot since many of the functions repeat the same lookups
#      It would reduce the modularity, but for speed the better option would be to have an
#      aggregate function that only evaluates things once.  This would be trivial to implement. */

#   ierr = EOSCompositeGetTwoPhaseLiquidus(eos, P, &liquidus);
#   CHKERRQ(ierr);
#   ierr = EOSCompositeGetTwoPhaseSolidus(eos, P, &solidus);
#   CHKERRQ(ierr);
#   eval->fusion = liquidus - solidus;
#   eval->fusion_curve = solidus + 0.5 * eval->fusion;
#   gphi = (T - solidus) / eval->fusion;
#   eval->phase_fraction = gphi;

#   /* truncation */
#   if (eval->phase_fraction > 1.0)
#   {
#     eval->phase_fraction = 1.0;
#   }
#   if (eval->phase_fraction < 0.0)
#   {
#     eval->phase_fraction = 0.0;
#   }

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

#   /* Rho */
#   /* Volume additivity */
#   eval->rho = eval->phase_fraction * (1.0 / eval_melt.rho) + (1 - eval->phase_fraction) * (1.0 / eval_solid.rho);
#   eval->rho = 1.0 / (eval->rho);

#   /* porosity */
#   /* i.e. volume fraction occupied by the melt */
#   eval->porosity = (eval_solid.rho - eval->rho) / (eval_solid.rho - eval_melt.rho);

#   /* Alpha */
#   /* positive for MgSiO3 since solid rho > melt rho.  But may need to adjust for compositional
#      effects */
#   /* Solomatov (2007), Treatise on Geophysics, Eq. 3.3 */
#   /* The first term is not included because it is small compared to the latent heat term */
#   eval->alpha = (eval_solid.rho - eval_melt.rho) / eval->fusion / eval->rho;

#   /* dTdPs */
#   /* Solomatov (2007), Treatise on Geophysics, Eq. 3.2 */
#   eval->dTdPs = eval->alpha * eval->T / (eval->rho * eval->Cp);

#   /* Conductivity */
#   /* Linear mixing by phase fraction, for lack of better knowledge about how conductivities could
#     be combined */
#   eval->cond = eval->phase_fraction * eval_melt.cond;
#   eval->cond += (1.0 - eval->phase_fraction) * eval_solid.cond;

#   /* Viscosity */
#   ierr = EOSEvalSetViscosity(composite->eos[composite->liquidus_slot], &eval_melt);
#   CHKERRQ(ierr);
#   ierr = EOSEvalSetViscosity(composite->eos[composite->solidus_slot], &eval_solid);
#   CHKERRQ(ierr);
#   fwt = tanh_weight(eval->phase_fraction, composite->phi_critical, composite->phi_width);
#   eval->log10visc = fwt * eval_melt.log10visc + (1.0 - fwt) * eval_solid.log10visc;
