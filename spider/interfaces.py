"""Interfaces

See the LICENSE file for licensing information.
"""

from __future__ import annotations

import inspect
import logging
from abc import ABC, abstractmethod
from configparser import SectionProxy
from dataclasses import dataclass
from typing import Any, Self

import numpy as np

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class DataclassFromConfiguration:
    """A dataclass that can source its attributes from a configuration section"""

    @classmethod
    def from_configuration(cls, *args, section: SectionProxy) -> Self:
        """Creates an instance from a configuration section.

        This reads the configuration data and sources the attributes from this data and performs
        type conversations.

        Args:
            *args: Positional arguments to pass through to the constructor
            config: A configuration section

        Returns:
            A dataclass with its attributes populated
        """
        init_dict: dict[str, Any] = {
            k: section.getany(k) for k in section.keys() if k in inspect.signature(cls).parameters
        }
        return cls(*args, **init_dict)


@dataclass
class ScaledDataclassFromConfiguration(DataclassFromConfiguration):
    """A dataclass that requires its attributes to be scaled."""

    def __post_init__(self):
        self.scale_attributes()

    @abstractmethod
    def scale_attributes(self) -> None:
        """Scales the attributes"""


@dataclass(kw_only=True, frozen=True)
class PropertyABC(ABC):
    """A property whose value can be evaluated at temperature and pressure.

    Args:
        name: Name of the property

    Attributes:
        name: Name of the property
    """

    name: str

    @abstractmethod
    def get_value(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """Computes the property value at temperature and pressure.

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            The property value evaluated at temperature and pressure.
        """

    def __call__(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        return self.get_value(temperature, pressure)
