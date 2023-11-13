"""Interfaces

See the LICENSE file for licensing information.
"""

from __future__ import annotations

import inspect
from ast import literal_eval
from configparser import ConfigParser, SectionProxy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Self


@dataclass
class DataclassFromConfiguration:
    """A dataclass that can source its attributes from a configuration section"""

    @classmethod
    def from_configuration(cls, *args, config: SectionProxy) -> Self:
        """Creates a class instance from a configuration section.

        Args:
            *args: Positional arguments
            config: A configuration section

        Returns:
            A dataclass with its attributes populated
        """
        init_dict: dict[str, Any] = {
            k: config.getany(k) for k in config.keys() if k in inspect.signature(cls).parameters
        }
        return cls(*args, **init_dict)


class MyConfigParser(ConfigParser):
    """A configuration parser with some default options

    Args:
        *filenames: Filenames of one or several configuration files
    """

    getpath: Callable[..., Path]  # For typing.

    def __init__(self, *filenames):
        kwargs: dict = {
            "comment_prefixes": ("#",),
            "converters": {"path": Path, "any": lambda x: literal_eval(x)},
        }
        super().__init__(**kwargs)
        self.read(filenames)
