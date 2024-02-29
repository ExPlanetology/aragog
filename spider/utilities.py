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
"""Utilities"""

from __future__ import annotations

from cProfile import Profile
from functools import wraps
from pathlib import Path
from pstats import SortKey, Stats
from typing import Any


def profile_decorator(func):
    """Decorator to profile a function"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        with Profile() as profile:
            result = func(*args, **kwargs)
        stats = Stats(profile).strip_dirs().sort_stats(SortKey.CALLS)
        stats.print_stats()
        return result

    return wrapper


def is_file(value: Any) -> bool:
    """Checks if value is a file.

    Args:
        value: Object to be checked

    Returns:
        True if the value is a file, otherwise False
    """
    if isinstance(value, (str, Path)):
        return Path(value).is_file()
    return False


def is_number(value: Any) -> bool:
    """Checks if value is a number.

    Args:
        value: Object to be checked

    Returns:
        True if the value is a number, otherwise False
    """
    try:
        float(value)
        return True
    except ValueError:
        return False
