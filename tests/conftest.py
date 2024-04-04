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
"""Utilities for tests"""

import importlib.resources

import pytest

from aragog import CFG_DATA

# Tolerances to compare the test results with target output.
RTOL: float = 1.0e-8
ATOL: float = 1.0e-8


class Helper:
    """Helper class with methods to check and confirm values."""

    @staticmethod
    def get_cfg_file(filename: str):
        return importlib.resources.as_file(CFG_DATA.joinpath(filename))


@pytest.fixture(scope="module")
def helper():
    return Helper()
