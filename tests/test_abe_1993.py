"""Tests for Abe (1993) model.

See the LICENSE file for licensing information.

"""

from spider import __version__, debug_logger

rtol: float = 1.0e-8
atol: float = 1.0e-8

debug_logger()


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"
