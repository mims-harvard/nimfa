
"""
    This package contains implementations of the MF algorithms and seeding methods.
"""

from . import factorization
from . import seeding


def list_mf_methods():
    """Return list of implemented MF methods."""
    return [name for name in factorization.methods]


def list_seeding_methods():
    """Return list of implemented seeding methods."""
    return [name for name in seeding.methods]
