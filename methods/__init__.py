
"""
    This package contains implementations of the MF algorithms and seeding methods.

"""

import mf
import seeding

def list_mf_methods():
    """Return list of implemented MF methods."""
    return [name for name in mf.methods] + [None]

def list_seeding_methods():
    """Return list of implemented seeding methods."""
    return [name for name in seeding.methods] + [None]