"""
navette
=======

Fast S-Matrix computations and handling.
"""

# Import submodules so they are immediately available
from . import needle
from . import smatrix

# Define what gets exported if someone runs `from navette import *`
__all__ = ["needle", "smatrix"]