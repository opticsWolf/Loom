"""
navette
=======

Fast S-Matrix computations and handling.
"""

# Import the compiled Rust extension so it is immediately available
from . import smatrix

# Define what gets exported if someone runs `from navette import *`
__all__ = ["smatrix"]