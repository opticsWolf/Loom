"""
navette
=======

Python bindings for the Navette unified color engine (Rust parity port).
"""

# Import the compiled Rust extension so it is immediately available
# when someone imports the 'navette' namespace.
from . import color

# Define what gets exported if someone runs `from navette import *`
__all__ = ["color"]