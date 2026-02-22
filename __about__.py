# -*- coding: utf-8 -*-
# Loom: Weaving the mathematics of light in thin film systems.
#
# Copyright (c) 2026 opticsWolf
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Project metadata for the opticsWolf Loom.
"""

from typing import Final

# Metadata Definitions
__title__: Final[str] = "Loom"
__description__: Final[str] = (
    "A high-performance optical engine utilizing a Scattering Matrix "
    "algorithm for stable simulation of light in stratified media."
)
__version__: Final[str] = "0.1.0"
__author__: Final[str] = "opticsWolf"
__license__: Final[str] = "LGPL-3.0-or-later"
__copyright__: Final[str] = "Copyright (c) 2026 opticsWolf"

def metadata_summary() -> dict[str, str]:
    """Returns a dictionary of project metadata for introspection."""
    return {
        "title": __title__,
        "version": __version__,
        "license": __license__,
        "description": __description__,
        "copyright": __copyright__,
    }