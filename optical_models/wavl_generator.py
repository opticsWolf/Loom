# -*- coding: utf-8 -*-
"""
Loom: Weaving the mathematics of light in thin film systems
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""

import numpy as np
from numba import njit #, prange
from typing import List, Tuple

__all__ = ["generate_wavelength_array_from_steps"]

def generate_wavelength_array_from_steps(points: List[Tuple[float, float]]) -> np.ndarray:
    """
    Generate wavelengths from ranges defined by (start, step) tuples and final point.

    Args:
        points: List of tuples where each defines a range (start, step) or final point (end, _).
            The last tuple only specifies the end wavelength; its second value is ignored.

    Returns:
        1D numpy array of concatenated wavelengths from all ranges and final point.
    """
    if len(points) < 2:
        raise ValueError("At least two points are needed to define ranges.")

    starts = np.array([p[0] for p in points], dtype=np.float64)
    steps = np.array([p[1] for p in points[:-1]], dtype=np.float64)  # skip last step

    return _generate_wavelength_array_from_steps_numba(starts, steps)
     
@njit (cache=True)#, parallel=True)
def _generate_wavelength_array_from_steps_numba(starts: np.ndarray,
                                               steps: np.ndarray) -> np.ndarray:
    """
    Efficiently generates wavelength array from ranges using Numba optimization.
    
    Args:
        starts: Start wavelengths and final endpoint.
        steps: Step sizes for each range.
    
    Returns:
        1D numpy float64 array of concatenated wavelengths.
    """
    total_points = 0

    # First, calculate total size
    for i in range(len(steps)):
        start = starts[i]
        end = starts[i + 1]
        step = steps[i]
        count = int(np.floor((end - start) / step))
        total_points += count

    total_points += 1  # include the last point

    wavelengths = np.empty(total_points, dtype=np.float64)
    idx = 0

    # Generate ranges
    for i in range(len(steps)):
        start = starts[i]
        end = starts[i + 1]
        step = steps[i]
        n_steps = int(np.floor((end - start) / step))
        for j in range(n_steps):
            wavelengths[idx] = start + j * step
            idx += 1

    # Add the last point explicitly
    wavelengths[idx] = starts[-1]
    return wavelengths
