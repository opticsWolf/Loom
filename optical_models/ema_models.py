# -*- coding: utf-8 -*-
"""
Loom: Weaving the mathematics of light in thin film systems
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""

import time
from typing import Tuple
import numpy as np
import numpy.typing as npt
from numba import njit, prange

__all__ = [
    "lichtenecker_eps",
    "looyenga_eps",
    "general_power_law_eps",
    "maxwell_garnett_eps",
    "bruggeman_eps",
    "mori_tanaka_eps",
    "wiener_bounds",
    "hashin_shtrikman_bounds",
    "roughness_interface_eps",
]


# OPTIMIZATION NOTE:
# Removed explicit Numba signatures to allow flexible default argument usage.
# Numba will now lazy-compile based on the input types provided at runtime.
# fastmath=True enabled globally for hardware-accelerated exp/log/pow.

@njit(cache=True, fastmath=True, parallel=True)
def lichtenecker_eps(
    n_i: npt.NDArray[np.complex128],
    n_h: npt.NDArray[np.complex128],
    f: float,
) -> npt.NDArray[np.complex128]:
    """Calculates Lichtenecker effective permittivity using logarithmic mixing.
    
    The Lichtenecker mixing rule (also known as the logarithmic mixture law) is
    a mathematical generalization rather than a derivation from first physical 
    principles. It sits between the Wiener upper bound (arithmetic mean, 
    parallel capacitors) and lower bound (harmonic mean, series capacitors).
    
    Formula:
        ε_eff = exp( f * ln(ε_i) + (1 - f) * ln(ε_h) )

    Args:
        n_i: Inclusion refractive index (complex128 array).
        n_h: Host refractive index (complex128 array).
        f: Volume fraction of inclusion (scalar float, 0 <= f <= 1).

    Returns:
        Effective permittivity (epsilon) as a complex128 array.
    """
    eps_i = n_i * n_i
    eps_h = n_h * n_h
    
    # Logarithmic mixing: f * ln(e_i) + (1-f) * ln(e_h)
    log_eps_eff = f * np.log(eps_i) + (1.0 - f) * np.log(eps_h)
    
    return np.exp(log_eps_eff)


@njit(cache=True, fastmath=True, parallel=True)
def looyenga_eps(
    n_i: npt.NDArray[np.complex128],
    n_h: npt.NDArray[np.complex128],
    f: float,
) -> npt.NDArray[np.complex128]:
    """Calculates Looyenga effective permittivity (Landau-Lifshitz-Looyenga).
    
    Formula:
        (ε_eff)^(1/3) = f * (ε_i)^(1/3) + (1 - f) * (ε_h)^(1/3)

    Args:
        n_i: Inclusion refractive index (complex128 array).
        n_h: Host refractive index (complex128 array).
        f: Volume fraction of inclusion (scalar float, 0 <= f <= 1).

    Returns:
        Effective permittivity (epsilon) as a complex128 array.
    """
    eps_i = n_i * n_i
    eps_h = n_h * n_h
    
    # Pre-calculate powers (1/3)
    power = 1.0 / 3.0
    cbrt_eps_i = eps_i ** power
    cbrt_eps_h = eps_h ** power
    
    # Linear interpolation of cube roots
    cbrt_eps_eff = f * cbrt_eps_i + (1.0 - f) * cbrt_eps_h
    
    return cbrt_eps_eff ** 3.0


@njit(cache=True, fastmath=True, parallel=True)
def general_power_law_eps(
    n_i: npt.NDArray[np.complex128],
    n_h: npt.NDArray[np.complex128],
    f: float,
    alpha: float = 0.5,
) -> npt.NDArray[np.complex128]:
    """Calculates the General Power Law (Birchak) effective permittivity.
    
    This generalizes several other models by tuning the exponent alpha.
    
    Common Alpha Values:
        1.0  : Linear mixing of Permittivity (Wiener Upper Bound)
        0.5  : Linear mixing of Refractive Index (Birchak)
        1/3  : Looyenga (Landau-Lifshitz-Looyenga)
        0.0  : Lichtenecker (Limit as alpha -> 0)
       -1.0  : Inverse mixing of Permittivity (Wiener Lower Bound)

    Formula:
        ε_eff = ( f * ε_i^α + (1-f) * ε_h^α )^(1/α)

    Args:
        n_i: Inclusion refractive index.
        n_h: Host refractive index.
        f: Volume fraction (scalar float).
        alpha: Power exponent. Default 0.5 (Refractive Index mixing).

    Returns:
        Effective permittivity (epsilon) as a complex128 array.
    """
    eps_i = n_i * n_i
    eps_h = n_h * n_h
    
    # Handle alpha close to 0 (Lichtenecker case) to avoid division by zero
    if abs(alpha) < 1e-6:
        log_eps_eff = f * np.log(eps_i) + (1.0 - f) * np.log(eps_h)
        return np.exp(log_eps_eff)

    # General Power Law
    pow_eps_i = eps_i ** alpha
    pow_eps_h = eps_h ** alpha
    
    pow_eps_eff = f * pow_eps_i + (1.0 - f) * pow_eps_h
    
    return pow_eps_eff ** (1.0 / alpha)


@njit(cache=True, fastmath=True)
def maxwell_garnett_eps(
    n_i: npt.NDArray[np.complex128],
    n_h: npt.NDArray[np.complex128],
    f: float,
) -> npt.NDArray[np.complex128]:
    """Calculates Maxwell-Garnett effective permittivity.

    Strictly valid only for dilute mixtures (f << 1). Assumes spherical inclusions.
    
    Formula:
        ε_eff = ε_h * [ (ε_i + 2ε_h + 2f(ε_i - ε_h)) / (ε_i + 2ε_h - f(ε_i - ε_h)) ]

    Args:
        n_i: Inclusion refractive index (complex128 array).
        n_h: Host refractive index (complex128 array).
        f: Volume fraction of inclusion (scalar float, 0 <= f <= 1).

    Returns:
        Effective permittivity (epsilon) as a complex128 array.
    """
    eps_i = n_i * n_i
    eps_h = n_h * n_h
    
    eps_diff = eps_i - eps_h
    eps_h_2 = 2.0 * eps_h
    
    numerator = eps_i + eps_h_2 + 2.0 * f * eps_diff
    denominator = eps_i + eps_h_2 - f * eps_diff
    
    return eps_h * (numerator / denominator)


@njit(cache=True, fastmath=True, parallel=True)
def bruggeman_eps(
    n_i: npt.NDArray[np.complex128],
    n_h: npt.NDArray[np.complex128],
    f: float,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> npt.NDArray[np.complex128]:
    """Calculates Bruggeman effective permittivity via Newton-Raphson.

    Implicit Equation:
        f * (ε_i - ε_eff)/(ε_i + 2ε_eff) + (1 - f) * (ε_h - ε_eff)/(ε_h + 2ε_eff) = 0

    Args:
        n_i: Inclusion refractive index (complex128 array).
        n_h: Host refractive index (complex128 array).
        f: Volume fraction of inclusion (scalar float).
        max_iter: Max Newton iterations.
        tol: Convergence tolerance.

    Returns:
        Effective permittivity (epsilon) as a complex128 array.
    """
    eps_i = n_i * n_i
    eps_h = n_h * n_h
    
    n_len = len(eps_i)
    eps_eff = np.empty(n_len, dtype=np.complex128)
    
    # Initialization: Arithmetic mean
    for k in prange(n_len):
        eps_eff[k] = (eps_i[k] + eps_h[k]) * 0.5

    # Parallel Solver
    for i in prange(n_len):
        e_i_val = eps_i[i]
        e_h_val = eps_h[i]
        f_val = f
        inv_f = 1.0 - f_val
        
        for _ in range(max_iter):
            # Term 1: f * (e_i - e_eff) / (e_i + 2e_eff)
            num_i = e_i_val - eps_eff[i]
            den_i = e_i_val + 2.0 * eps_eff[i]
            term_i = num_i / den_i
            
            # Term 2: (1-f) * (e_h - e_eff) / (e_h + 2e_eff)
            num_h = e_h_val - eps_eff[i]
            den_h = e_h_val + 2.0 * eps_eff[i]
            term_h = num_h / den_h

            # F(e_eff)
            f_total = f_val * term_i + inv_f * term_h
            
            # Derivative F'(e_eff)
            deriv_i = (-3.0 * e_i_val) / (den_i * den_i)
            deriv_h = (-3.0 * e_h_val) / (den_h * den_h)
            
            df = f_val * deriv_i + inv_f * deriv_h
            
            # Newton Step
            delta = -f_total / (df + 1e-15)
            
            eps_eff[i] += delta
            
            # Convergence check
            if (delta.real*delta.real + delta.imag*delta.imag) < (tol*tol):
                break
                
    return eps_eff


@njit(cache=True, fastmath=True)
def mori_tanaka_eps(
    n_i: npt.NDArray[np.complex128],
    n_h: npt.NDArray[np.complex128],
    f: float,
    L: float = 0.333333333333,
) -> npt.NDArray[np.complex128]:
    """Calculates Mori-Tanaka effective permittivity for ellipsoidal inclusions.
    
    Extension of Maxwell-Garnett for shaped inclusions defined by the 
    depolarization factor L.
    
    L Values:
      0.0   : Needles / Nanowires (aligned with E-field)
      1/3   : Spheres (Mathematically equivalent to Maxwell-Garnett)
      1.0   : Discs / Platelets (Perpendicular to E-field)

    Formula:
      ε_eff = ε_h + [ f(ε_i - ε_h)ε_h ] / [ ε_h + (1-f)L(ε_i - ε_h) ]

    Args:
        n_i: Inclusion refractive index.
        n_h: Host refractive index.
        f: Volume fraction (scalar float).
        L: Depolarization factor (0.0 to 1.0). Default is 1/3 (Spheres).

    Returns:
        Effective permittivity (epsilon) as a complex128 array.
    """
    eps_i = n_i * n_i
    eps_h = n_h * n_h
    
    # Pre-calculate differences
    eps_diff = eps_i - eps_h
    
    # Mori-Tanaka Explicit Form
    # Numerator term: f * (e_i - e_h) * e_h
    numerator = f * eps_diff * eps_h
    
    # Denominator term: e_h + (1-f) * L * (e_i - e_h)
    denominator = eps_h + (1.0 - f) * L * eps_diff
    
    return eps_h + (numerator / denominator)


@njit(cache=True, fastmath=True, parallel=True)
def wiener_bounds(
    n_i: npt.NDArray[np.complex128],
    n_h: npt.NDArray[np.complex128],
    f: float,
) -> Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    """Calculates the Wiener Upper and Lower bounds."""
    eps_i = n_i * n_i
    eps_h = n_h * n_h
    inv_f = 1.0 - f

    # Upper Bound (Arithmetic)
    eps_upper = f * eps_i + inv_f * eps_h
    
    # Lower Bound (Harmonic)
    numerator = eps_i * eps_h
    denominator = f * eps_h + inv_f * eps_i
    eps_lower = numerator / denominator

    return eps_lower, eps_upper


@njit(cache=True, fastmath=True)
def hashin_shtrikman_bounds(
    n_i: npt.NDArray[np.complex128],
    n_h: npt.NDArray[np.complex128],
    f: float,
) -> Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    """Calculates Hashin-Shtrikman (HS) bounds for isotropic composites."""
    # Bound 1: Standard MG (Host is matrix)
    bound_h_matrix = maxwell_garnett_eps(n_i, n_h, f)
    
    # Bound 2: Inverted MG (Inclusion is matrix). 
    bound_i_matrix = maxwell_garnett_eps(n_h, n_i, 1.0 - f)
    
    return bound_h_matrix, bound_i_matrix


@njit(cache=True, fastmath=True, parallel=True)
def roughness_interface_eps(
    n_bottom: npt.NDArray[np.complex128],
    n_top: npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128]:
    """Calculates effective permittivity for a standard 50:50 roughness interface.

    This function is optimized for thin-film models needing a general-purpose
    interface layer. It uses the Looyenga model (Landau-Lifshitz-Looyenga) 
    with f=0.5. Looyenga is fully analytical (fast) and symmetric (stable),
    making it ideal for automated interface layer generation.
    
    Args:
        n_bottom: Refractive index of the bottom layer (complex128 array).
        n_top: Refractive index of the top layer (complex128 array).

    Returns:
        Effective permittivity (epsilon) as a complex128 array.
    """
    # Delegate to Looyenga (Analytical, Symmetric, Fast)
    return looyenga_eps(n_bottom, n_top, 0.5)


# --- Benchmarking Utilities ---

def benchmark_models(n_samples: int = 50_000, n_iterations: int = 100) -> None:
    """Runs performance benchmarks on all effective medium models."""
    rng = np.random.default_rng(42)

    # 1. Generate Input Data (Double Precision)
    print(f"\nRunning {n_iterations} iterations of {n_samples}-sample benchmarks...")
    print("Precision: complex128 / float64 (High Precision Mode)")
    
    n_i = rng.uniform(1.5, 3.0, n_samples) + 1j * rng.uniform(0.0, 0.5, n_samples)
    n_h = rng.uniform(1.0, 2.0, n_samples) + 1j * rng.uniform(0.0, 0.3, n_samples)
    
    # We use a single scalar fraction for benchmarking the new signature
    f_val = 0.5

    # JIT Warm-up (forcing compilation for Scalar inputs)
    _ = lichtenecker_eps(n_i, n_h, f_val)
    _ = looyenga_eps(n_i, n_h, f_val)
    _ = general_power_law_eps(n_i, n_h, f_val, 0.5)
    _ = maxwell_garnett_eps(n_i, n_h, f_val)
    _ = bruggeman_eps(n_i, n_h, f_val)
    _ = mori_tanaka_eps(n_i, n_h, f_val, 0.333)
    _ = roughness_interface_eps(n_i, n_h)
    _ = wiener_bounds(n_i, n_h, f_val)
    _ = hashin_shtrikman_bounds(n_i, n_h, f_val)

    timings = {}

    # 2. Execution Loop
    start_global = time.perf_counter()

    t0 = time.perf_counter()
    for _ in range(n_iterations):
        lichtenecker_eps(n_i, n_h, f_val)
    timings["Lichtenecker"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n_iterations):
        looyenga_eps(n_i, n_h, f_val)
    timings["Looyenga"] = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        general_power_law_eps(n_i, n_h, f_val, 0.5)
    timings["PowerLaw (Birchak)"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n_iterations):
        maxwell_garnett_eps(n_i, n_h, f_val)
    timings["Maxwell Garnett"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n_iterations):
        bruggeman_eps(n_i, n_h, f_val)
    timings["Bruggeman"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n_iterations):
        mori_tanaka_eps(n_i, n_h, f_val, 0.333333)
    timings["Mori-Tanaka"] = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        roughness_interface_eps(n_i, n_h)
    timings["Roughness (f=0.5)"] = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        wiener_bounds(n_i, n_h, f_val)
    timings["Wiener Bounds"] = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        hashin_shtrikman_bounds(n_i, n_h, f_val)
    timings["Hashin-Shtrikman"] = time.perf_counter() - t0

    end_global = time.perf_counter()

    # 3. Report Results
    print(f"{'Model':<20} {'Total Time (s)':>15} {'Avg Time/Call (s)':>20}")
    print("-" * 57)
    for name, t_total in timings.items():
        print(f"{name:<20} {t_total:15.6f} {t_total/n_iterations:20.8f}")
    print("-" * 57)
    print(f"{'Benchmark Overhead':<20} {end_global - start_global:.6f} seconds")


def run_verification() -> None:
    """Runs a verification table for specific material pairs."""
    metal1 = np.array([0.2 + 4.0j], dtype=np.complex128)
    diel1  = np.array([1.5 + 0.0j], dtype=np.complex128)
    diel2  = np.array([2.0 + 0.1j], dtype=np.complex128)
    metal2 = np.array([0.6 + 5.5j], dtype=np.complex128)

    fractions = [0.1, 0.3, 0.5, 0.7, 0.9]

    test_cases = [
        ("Metal–Dielectric", metal1, diel1),
        ("Dielectric-Metal", diel2, metal2),
        ("Dielectric1–Dielectric2", diel1, diel2),
        ("Dielectric2–Dielectric1", diel2, diel1),
        ("Metal–Metal", metal1, metal2),
    ]

    def fmt_n(eps_arr: npt.NDArray[np.complex128]) -> str:
        """Format sqrt(epsilon) -> n."""
        val = np.sqrt(eps_arr[0])
        return f"{val.real:.3f}+{val.imag:.3f}j"

    for label, n_a, n_b in test_cases:
        print(f"\n--- {label} ---")
        # Header covering all implemented models
        header = (f"{'f':>5} | {'Licht.':>18} | {'Looy.':>18} | {'Birch.':>18} | {'Brug.':>18} | "
                  f"{'HS1 (MG)':>18} | {'MT (Sph)':>18} | {'HS2 (Inv)':>18} | "
                  f"{'Wien(Lo)':>18} | {'Wien(Hi)':>18}")
        print(header)
        print("-" * 193)  # Expanded width to match new column count

        for f_val in fractions:
            # All functions now accept scalar f_val directly
            
            e_l = lichtenecker_eps(n_a, n_b, f_val)
            e_loo = looyenga_eps(n_a, n_b, f_val)
            e_birchak = general_power_law_eps(n_a, n_b, f_val, 0.5)
            e_bg = bruggeman_eps(n_a, n_b, f_val)
            e_hs1, e_hs2 = hashin_shtrikman_bounds(n_a, n_b, f_val)
            e_mt = mori_tanaka_eps(n_a, n_b, f_val, 1.0/3.0)
            e_w_low, e_w_high = wiener_bounds(n_a, n_b, f_val)

            # Print all columns
            print(f"{f_val:5.1f} | {fmt_n(e_l):>18} | {fmt_n(e_loo):>18} | {fmt_n(e_birchak):>18} | "
                  f"{fmt_n(e_bg):>18} | {fmt_n(e_hs1):>18} | {fmt_n(e_mt):>18} | {fmt_n(e_hs2):>18} | "
                  f"{fmt_n(e_w_low):>18} | {fmt_n(e_w_high):>18}")


if __name__ == "__main__":
    benchmark_models(n_samples=50_000, n_iterations=100)
    run_verification()