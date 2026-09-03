"""
Validation tests for the partial-coherence modes of
`core_engine_rigorous_ellipsometry`.

Two invariants follow directly from the isotropic coherency-matrix decoupling:

  Test 1 — Mode B (coherency_matrix) must reproduce Mode A (front_block) on
           R, T and Psi to machine precision. Those quantities live entirely
           in the diagonal (intensity) channels, which run identical code in
           both modes; only the off-diagonal p-s coherence (-> Delta, DOP)
           changes. The test also asserts the two modes are *distinguishable*
           on DOP/Delta, so the equality above is not vacuous.

  Test 2 — For a stack with no incoherent layers (a single coherent block),
           Modes A, B and C must agree on everything, because the coherency
           off-diagonal is rank-1 (factorable into r_p*conj(r_s)) and Mode C's
           single-block forcing is a no-op. Reflection DOP must additionally
           be 1 (a single coherent block is fully polarized).

Run with:  pytest -q test_coherence_modes.py
       or:  python test_coherence_modes.py
Requires the compiled extension (`maturin develop` / `maturin build`).
"""

import numpy as np
from enum import IntEnum

# --- import the compiled engine (adjust the module path to your build) -------
try:
    from smatrix import core_engine_rigorous_ellipsometry as engine
except ImportError:  # pragma: no cover - depends on local build/layout
    from loom_matrix import core_engine_rigorous_ellipsometry as engine


class CoherenceMode(IntEnum):
    FRONT_BLOCK = 0       # Mode A (current / default)
    COHERENCY_MATRIX = 1  # Mode B (rigorous isotropic coherency)
    FULLY_COHERENT = 2    # Mode C (ignore incoherent flags)


# Output-tuple indices (13 real arrays, then 6 complex arrays).
(PSI_R, DELTA_R, DOP_R, RS, RP, R_AVG,
 PSI_T, DELTA_T, DOP_T, TS, TP, T_AVG,
 CONS, RS_C, RP_C, TS_C, TP_C, CROSS_R, CROSS_T) = range(19)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_n_stack_cache(n_layers_list, num_wavs):
    """Pack a wavelength-independent index profile into the engine's flat
    (num_wavs * n_layers * 2,) real/imag-interleaved layout."""
    nc = np.asarray(n_layers_list, dtype=complex)          # (n_layers,)
    n_layers = nc.size
    nc = np.tile(nc, (num_wavs, 1))                        # (num_wavs, n_layers)
    flat = np.empty(num_wavs * n_layers * 2, dtype=np.float64)
    flat[0::2] = np.ascontiguousarray(nc.real).ravel()
    flat[1::2] = np.ascontiguousarray(nc.imag).ravel()
    return flat, n_layers


def run(n_list, thick, inc, wavls, sin_theta, mode,
        rough_types=None, rough_vals=None, debug=1):
    """Thin typed wrapper around the engine."""
    num_wavs = wavls.size
    cache, n_layers = make_n_stack_cache(n_list, num_wavs)
    if rough_types is None:
        rough_types = np.zeros(n_layers, dtype=np.int32)
    if rough_vals is None:
        rough_vals = np.zeros(n_layers, dtype=np.float64)
    return engine(
        np.ascontiguousarray(wavls, dtype=np.float64),
        np.ascontiguousarray(sin_theta, dtype=np.float64),
        int(n_layers),
        cache,
        np.ascontiguousarray(thick, dtype=np.float64),
        np.ascontiguousarray(inc, dtype=np.int32),
        np.ascontiguousarray(rough_types, dtype=np.int32),
        np.ascontiguousarray(rough_vals, dtype=np.float64),
        int(debug),
        int(mode),
    )


def ang_diff(a, b):
    """Wrap-safe |a - b| for angles in radians."""
    d = (a - b + np.pi) % (2.0 * np.pi) - np.pi
    return np.abs(d)


# ---------------------------------------------------------------------------
# Test 1: Mode B == Mode A on R / T / Psi  (and is distinct on DOP/Delta)
# ---------------------------------------------------------------------------
def test_modeB_matches_modeA_on_RTpsi():
    wavls = np.array([450.0, 550.0, 650.0])                 # nm
    sin_theta = np.sin(np.deg2rad([40.0, 55.0, 65.0]))

    # air | TiO2 | SiO2 | [thick lossless glass = incoherent] | MgF2 | air
    n_list = [1.0 + 0j, 2.30 + 0.01j, 1.46 + 0j, 1.52 + 0j, 1.38 + 0j, 1.0 + 0j]
    thick = np.array([0.0, 80.0, 120.0, 1.0e6, 100.0, 0.0])  # nm
    inc = np.array([0, 0, 0, 1, 0, 0], dtype=np.int32)       # one incoherent block

    a = run(n_list, thick, inc, wavls, sin_theta, CoherenceMode.FRONT_BLOCK)
    b = run(n_list, thick, inc, wavls, sin_theta, CoherenceMode.COHERENCY_MATRIX)

    # Diagonal (intensity) channels are identical code -> bitwise-tight match.
    for idx, name in [(RS, "Rs"), (RP, "Rp"), (TS, "Ts"), (TP, "Tp"),
                      (R_AVG, "R_avg"), (T_AVG, "T_avg"),
                      (PSI_R, "Psi_R"), (PSI_T, "Psi_T")]:
        np.testing.assert_allclose(
            a[idx], b[idx], rtol=0.0, atol=1e-12,
            err_msg=f"Mode A and B disagree on {name} (diagonal channel must match)",
        )

    # Distinctness: the off-diagonal (DOP_R) must actually differ, otherwise the
    # equality above would be trivially satisfied by Mode B doing nothing.
    max_dop_gap = np.max(np.abs(a[DOP_R] - b[DOP_R]))
    max_delta_gap = np.max(ang_diff(a[DELTA_R], b[DELTA_R]))
    assert max_dop_gap > 1e-6, (
        "Modes A and B are indistinguishable on DOP_R; the test stack is not "
        "exercising the coherency channel (need >=2 coherent blocks with a "
        "surviving incoherent echo)."
    )
    print(f"[test 1] R/T/Psi identical; A-vs-B max |dDOP_R|={max_dop_gap:.3e}, "
          f"max |dDelta_R|={max_delta_gap:.3e} rad (expected nonzero).")


# ---------------------------------------------------------------------------
# Test 2: single coherent block -> A == B == C on Delta/DOP (and DOP_R == 1)
# ---------------------------------------------------------------------------
def test_single_block_all_modes_agree():
    wavls = np.array([500.0, 600.0])                        # nm
    sin_theta = np.sin(np.deg2rad([45.0, 60.0]))

    # air | TiO2 | SiO2 | TiO2 | glass  -- all lossless, NO incoherent layers
    n_list = [1.0 + 0j, 2.30 + 0j, 1.46 + 0j, 2.30 + 0j, 1.52 + 0j]
    thick = np.array([0.0, 70.0, 110.0, 70.0, 0.0])         # nm
    inc = np.zeros(len(n_list), dtype=np.int32)             # single coherent block

    outs = {m: run(n_list, thick, inc, wavls, sin_theta, m) for m in CoherenceMode}
    A = outs[CoherenceMode.FRONT_BLOCK]
    B = outs[CoherenceMode.COHERENCY_MATRIX]
    C = outs[CoherenceMode.FULLY_COHERENT]

    # All three modes must coincide for a single coherent block.
    for idx, name in [(DELTA_R, "Delta_R"), (DELTA_T, "Delta_T"),
                      (DOP_R, "DOP_R"), (DOP_T, "DOP_T"),
                      (PSI_R, "Psi_R"), (PSI_T, "Psi_T"),
                      (RS, "Rs"), (RP, "Rp"), (TS, "Ts"), (TP, "Tp")]:
        if idx in (DELTA_R, DELTA_T):
            assert np.max(ang_diff(A[idx], B[idx])) < 1e-10, f"A!=B on {name}"
            assert np.max(ang_diff(A[idx], C[idx])) < 1e-10, f"A!=C on {name}"
        else:
            np.testing.assert_allclose(A[idx], B[idx], rtol=0, atol=1e-10,
                                       err_msg=f"A!=B on {name}")
            np.testing.assert_allclose(A[idx], C[idx], rtol=0, atol=1e-10,
                                       err_msg=f"A!=C on {name}")

    # A single coherent block is fully polarized in reflection.
    np.testing.assert_allclose(A[DOP_R], 1.0, rtol=0, atol=1e-9,
                               err_msg="DOP_R != 1 for a single coherent block")

    # Lossless stack -> energy is conserved (R + T = 1 per polarization).
    assert np.max(A[CONS]) < 1e-6, "energy not conserved for lossless stack"

    # Sanity on the appended complex outputs: cross_R must equal rp*conj(rs)
    # here (rank-1 coherency for a single block), confirming the derivations.
    recon = A[RP_C] * np.conjugate(A[RS_C])
    np.testing.assert_allclose(A[CROSS_R], recon, rtol=0, atol=1e-10,
                               err_msg="cross_R != rp*conj(rs) for single block")
    print("[test 2] A == B == C on Delta/DOP/Psi/R/T; DOP_R == 1; "
          "cross_R factorizes as rp*conj(rs).")


if __name__ == "__main__":
    test_modeB_matches_modeA_on_RTpsi()
    test_single_block_all_modes_agree()
    print("\nAll coherence-mode validation tests passed.")
