"""
Validation tests for the unified request-driven engine `core_engine` (func_6).

  Test 1 — A single-pol request (Request.PHI_RS) returns only the s-pol key and
           never runs the p branch. We can't observe branch execution directly,
           so we assert it behaviorally: the returned dict contains exactly
           {"phi_rs"}; the s-pol result is bit-identical whether or not the p
           request is also present (s is independent of p); and it matches the
           first-block amplitude phase that func_4 reports (np.angle of rs_c).

  Test 2 — Request.ELLIPSOMETRY through func_6 reproduces the legacy func_4
           tuple (Psi/Delta/DOP, reflection and transmission) bit-for-bit, in
           both Mode A and Mode B, since both consume the same core.

  Test 3 — func_6 photometry matches func_5 (front_block), full and single-pol.

Run: pytest -q test_func6.py   or   python test_func6.py
Requires the compiled extension and request_flags.py on the path.
"""

import numpy as np

try:
    from smatrix import (
        core_engine,
        core_engine_rigorous_ellipsometry as engine4,
        core_engine_photometry_only as engine5,
    )
except ImportError:  # pragma: no cover
    from loom_matrix import (
        core_engine,
        core_engine_rigorous_ellipsometry as engine4,
        core_engine_photometry_only as engine5,
    )

from request_flags import Request, CoherenceMode

# func_4 tuple indices
(PSI_R, DELTA_R, DOP_R, RS, RP, R_AVG,
 PSI_T, DELTA_T, DOP_T, TS, TP, T_AVG,
 CONS, RS_C, RP_C, TS_C, TP_C, CROSS_R, CROSS_T) = range(19)


# --- shared stack fixtures ---------------------------------------------------
def make_n_stack_cache(n_layers_list, num_wavs):
    nc = np.asarray(n_layers_list, dtype=complex)
    n_layers = nc.size
    nc = np.tile(nc, (num_wavs, 1))
    flat = np.empty(num_wavs * n_layers * 2, dtype=np.float64)
    flat[0::2] = np.ascontiguousarray(nc.real).ravel()
    flat[1::2] = np.ascontiguousarray(nc.imag).ravel()
    return flat, n_layers


# multi-block stack with one incoherent layer (exercises Mode B)
N_LIST = [1.0 + 0j, 2.30 + 0.01j, 1.46 + 0j, 1.52 + 0j, 1.38 + 0j, 1.0 + 0j]
THICK = np.array([0.0, 80.0, 120.0, 1.0e6, 100.0, 0.0])
INC = np.array([0, 0, 0, 1, 0, 0], dtype=np.int32)
WAVLS = np.array([450.0, 550.0, 650.0])
SIN_THETA = np.sin(np.deg2rad([40.0, 55.0, 65.0]))


def args6(requested, mode):
    cache, n_layers = make_n_stack_cache(N_LIST, WAVLS.size)
    rt = np.zeros(n_layers, np.int32)
    rv = np.zeros(n_layers, np.float64)
    return core_engine(
        np.ascontiguousarray(WAVLS), np.ascontiguousarray(SIN_THETA),
        int(n_layers), cache, np.ascontiguousarray(THICK),
        INC, rt, rv, int(mode), int(requested),
    )


def args4(mode):
    cache, n_layers = make_n_stack_cache(N_LIST, WAVLS.size)
    rt = np.zeros(n_layers, np.int32)
    rv = np.zeros(n_layers, np.float64)
    return engine4(
        np.ascontiguousarray(WAVLS), np.ascontiguousarray(SIN_THETA),
        int(n_layers), cache, np.ascontiguousarray(THICK),
        INC, rt, rv, 0, int(mode),
    )


def args5(calc_s, calc_p):
    cache, n_layers = make_n_stack_cache(N_LIST, WAVLS.size)
    rt = np.zeros(n_layers, np.int32)
    rv = np.zeros(n_layers, np.float64)
    return engine5(
        np.ascontiguousarray(WAVLS), np.ascontiguousarray(SIN_THETA),
        int(n_layers), cache, np.ascontiguousarray(THICK),
        INC, rt, rv, int(calc_s), int(calc_p),
    )


def ang_diff(a, b):
    return np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)


# --- Test 1: single-pol isolation -------------------------------------------
def test_phi_rs_single_pol_isolation():
    out = args6(Request.PHI_RS, CoherenceMode.COHERENCY_MATRIX)

    # Only the s-pol phase comes back; no p-derived keys exist.
    assert set(out.keys()) == {"phi_rs"}, f"unexpected keys: {set(out.keys())}"

    # s-pol result is unchanged by also requesting the p phase (s ⟂ p).
    out2 = args6(Request.PHI_RS | Request.PHI_RP, CoherenceMode.COHERENCY_MATRIX)
    np.testing.assert_array_equal(out["phi_rs"], out2["phi_rs"])
    assert "phi_rp" in out2  # the p branch only runs when actually asked

    # Matches the first-block amplitude phase func_4 reports.
    f4 = args4(CoherenceMode.COHERENCY_MATRIX)
    np.testing.assert_allclose(out["phi_rs"], np.angle(f4[RS_C]), rtol=0, atol=1e-9)
    print("[test 1] PHI_RS request: dict == {'phi_rs'}, s independent of p, "
          "matches func_4 rs_c phase.")


# --- Test 2: ellipsometry parity func_6 vs func_4 ---------------------------
def test_func6_ellipsometry_matches_func4():
    req = (Request.PSI_R | Request.DELTA_R | Request.DOP_R
           | Request.PSI_T | Request.DELTA_T | Request.DOP_T)
    for mode in (CoherenceMode.FRONT_BLOCK, CoherenceMode.COHERENCY_MATRIX):
        out = args6(req, mode)
        f4 = args4(mode)
        np.testing.assert_allclose(out["Psi_R"], f4[PSI_R], rtol=0, atol=1e-12)
        np.testing.assert_allclose(out["DOP_R"], f4[DOP_R], rtol=0, atol=1e-12)
        np.testing.assert_allclose(out["Psi_T"], f4[PSI_T], rtol=0, atol=1e-12)
        np.testing.assert_allclose(out["DOP_T"], f4[DOP_T], rtol=0, atol=1e-12)
        assert np.max(ang_diff(out["Delta_R"], f4[DELTA_R])) < 1e-12
        assert np.max(ang_diff(out["Delta_T"], f4[DELTA_T])) < 1e-12
    print("[test 2] func_6 ELLIPSOMETRY == func_4 tuple in Modes A and B.")


# --- Test 3: photometry parity func_6 vs func_5 -----------------------------
def test_func6_photometry_matches_func5():
    # func_5 is front_block; compare against func_6 Mode A.
    rs5, rp5, ts5, tp5 = args5(1, 1)
    out = args6(Request.RS | Request.RP | Request.TS | Request.TP,
                CoherenceMode.FRONT_BLOCK)
    np.testing.assert_allclose(out["Rs"], rs5, rtol=0, atol=1e-12)
    np.testing.assert_allclose(out["Rp"], rp5, rtol=0, atol=1e-12)
    np.testing.assert_allclose(out["Ts"], ts5, rtol=0, atol=1e-12)
    np.testing.assert_allclose(out["Tp"], tp5, rtol=0, atol=1e-12)

    # Single-pol: Request.RS returns only Rs and matches the dual-pol Rs.
    out_s = args6(Request.RS, CoherenceMode.FRONT_BLOCK)
    assert set(out_s.keys()) == {"Rs"}
    np.testing.assert_array_equal(out_s["Rs"], out["Rs"])
    print("[test 3] func_6 photometry == func_5; single-pol Rs isolated.")


if __name__ == "__main__":
    test_phi_rs_single_pol_isolation()
    test_func6_ellipsometry_matches_func4()
    test_func6_photometry_matches_func5()
    print("\nAll func_6 validation tests passed.")
