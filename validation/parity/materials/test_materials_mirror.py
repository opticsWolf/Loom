#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mirror of the Rust materials tests through the installed extension.

Covers ``crates/navette-materials/tests/parity.rs`` (22 golden tests —
same ``.npy`` references, same rtol/atol) and ``table.rs`` (Konstant flat,
Table nodes/clamp), calling ``navette._materials`` with the identical
arguments as the Rust tests.

Bonus beyond Rust: every ``MaterialSpec`` model is evaluated through the
``navette.materials.evaluate`` dispatcher and asserted equal to the direct
native call — pinning the wrapper's parameter mapping (the Rust tests only
cover the kernels).

Run explicitly:  python validation/parity/materials/test_materials_mirror.py
"""

import os
import sys

import numpy as np

from navette import _materials as N
from navette.materials import MaterialSpec, evaluate

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.normpath(os.path.join(HERE, "goldens"))

FAILURES = []


def check(name, ok, detail=""):
    print(f"  {name}: {'OK' if ok else 'FAIL'} {detail}")
    if not ok:
        FAILURES.append(name)


def load(case, part):
    return np.load(os.path.join(GOLDEN, f"{case}__{part}.npy"))


def check_close(name, got, case, rtol, atol):
    got = np.asarray(got)
    re = load(case, "re")
    im = load(case, "im")
    ok_len = got.shape == re.shape == im.shape
    dr = np.abs(got.real - re)
    di = np.abs(got.imag - im)
    tr = atol + rtol * np.abs(re)
    ti = atol + rtol * np.abs(im)
    ok = ok_len and bool(np.all(dr <= tr) and np.all(di <= ti))
    worst = float(max(np.max(dr / tr), np.max(di / ti))) if ok_len else float("nan")
    check(name, ok, f"worst tol-ratio {worst:.2f}")


def test_parity_goldens():
    print("--- tests/parity.rs goldens (22) ---")
    wl = load("cauchy_basic", "wl")
    check_close("cauchy_basic", N.cauchy_nk(wl, 1.5, 0.01, 0.0001), "cauchy_basic", 1e-10, 1e-12)
    wl = load("cauchy_single", "wl")
    check_close("cauchy_single", N.cauchy_nk(wl, 1.5, 0.01, 0.0001), "cauchy_single", 1e-10, 1e-12)
    wl = load("cauchy_urbach", "wl")
    check_close("cauchy_urbach", N.cauchy_urbach_nk(wl, 2.5, 0.02, 0.0005, 1e4, 0.05, 400.0),
                "cauchy_urbach", 1e-9, 1e-14)
    wl = load("sellmeier_bk7", "wl")
    check_close("sellmeier_bk7",
                N.sellmeier_nk(wl, 1.03961212, 0.00600069867, 0.231792344, 0.0200179144,
                               1.01046945, 103.560653), "sellmeier_bk7", 1e-10, 1e-12)
    wl = load("sellmeier_2term", "wl")
    check_close("sellmeier_2term", N.sellmeier_nk(wl, 1.0, 0.01, 0.3, 0.05, 0.0, 0.0),
                "sellmeier_2term", 1e-10, 1e-12)
    wl = load("sellmeier_urbach", "wl")
    check_close("sellmeier_urbach",
                N.sellmeier_urbach_nk(wl, 1.4313, 0.01, 0.65, 0.025, 0.0, 0.0, 1e5, 0.06, 380.0),
                "sellmeier_urbach", 1e-9, 1e-14)
    wl = load("lorentz_2osc", "wl")
    check_close("lorentz_2osc", N.lorentz_nk(wl, np.array([[3.0, 0.2, 0.5], [4.5, 0.1, 0.7]]), 1.0),
                "lorentz_2osc", 1e-10, 1e-12)
    wl = load("drude_basic", "wl")
    check_close("drude_basic", N.drude_nk(wl, 2.5, 0.3, 3.5), "drude_basic", 1e-10, 1e-12)
    wl = load("drude_lorentz", "wl")
    check_close("drude_lorentz",
                N.drude_lorentz_nk(wl, 9.0, 0.05, 1.0, np.array([[2.0, 0.5, 1.0], [3.5, 0.8, 0.4]])),
                "drude_lorentz", 1e-10, 1e-12)
    wl = load("fb_single", "wl")
    check_close("fb_single", N.fb_interband_nk(wl, 1.5, np.array([[3.0, 0.1, 6.0, 12.0]])),
                "fb_single", 1e-10, 1e-12)
    wl = load("fb_multi", "wl")
    check_close("fb_multi",
                N.fb_interband_nk(wl, 1.2, np.array([[3.0, 0.1, 6.0, 12.0], [4.5, 0.05, 9.0, 22.0]])),
                "fb_multi", 1e-10, 1e-12)
    wl = load("fb_edge", "wl")
    check_close("fb_edge", N.fb_interband_nk(wl, 1.0, np.array([[2.0, 0.1, 10.0, 20.0]])),
                "fb_edge", 1e-10, 1e-12)
    wl = load("fb_metal", "wl")
    check_close("fb_metal",
                N.fb_metal_nk(wl, 1.0, np.array([5.0, 0.5, 0.3]), np.array([[2.0, 0.2, 4.0, 5.0]])),
                "fb_metal", 1e-10, 1e-12)
    wl = load("tauc_single", "wl")
    check_close("tauc_single", N.tauc_lorentz_nk(wl, 1.2, np.array([[100.0, 4.0, 2.0]]), 1.0),
                "tauc_single", 1e-9, 1e-11)
    wl = load("tauc_multi", "wl")
    check_close("tauc_multi",
                N.tauc_lorentz_nk(wl, 1.2, np.array([[100.0, 4.0, 2.0], [50.0, 6.5, 1.5]]), 1.5),
                "tauc_multi", 1e-9, 1e-11)
    wl = load("ubf_single", "wl")
    check_close("ubf_single", N.ubf_nk(wl, np.array([[1.5, 3.0, 1.0 / 0.2, 10.0, 1.0, 2.0]]), 1.0),
                "ubf_single", 1e-9, 1e-11)
    wl = load("ubf_multi", "wl")
    check_close("ubf_multi",
                N.ubf_nk(wl, np.array([[1.5, 3.0, 1.0 / 0.2, 10.0, 1.0, 2.0],
                                       [2.2, 4.5, 1.0 / 0.15, 6.0, 1.5, 0.5]]), 1.0),
                "ubf_multi", 1e-9, 1e-11)
    wl = load("cody_single", "wl")
    check_close("cody_single",
                N.cody_lorentz_nk(wl, 1.64, 1.80, 0.15, np.array([[3.40, 60.0, 2.4, 1.0]]), 1.0),
                "cody_single", 1e-9, 1e-11)
    wl = load("cody_multi", "wl")
    check_close("cody_multi",
                N.cody_lorentz_nk(wl, 1.64, 1.80, 0.15,
                                  np.array([[3.40, 60.0, 2.4, 1.0], [4.70, 40.0, 1.8, 0.5]]), 1.0),
                "cody_multi", 1e-9, 1e-11)
    # EMA references store epsilon, not nk — the native ema_* return eps.
    t = np.arange(120) / 119.0
    n_i = (2.0 + 0.4 * t) + 1j * (0.05 + 0.15 * t)
    n_h = (1.4 + 0.1 * t) + 1j * (0.0 + 0.02 * t)
    check_close("ema_looyenga", N.ema_looyenga(n_i, n_h, 0.3), "ema_looyenga", 1e-9, 1e-12)
    check_close("ema_maxwell_garnett", N.ema_maxwell_garnett(n_i, n_h, 0.3),
                "ema_maxwell_garnett", 1e-10, 1e-12)
    check_close("ema_bruggeman", N.ema_bruggeman(n_i, n_h, 0.3, 100, 1e-9),
                "ema_bruggeman", 1e-6, 1e-9)


def test_table_unit():
    print("--- table.rs unit tests ---")
    nk = np.asarray(N.konstant_nk(np.array([400., 500., 600.]), 1.5, 0.01))
    check("konstant_is_flat", bool(np.all(nk == complex(1.5, 0.01))), f"{nk}")
    nk = np.asarray(N.table_nk(np.array([400., 450., 600., 700.]),
                               np.array([400., 500., 600.]), np.array([1., 2., 3.]),
                               None, 1.0, 1.0))
    check("table_nodes_clamp",
          abs(nk[0].real - 1.0) < 1e-12 and abs(nk[1].real - 1.5) < 1e-12
          and abs(nk[2].real - 3.0) < 1e-12 and abs(nk[3].real - 3.0) < 1e-12
          and bool(np.all(nk.imag == 0.0)), f"{nk.real}")


def test_evaluate_dispatch():
    print("--- evaluate() dispatcher vs direct native ---")
    wl = np.linspace(400.0, 800.0, 9)
    cases = [
        ("Konstant", dict(n=1.5, k=0.01), lambda: N.konstant_nk(wl, 1.5, 0.01)),
        ("Table", dict(n_data=([400., 600., 800.], [1.0, 2.0, 1.5]),
                       k_data=([400., 600., 800.], [0.0, 0.1, 0.0])),
         lambda: N.table_nk(wl, np.array([400., 600., 800.]), np.array([1., 2., 1.5]),
                            np.array([0.0, 0.1, 0.0]), 1.0, 1.0)),
        ("Cauchy", dict(A=1.5, B=0.01, C=0.0001), lambda: N.cauchy_nk(wl, 1.5, 0.01, 0.0001)),
        ("CauchyUrbach", dict(A=2.5, B=0.02, C=0.0005, alpha0=1e4, Eu=0.05, lambda_g=400.0),
         lambda: N.cauchy_urbach_nk(wl, 2.5, 0.02, 0.0005, 1e4, 0.05, 400.0)),
        ("Sellmeier", dict(B1=1.03961212, C1=0.00600069867, B2=0.231792344,
                            C2=0.0200179144, B3=1.01046945, C3=103.560653),
         lambda: N.sellmeier_nk(wl, 1.03961212, 0.00600069867, 0.231792344, 0.0200179144,
                                1.01046945, 103.560653)),
        ("SellmeierUrbach", dict(B1=1.4313, C1=0.01, B2=0.65, C2=0.025, B3=0.0, C3=0.0,
                                 alpha0=1e5, Eu=0.06, lambda_g=380.0),
         lambda: N.sellmeier_urbach_nk(wl, 1.4313, 0.01, 0.65, 0.025, 0.0, 0.0, 1e5, 0.06, 380.0)),
        ("Lorentz", dict(osc=[[3.0, 0.2, 0.5], [4.5, 0.1, 0.7]], epsilon_inf=1.0),
         lambda: N.lorentz_nk(wl, np.array([[3.0, 0.2, 0.5], [4.5, 0.1, 0.7]]), 1.0)),
        ("Drude", dict(omega_p=2.5, gamma=0.3, epsilon_inf=3.5),
         lambda: N.drude_nk(wl, 2.5, 0.3, 3.5)),
        ("DrudeLorentz", dict(omega_p=9.0, gamma=0.05, epsilon_inf=1.0,
                              osc=[[2.0, 0.5, 1.0], [3.5, 0.8, 0.4]]),
         lambda: N.drude_lorentz_nk(wl, 9.0, 0.05, 1.0, np.array([[2.0, 0.5, 1.0], [3.5, 0.8, 0.4]]))),
        ("CodyLorentz", dict(Eg=1.64, Et=1.80, Eu=0.15, osc=[[3.40, 60.0, 2.4, 1.0]],
                             epsilon_inf=1.0),
         lambda: N.cody_lorentz_nk(wl, 1.64, 1.80, 0.15, np.array([[3.40, 60.0, 2.4, 1.0]]), 1.0)),
        ("ForouhiBloomerSingle", dict(n_inf=1.5, ib=[[3.0, 0.1, 6.0, 12.0]]),
         lambda: N.fb_interband_nk(wl, 1.5, np.array([[3.0, 0.1, 6.0, 12.0]]))),
        ("ForouhiBloomerMulti", dict(n_inf=1.2, ib=[[3.0, 0.1, 6.0, 12.0], [4.5, 0.05, 9.0, 22.0]]),
         lambda: N.fb_interband_nk(wl, 1.2, np.array([[3.0, 0.1, 6.0, 12.0], [4.5, 0.05, 9.0, 22.0]]))),
        ("ForouhiBloomerMetal", dict(n_inf=1.0, ib=[[2.0, 0.2, 4.0, 5.0]],
                                     A_fe=5.0, B_fe=0.5, C_fe=0.3),
         lambda: N.fb_metal_nk(wl, 1.0, np.array([5.0, 0.5, 0.3]), np.array([[2.0, 0.2, 4.0, 5.0]]))),
        ("TaucLorentz", dict(Eg=1.2, osc=[[100.0, 4.0, 2.0]], epsilon_inf=1.0),
         lambda: N.tauc_lorentz_nk(wl, 1.2, np.array([[100.0, 4.0, 2.0]]), 1.0)),
        ("UBF", dict(osc=[dict(Eg=1.5, Ec=3.0, Eu=0.2, A=10.0, Gamma=1.0, gamma=2.0)],
                     epsilon_inf=1.0),
         lambda: N.ubf_nk(wl, np.array([[1.5, 3.0, 1.0 / 0.2, 10.0, 1.0, 2.0]]), 1.0)),
        ("Bruggeman", dict(host=dict(model="Konstant", params=dict(n=1.4)),
                            inclusion=dict(model="Konstant", params=dict(n=2.0, k=0.1)),
                            fraction=0.3),
         lambda: N.eps_to_nk(N.ema_bruggeman(evaluate(
             MaterialSpec("Konstant", dict(n=2.0, k=0.1)), wl),
             evaluate(MaterialSpec("Konstant", dict(n=1.4)), wl), 0.3))),
        ("Roughness", dict(bottom=dict(model="Konstant", params=dict(n=1.5)),
                            top=dict(model="Konstant", params=dict(n=1.0))),
         lambda: N.eps_to_nk(N.ema_roughness(
             evaluate(MaterialSpec("Konstant", dict(n=1.5)), wl),
             evaluate(MaterialSpec("Konstant", dict(n=1.0)), wl)))),
    ]
    for model, params, direct in cases:
        got = np.asarray(evaluate(MaterialSpec(model, params), wl))
        want = np.asarray(direct())
        ok = got.shape == want.shape and bool(np.allclose(got, want, rtol=0, atol=0))
        worst = float(np.max(np.abs(got - want))) if got.shape == want.shape else float("nan")
        check(f"dispatch_{model}", ok, f"max|d|={worst:.2e}")
    # error paths
    for name, thunk in [
        ("unknown_model", lambda: evaluate(MaterialSpec("Nope", {}), wl)),
        ("missing_param", lambda: evaluate(MaterialSpec("Cauchy", dict(A=1.5)), wl)),
        ("empty_wl", lambda: evaluate(MaterialSpec("Konstant", dict(n=1.5)), np.array([]))),
    ]:
        try:
            thunk()
            check(name, False)
        except (ValueError, Exception):
            check(name, True)


if __name__ == "__main__":
    test_parity_goldens()
    test_table_unit()
    test_evaluate_dispatch()
    print("ALL OK" if not FAILURES else f"MISMATCH {FAILURES}")
    sys.exit(1 if FAILURES else 0)
