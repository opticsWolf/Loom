"""Native program loading twins the Python path, bitwise."""
import json
import numpy as np

import yaml

from navette.config.program import load_program
from navette._structure import load_program as native_load_program

EXAMPLE = "src/navette/config/example_program.yaml"
WL = np.array([500.0, 600.0, 700.0])


def _doc():
    with open(EXAMPLE, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_native_matches_python_solver_inputs():
    text = json.dumps(_doc())
    py = load_program(EXAMPLE, WL)
    nat = native_load_program(text, WL)
    assert nat["name"] == py.name
    assert set(nat["structures"].keys()) == set(py.structures.keys())
    for label in py.structures:
        a = py.structures[label].get_solver_inputs()
        b = nat["structures"][label].solver_inputs()
        assert np.array_equal(np.asarray(a.thicknesses), np.asarray(b.thicknesses))
        assert np.array_equal(np.asarray(a.indices), np.asarray(b.indices))


def test_native_prefix_and_solve():
    from navette._structure import solve_arrays_fn
    from navette._smatrix import solver_rt_request
    text = json.dumps(_doc())
    nat = native_load_program(text, WL, prefix="p_")
    assert set(nat["structures"].keys()) == {"p_ar"}
    sa = nat["structures"]["p_ar"].solver_inputs()
    out, warns = solve_arrays_fn(sa, WL, np.array([0.0]), solver_rt_request("u"))
    assert set(out) == {"Rs", "Rp", "Ts", "Tp", "R_avg", "T_avg"}
    assert list(warns) == [] or all(isinstance(w, str) for w in warns)
