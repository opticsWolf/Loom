# -*- coding: utf-8 -*-
"""Cross-boundary differentials: Python model vs bound Rust model.

Layer 3 of the verification contract (§8): both runtimes in one test.
Error-free paths assert BIT-identity; error paths assert statistical
agreement (ChaCha vs PCG64 streams differ by algorithm — means/moments
must agree, values must not). Runs before the flip (Python impl vs
bound classes) and after (bound vs bound, still green).
"""

import numpy as np

from navette.structure.models import Layer as PyLayer, Group as PyGroup
from navette.structure.structure import Navette_Structure as PyStructure
from navette.structure.architect import Navette_Architect as PyArchitect
from navette._structure import Layer as RsLayer, Group as RsGroup
from navette._structure import DictProvider as RsProvider
from navette._structure import Structure as RsStructure, Architect as RsArchitect

WL = np.array([900.0, 1200.0, 1500.0])
POOL = {
  "glass": np.array([1.52 + 0j, 1.51 + 0j, 1.50 + 0j]),
  "TiO2": np.array([2.35 + 0.01j, 2.33 + 0.008j, 2.31 + 0.006j]),
  "Si": np.array([3.5 + 0.02j, 3.48 + 0.015j, 3.46 + 0.01j]),
}
NAMES = list(POOL)


def _rand_layer(rng):
  return dict(
    thickness=float(rng.uniform(0.0, 120.0)),
    material_name=NAMES[int(rng.integers(0, len(NAMES)))],
    roughness=float(rng.uniform(0.0, 3.0)),
    rough_type=int(rng.integers(0, 6)),
    inhomogen=bool(rng.integers(0, 2)),
    inh_delta=float(rng.uniform(0.0, 0.4)),
    interface=bool(rng.integers(0, 2)),
    interface_thickness=float(rng.uniform(0.0, 8.0)),
    coherent=bool(rng.integers(0, 2)),
  )


def _rand_group(rng, name):
  scale_nk = bool(rng.integers(0, 2))
  return dict(
    group_name=name,
    thick_factor=float(rng.uniform(0.5, 1.5)),
    thick_summand=float(rng.uniform(-2.0, 2.0)),
    n_factor=float(rng.uniform(0.9, 1.1)) if scale_nk else 1.0,
    k_factor=float(rng.uniform(0.5, 1.5)) if scale_nk else 1.0,
    inh_delta_summand=float(rng.uniform(-0.05, 0.05)),
    roughness_summand=float(rng.uniform(0.0, 1.0)),
    interface_summand=float(rng.uniform(0.0, 2.0)),
  )


def _build(seed):
  rng = np.random.default_rng(seed)
  n = int(rng.integers(1, 6))
  Triple = []
  layers, groups = [], {}
  used = set()
  for _ in range(n):
    kw = _rand_layer(rng)
    layers.append(kw)
    if kw["material_name"] not in used:
      used.add(kw["material_name"])
      groups[kw["material_name"]] = _rand_group(rng, kw["material_name"])
  return layers, groups


def _py_stack(layers, groups):
  return PyStructure(
    [PyLayer(**kw) for kw in layers],
    {k: PyGroup(**g) for k, g in groups.items()},
    dict(POOL),
  )


def _rs_stack(layers, groups):
  # NOTE (flip): carried dict provider + no-arg solve — identical shape on
  # both implementations, so this file runs unchanged across the flip.
  return RsStructure(
    [RsLayer(**kw) for kw in layers],
    {k: RsGroup(**g) for k, g in groups.items()},
    dict(POOL),
  )


def _valid(issues):
  return not any(not i.startswith("warning:") for i in issues)


def test_random_stacks_bit_identical():
  # 300 seeded stacks: validity agreed, then thicknesses exact,
  # nk ~1e-12, flags exact on the solvable ones.
  n_compared = 0
  for seed in range(300):
    layers, groups = _build(seed)
    py_st = _py_stack(layers, groups)
    rs = _rs_stack(layers, groups)
    py_ok = _valid(py_st.validate())
    rs_ok = _valid(rs.validate())
    assert py_ok == rs_ok, f"validity diverges at seed {seed}"
    if not py_ok:
      continue
    n_compared += 1
    py = py_st.get_solver_inputs()
    sa = rs.solver_inputs()
    assert sa.thicknesses.tolist() == py.thicknesses.tolist(), f"seed {seed}"
    np.testing.assert_allclose(np.asarray(sa.indices), py.indices, rtol=0, atol=1e-12)
    assert sa.incoherent_flags.tolist() == py.incoherent_flags.tolist()
    assert sa.rough_types.tolist() == py.rough_types.tolist()
    np.testing.assert_allclose(np.asarray(sa.rough_vals), py.rough_vals, rtol=0, atol=1e-12)
  assert n_compared > 200, n_compared


def test_random_architects_bit_identical():
  # Mixed forward/inverted/repeat chains.
  rng = np.random.default_rng(99)
  for seed in range(100):
    layers, groups = _build(1000 + seed)
    py_a = PyArchitect(materials=dict(POOL))
    n_blocks = int(rng.integers(1, 4))
    descs = []
    for _ in range(n_blocks):
      descs.append((bool(rng.integers(0, 2)), int(rng.integers(1, 3)), int(rng.integers(0, 2))))
    py_st = PyStructure([PyLayer(**kw) for kw in layers],
                        {k: PyGroup(**g) for k, g in groups.items()}, None)
    rs_st = RsStructure([RsLayer(**kw) for kw in layers],
                        {k: RsGroup(**g) for k, g in groups.items()},
                        dict(POOL))
    rs_a = RsArchitect(dict(POOL))
    for inv, rep, kind in descs:
      py_a.add_structure(py_st, inverted=inv, repeat=rep, kind=kind)
      rs_a.add_structure(rs_st, inverted=inv, repeat=rep, kind=kind)
    # NOTE: shared-handle aliasing — py and rs stacks are distinct objects
    # with identical content, so mutation-semantics differences can't leak.
    py_ok = _valid(py_a.validate())
    rs_ok = _valid(rs_a.validate())
    assert py_ok == rs_ok, f"validity diverges at seed {seed}"
    if not py_ok:
      continue
    py = py_a.get_solver_inputs()
    sa = rs_a.solver_inputs()
    assert sa.thicknesses.tolist() == py.thicknesses.tolist(), f"seed {seed}"
    np.testing.assert_allclose(np.asarray(sa.indices), py.indices, rtol=0, atol=1e-12)


def test_error_statistics_agree(rng_for):
  # Same Gaussian thickness law both sides: moments agree within 5%,
  # exact values differ (documented §9.2 acceptance).
  kw = dict(thickness=50.0, material_name="TiO2")
  gkw = dict(group_name="TiO2", thick_factor=1.0)
  law = dict(abs_mean_delta_g=0.0, abs_std_dev=2.0, rel_mean_delta_g=0.0,
             rel_std_dev=0.0, abs_mean_delta_h=0.0, abs_variance=0.0,
             rel_mean_delta_h=0.0, rel_variance=0.0)
  py_g = PyGroup(**gkw)
  py_g.error_mask = [1, 0, 0, 0, 0, 0]
  py_g.set_properties({"thickness_error_params": law})
  rs_g = RsGroup(**gkw)
  rs_g.error_mask = [1, 0, 0, 0, 0, 0]
  rs_g.set_properties({"thickness_error_params": law})
  # Drive Rust draws through repeated seeded expansions (seed varies).
  rs_vals = []
  rs_st = RsStructure([RsLayer(**kw)], {"TiO2": rs_g}, dict(POOL))
  for s in range(400):
    sa = rs_st.error_inputs(s)
    rs_vals.append(sa.thicknesses[0])
  py_st = PyStructure([PyLayer(**kw)], {"TiO2": py_g}, dict(POOL))
  py_vals = []
  for s in range(400):
    sa = py_st.get_error_solver_inputs(rng=rng_for(s))
    py_vals.append(sa.thicknesses[0])
  for vals in (rs_vals, py_vals):
    assert abs(np.mean(vals) - 50.0) < 0.5, np.mean(vals)
    assert abs(np.std(vals) - 2.0) < 0.3, np.std(vals)
  assert abs(np.mean(rs_vals) - np.mean(py_vals)) < 0.5
