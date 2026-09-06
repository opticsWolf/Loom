# -*- coding: utf-8 -*-
"""Pytest collection rules for validation/.

Collected (pytest-style suites):
  smoke/                 - package import surface, no build quirks assumed
  goldens/spectralweave/ - pinned numeric regression tests

Ignored (standalone scripts with top-level benchmark/parity code that must
be run explicitly, not imported by pytest):
  parity/   - numba-vs-Rust parity scripts, e.g.
              python validation/parity/smatrix/test_w_function.py
  benches/  - timing scripts, e.g.
              python validation/benches/spectralweave/navette_spectral_bench.py --quick
"""

collect_ignore = ["parity", "benches"]

import pytest  # noqa: E402  (fixture support below)


@pytest.fixture
def rng_for():
  """Injectable flip-proof RNG factory (see `_rng_for` below)."""
  return _rng_for


def _rng_for(seed):
  """Flip-proof RNG handle: `Generator` pre-flip, raw seed post-flip.

  All error-path tests take their randomness through this helper so the
  twin files run byte-identical against the Python implementation (which
  consumes `Generator`s) and the bound classes (which consume seeds).
  Post-flip this returns the seed itself; until then, a `Generator`.
  """
  import numpy as np
  try:
    from navette.structure import Layer
    from navette._structure import Layer as RsLayer
    if Layer is RsLayer:
      return seed
  except ImportError:
    pass
  return np.random.default_rng(seed)
