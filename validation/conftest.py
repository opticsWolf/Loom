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
