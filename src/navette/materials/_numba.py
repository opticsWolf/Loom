# -*- coding: utf-8 -*-
"""
Optional numba import shim.

``numba`` is an optional accelerator (``pip install navette[numba]``).
When it is unavailable, :func:`njit` degrades to an identity decorator
and :func:`prange` falls back to :func:`range`, so all models run as
pure Python / numpy.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

try:
  from numba import njit as _njit
  from numba import prange as _prange

  njit = _njit
  prange = _prange
except ImportError:  # pragma: no cover - numba not installed
  F = TypeVar("F", bound=Callable[..., Any])

  def njit(*args: Any, **kwargs: Any) -> Any:
    """Identity fallback when numba is missing."""
    if len(args) == 1 and callable(args[0]) and not kwargs:
      return args[0]

    def wrap(fn: F) -> F:
      return fn

    return wrap

  prange = range  # type: ignore[assignment]
