# -*- coding: utf-8 -*-
"""
Loom: Weaving the mathematics of light in thin film systems
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""

import json
import time
import os
import numpy as np
import tempfile
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Iterator, Generator
from numpy.typing import NDArray

__all__ = ["CIEJSONReader"]

# --- 1. The Class Definition (Integrated for standalone execution) ---
class CIEJSONReader:
    """A high-performance reader for CIE JSON spectral data."""

    def read_file(self, file_path: str | Path) -> dict[str, Any]:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def list_quantities(self, json_content: dict[str, Any]) -> list[str]:
        data_section = json_content.get('data', {})
        return [
            k for k, v in data_section.items() 
            if isinstance(v, dict) and 'quantity' in v and k.lower() not in ['lambda', 'm']
        ]

    def get_spectrum(self, json_content: dict, quantity_key: str) -> tuple[NDArray, NDArray]:
        # simplified for benchmark speed - assumes clean data or handles basics
        data = json_content['data'][quantity_key]
        vals = np.array(data['values'], dtype=np.float64)
        
        # Reconstruction logic (simplified)
        lam = json_content['data'].get('lambda', {})
        if 'values' in lam:
            wl = np.array(lam['values'], dtype=np.float64)
        else:
            start, end, step = lam.get('wavelength_first', 380), lam.get('wavelength_last', 780), lam.get('wavelength_step', 5)
            count = int((end - start) / step) + 1
            wl = np.linspace(start, end, count)
        
        # Basic trimming
        min_len = min(len(wl), len(vals))
        return wl[:min_len], vals[:min_len]

    def get_metadata(self, json_content: dict) -> dict:
        def clean(n):
            if isinstance(n, dict): return {k: clean(v) for k, v in n.items() if k != 'values'}
            if isinstance(n, list) and len(n) > 20: return f"<List {len(n)}>"
            return n
        return clean(json_content)

    def batch_process(self, directory: Path) -> Iterator:
        for f in directory.rglob("*.json"):
            try:
                rel = str(f.relative_to(directory))
                c = self.read_file(f)
                yield rel, self.get_metadata(c), {q: self.get_spectrum(c, q) for q in self.list_quantities(c)}
            except Exception: continue