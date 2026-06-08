# -*- coding: utf-8 -*-
"""
Navette: Weaving the mathematics of light in thin film systems
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later

Enhanced CIE JSON reader with Pydantic v2 validation, parallel processing,
and robust error handling.
"""

import json
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Iterator, List, Dict, Tuple, Optional, Union
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, model_validator, ValidationError

__all__ = ["CIEJSONReader", "CIEJSON", "LambdaConfig", "SpectralData"]

# -----------------------------------------------------------------------------
# Pydantic Models
# -----------------------------------------------------------------------------

class LambdaConfig(BaseModel):
    """
    Wavelength definition: either explicit values or a range with step.
    """
    values: Optional[List[float]] = None
    wavelength_first: Optional[float] = None
    wavelength_last: Optional[float] = None
    wavelength_step: Optional[float] = None

    @model_validator(mode='after')
    def _check_values_or_range(self) -> 'LambdaConfig':
        """Ensure either `values` or a complete range is provided."""
        if self.values is not None:
            return self
        required = (self.wavelength_first, self.wavelength_last, self.wavelength_step)
        if all(v is not None for v in required):
            return self
        raise ValueError(
            'Either "values" or a complete wavelength range (wavelength_first, '
            'wavelength_last, wavelength_step) must be provided'
        )

    def to_wavelengths(self) -> NDArray[np.float64]:
        """Generate wavelength array as numpy float64."""
        if self.values is not None:
            return np.array(self.values, dtype=np.float64)
        start = self.wavelength_first
        end = self.wavelength_last
        step = self.wavelength_step
        count = int((end - start) / step) + 1
        return np.linspace(start, end, count, dtype=np.float64)


class SpectralData(BaseModel):
    """
    A single spectral quantity (e.g., x-bar, y-bar, z-bar).
    """
    quantity: str
    values: List[float]
    # allow extra fields like 'comment', 'unit' etc.
    model_config = {"extra": "allow"}

    def to_array(self) -> NDArray[np.float64]:
        """Return values as numpy array."""
        return np.array(self.values, dtype=np.float64)


class CIEJSON(BaseModel):
    """
    Root model for a CIE JSON file.
    """
    description: Optional[str] = None
    copyright: Optional[str] = None
    license: Optional[str] = None
    data: Dict[str, Any]

    model_config = {"extra": "allow"}

    @property
    def lambda_config(self) -> LambdaConfig:
        """Extract and validate the wavelength definition."""
        lam = self.data.get('lambda')
        if not lam:
            raise KeyError("Missing 'lambda' section in data")
        return LambdaConfig(**lam)

    @property
    def wavelengths(self) -> NDArray[np.float64]:
        """Get wavelengths as a numpy array."""
        return self.lambda_config.to_wavelengths()

    def list_quantities(self) -> List[str]:
        """
        Return a list of quantity keys (excludes 'lambda' and special keys like 'm').
        """
        excluded = {'lambda', 'm'}
        return [
            k for k, v in self.data.items()
            if isinstance(v, dict) and 'quantity' in v and k.lower() not in excluded
        ]

    def get_spectrum(self, quantity_key: str) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Return (wavelengths, values) for the given quantity key.
        If lengths differ, both are truncated to the shorter length.
        """
        quant = self.data.get(quantity_key)
        if not quant or 'values' not in quant:
            raise ValueError(f"Quantity '{quantity_key}' not found or has no 'values'")
        wl = self.wavelengths
        vals = np.array(quant['values'], dtype=np.float64)
        min_len = min(len(wl), len(vals))
        return wl[:min_len], vals[:min_len]

    def to_dict_clean(self) -> Dict[str, Any]:
        """
        Return a dictionary with large value arrays stripped (useful for metadata).
        """
        def clean(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: clean(v) for k, v in obj.items() if k != 'values'}
            if isinstance(obj, list) and len(obj) > 20:
                return f"<List {len(obj)}>"
            return obj
        return clean(self.model_dump())


# -----------------------------------------------------------------------------
# Reader Class
# -----------------------------------------------------------------------------

class CIEJSONReader:
    """
    High-performance reader for CIE JSON spectral data with Pydantic validation,
    parallel batch processing, and configurable error handling.
    """

    def __init__(self, strict: bool = True, logger: Optional[logging.Logger] = None):
        """
        Args:
            strict: If True, raise ValidationError on malformed files.
                    If False, attempt to parse with best effort (fallback to original logic).
            logger: Optional logger for warning/error messages.
        """
        self.strict = strict
        self.logger = logger or logging.getLogger(__name__)

    def read_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Read and validate a CIE JSON file. Returns the raw dictionary.
        Raises ValidationError if strict=True and the file does not conform.
        """
        path = Path(file_path)
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        if self.strict:
            # Validate using the model
            CIEJSON(**raw)
        return raw

    def read_model(self, file_path: Union[str, Path]) -> CIEJSON:
        """Read a CIE JSON file and return a validated CIEJSON object."""
        path = Path(file_path)
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        return CIEJSON(**raw)

    def list_quantities(self, json_content: Dict[str, Any]) -> List[str]:
        """Return quantity keys from a raw JSON dict."""
        try:
            model = CIEJSON(**json_content)
            return model.list_quantities()
        except ValidationError:
            if not self.strict:
                # Fallback to original heuristic
                data_section = json_content.get('data', {})
                return [
                    k for k, v in data_section.items()
                    if isinstance(v, dict) and 'quantity' in v and k.lower() not in ('lambda', 'm')
                ]
            raise

    def get_spectrum(
        self, json_content: Dict[str, Any], quantity_key: str
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Extract (wavelengths, values) for a given quantity.
        """
        try:
            model = CIEJSON(**json_content)
            return model.get_spectrum(quantity_key)
        except ValidationError:
            if self.strict:
                raise
            # Fallback: original simplified reconstruction
            data = json_content['data'][quantity_key]
            vals = np.array(data['values'], dtype=np.float64)
            lam = json_content['data'].get('lambda', {})
            if 'values' in lam:
                wl = np.array(lam['values'], dtype=np.float64)
            else:
                start = lam.get('wavelength_first', 380)
                end = lam.get('wavelength_last', 780)
                step = lam.get('wavelength_step', 5)
                count = int((end - start) / step) + 1
                wl = np.linspace(start, end, count, dtype=np.float64)
            min_len = min(len(wl), len(vals))
            return wl[:min_len], vals[:min_len]

    def get_metadata(self, json_content: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata with large arrays stripped."""
        try:
            model = CIEJSON(**json_content)
            return model.to_dict_clean()
        except ValidationError:
            if self.strict:
                raise
            # Fallback: recursive cleaning
            def clean(obj):
                if isinstance(obj, dict):
                    return {k: clean(v) for k, v in obj.items() if k != 'values'}
                if isinstance(obj, list) and len(obj) > 20:
                    return f"<List {len(obj)}>"
                return obj
            return clean(json_content)

    def batch_process(
        self,
        directory: Path,
        skip_on_error: bool = True,
        recursive: bool = True
    ) -> Iterator[Tuple[str, Dict[str, Any], Dict[str, Tuple[NDArray, NDArray]]]]:
        """
        Iterate over JSON files in `directory` (non‑parallel), yielding
        (relative_path, metadata, spectra_dict) for each successfully read file.

        Args:
            directory: Root directory to scan.
            skip_on_error: If True, skip files that cause exceptions; if False, re-raise.
            recursive: If True, scan subdirectories recursively.
        """
        pattern = "**/*.json" if recursive else "*.json"
        for file_path in directory.glob(pattern):
            try:
                rel = str(file_path.relative_to(directory))
                model = self.read_model(file_path)
                metadata = model.to_dict_clean()
                spectra = {q: model.get_spectrum(q) for q in model.list_quantities()}
                yield rel, metadata, spectra
            except Exception as e:
                if skip_on_error:
                    self.logger.warning(f"Skipping {file_path}: {e}")
                    continue
                raise

    def batch_process_parallel(
        self,
        directory: Path,
        max_workers: Optional[int] = None,
        skip_on_error: bool = True,
        recursive: bool = True
    ) -> Iterator[Tuple[str, Dict[str, Any], Dict[str, Tuple[NDArray, NDArray]]]]:
        """
        Parallel version using ProcessPoolExecutor.

        Args:
            directory: Root directory to scan.
            max_workers: Number of worker processes (default: CPU count).
            skip_on_error: If True, skip files that cause exceptions; if False, re-raise.
            recursive: If True, scan subdirectories recursively.
        """
        pattern = "**/*.json" if recursive else "*.json"
        files = list(directory.glob(pattern))
        if not files:
            return

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self._process_one, file_path, directory): file_path
                for file_path in files
            }
            # Yield results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    if result is not None:
                        yield result
                except Exception as e:
                    if skip_on_error:
                        self.logger.warning(f"Skipping {file_path} (parallel): {e}")
                        continue
                    raise

    def _process_one(
        self,
        file_path: Path,
        base_dir: Path
    ) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Tuple[NDArray, NDArray]]]]:
        """
        Internal helper for parallel processing.
        """
        try:
            rel = str(file_path.relative_to(base_dir))
            model = self.read_model(file_path)
            metadata = model.to_dict_clean()
            spectra = {q: model.get_spectrum(q) for q in model.list_quantities()}
            return rel, metadata, spectra
        except Exception as e:
            # Exception will be caught in the main thread if skip_on_error=True
            raise RuntimeError(f"Error processing {file_path}: {e}") from e


# -----------------------------------------------------------------------------
# Command-line example
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Example usage
    import sys
    logging.basicConfig(level=logging.INFO)
    reader = CIEJSONReader(strict=True)

    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if path.is_file():
            try:
                model = reader.read_model(path)
                print(f"File: {path.name}")
                print(f"Quantities: {model.list_quantities()}")
                for q in model.list_quantities():
                    wl, vals = model.get_spectrum(q)
                    print(f"  {q}: {len(wl)} points, range {wl[0]:.1f}–{wl[-1]:.1f} nm")
            except ValidationError as e:
                print(f"Validation error: {e}")
        elif path.is_dir():
            print("Batch processing (sequential)...")
            for rel, meta, spectra in reader.batch_process(path, skip_on_error=True):
                print(f"  {rel}: {list(spectra.keys())}")
    else:
        print("Usage: python colorengine_ciejson.py <file_or_directory>")