# -*- coding: utf-8 -*-
"""
Loom: Weaving the mathematics of light in thin film systems
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""
import re
import io
from collections import Counter

import numpy as np
import polars as pl

class CSVData:
    def __init__(self):
        """Initialize the CSV database with default settings."""
        self.current_file = None
        self.dataframe = None
        self.raw_content = None
        self.separator = ','
        self.decimal_sep = '.'
        self.skip_initial_rows = 0
        self.skip_end_rows = 0

        # Interpolation mode default (crop | extend)
        self.interp_mode = "crop"

        # Manual scaling options
        self.manual_scaling = False
        self.wavelength_unit = 'nm'  # Default wavelength unit
        self.data_type_unit = '%'    # Default data type unit

        # Auto-sorting option (new)
        self.auto_sorting = True

        # Predefined conversion factors for manual mode
        self.wavelength_factors = {
            'Å' : 0.1,
            'nm': 1.0,
            'µm': 1000.0,
            'mm': 1000000.0
        }
        self.data_type_factors = {
            "Percentage (%)": 0.01,       # Convert percentage to decimal (e.g., 50% -> 0.5)
            "Normalized (0 – 1)": 1.0,      # No conversion needed for normalized data
            "Radians (°→rad)": np.pi/180,           
        }

    def detect_separator(self, raw_data: str) -> str:
        """
        Detect the most likely separator from common options.

        Args:
            raw_data: The content of the CSV file as a string

        Returns:
            The detected separator character
        """
        if not raw_data.strip():
            return ','

        lines = []
        for line in raw_data.splitlines():
            stripped = line.strip()
            if stripped and len(lines) < 5:  # Take first 5 non-empty lines
                lines.append(stripped)

        common_seps = [',', ';', '\t', '|']
        sep_scores = {}

        for sep in common_seps:
            score = 0
            expected_fields = None
            valid_lines = 0
            for line in lines:
                fields = line.split(sep)
                num_fields = len(fields)
                if num_fields >= 2:
                    valid_lines += 1
                    if expected_fields is None:
                        expected_fields = num_fields
                    else:
                        if num_fields == expected_fields:
                            score += 1

            sep_scores[sep] = (score, valid_lines)

        best_sep = ','
        best_score = -1
        for sep in common_seps:
            score, valid_lines = sep_scores[sep]
            if valid_lines > 0 and score >= valid_lines * 0.8:  # At least 80% consistency
                if score > best_score or (score == best_score and sep == ','):
                    best_score = score
                    best_sep = sep

        if best_score < 2:
            first_line = lines[0] if lines else ''
            chars = [c for c in first_line if not c.isalnum() and not c.isspace()]
            if chars:
                most_common_char, _ = Counter(chars).most_common(1)[0]
                best_sep = most_common_char

        return best_sep

    def detect_decimal_sign(self, raw_data: str) -> str:
        """
        Detect the decimal sign used in numerical values.

        Args:
            raw_data: The content of the CSV file as a string
            separator: Currently selected separator character

        Returns:
            The detected decimal sign (either '.' or ',')
        """
        lines = []
        for line in raw_data.splitlines():
            stripped = line.strip()
            if stripped and len(lines) < 5:  # Take first 5 non-empty lines
                lines.append(stripped)

        decimal_sign = '.'

        for line in lines:
            fields = [f.strip('"').strip("'") for f in line.split(self.separator)]
            for field in fields:
                if re.match(r'^[-+]?\d+[\.,]\d+$', field):
                    if '.' in field:
                        return '.'
                    elif ',' in field:
                        return ','

        for line in lines:
            fields = [f.strip('"').strip("'") for f in line.split(self.separator)]
            for field in fields:
                if re.match(r'^[-+]?\d{1,3}(,\d{3})*(\.[0-9]+)?$', field):
                    if '.' in field and not (',' in field and field.count(',') > 1):
                        return '.'
                elif re.match(r'^[-+]?\d{1,3}(,\d{3})*,[0-9]+$', field) and ',' in field:
                    if not '.' in field:
                        return ','

        return decimal_sign

    def detect_header_rows_from_df(self, df: pl.DataFrame) -> int:
        """
        Detect how many rows at the top are likely headers.

        Args:
            df: The DataFrame to analyze

        Returns:
            Number of header rows detected
        """
        if len(df) < 2:
            return 0

        header_count = 0
        for row in range(min(10, len(df))):
            numeric_cells = 0
            non_empty_cells = 0
            for col in df.columns:
                cell_value = df[row, col]
                str_value = str(cell_value).strip().strip('"').strip("'")
                if str_value != "":
                    non_empty_cells += 1
                    try:
                        float(str_value.replace(',', '.').replace(' ', ''))
                        numeric_cells += 1
                    except ValueError:
                        pass
            if non_empty_cells == 0 or numeric_cells < (non_empty_cells / 2):
                header_count += 1
            else:
                break
        return min(header_count, len(df) - 1)

    def get_conversion_factor(self, col_name: str) -> float:
        """
        Determine unit conversion factor based on column name.

        Args:
            col_name: The column name to analyze

        Returns:
            Conversion factor (0.01 for %, 0.001 for µm, etc.)
        """
        col_name = col_name.lower()

        if re.search(r'(?:%|percent| per[- ]cent)', col_name):
            return 0.01
        elif re.search(r'(?:°|(?:^|\W)(?:degree|deg|degs?)(?=$|\W))', col_name):
            return np.pi / 180
        elif re.search(r'(?:^|\W)(?:nm|nanometer)(?=$|\W)', col_name, re.IGNORECASE):
            return 1.0
        elif re.search(r'(?:^|\W)(?:µm|um|micron)(?=$|\W)', col_name, re.IGNORECASE):
            return 1000
        else:
            return 1.0

    def is_wavelength_column(self, name: str) -> bool:
        """
        Check if column represents wavelength.

        Args:
            name: The column name to check

        Returns:
            True if the column appears to represent wavelength
        """
        if not name:
            return False
        pattern = r"(?:^|[^a-z0-9])(wavl|wavelength|wavls|wvls|wvl|wavelengths|lambda|wellenl)(?:[^a-z0-9]|$)"
        return bool(re.search(pattern, name, re.IGNORECASE))

    def interpolate_missing(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Interpolate missing values in the DataFrame.
        Handles multiple wavelength columns by unifying them.
    
        Returns:
            DataFrame with missing values interpolated and unified.
        """
        # Identify wavelength columns
        wl_cols = [c for c in df.columns if self.is_wavelength_column(c)]
        if not wl_cols:
            return df
    
        # If multiple wavelength columns, unify them
        if len(wl_cols) > 1:
            arrays = []
            for c in wl_cols:
                try:
                    x = df[c].drop_nulls().to_numpy().astype(float)
                    arrays.append(x)
                except Exception:
                    continue
        
            if not arrays:
                return df
        
            # Combine, sort, and deduplicate all wavelength values
            if self.auto_sorting:
                # Sorted unique values – keep the original behaviour
                merged_wavelengths = np.unique(np.sort(np.concatenate(arrays)))
            else:
                # Preserve the first‑seen order (no explicit sort)
                concat = np.concatenate(arrays)                     # all values in one array
                _, idx = np.unique(concat, return_index=True)       # indices of first occurrence
                merged_wavelengths = concat[np.sort(idx)]           # keep appearance order
        
            if self.interp_mode == "crop":
                wl_min = max(arr.min() for arr in arrays)
                wl_max = min(arr.max() for arr in arrays)
                base_wavelengths = merged_wavelengths[
                    (merged_wavelengths >= wl_min) & (merged_wavelengths <= wl_max)
                ]
            else:
                base_wavelengths = merged_wavelengths
    
            # Interpolate all numeric columns to unified wavelengths
            new_cols = {"Wavelength": base_wavelengths}
            for col in df.columns:
                if col in wl_cols:
                    continue
                try:
                    for wl_col in wl_cols:
                        y = df[col].to_numpy().astype(float)
                        x = df[wl_col].to_numpy().astype(float)
                        mask = ~np.isnan(y)
                        if mask.sum() > 1:
                            y_interp = np.interp(base_wavelengths, x[mask], y[mask])
                            new_cols[col] = y_interp
                            break
                except Exception:
                    continue
            df = pl.DataFrame(new_cols)
            return df
    
        # Single wavelength column: standard interpolation
        wl_col = wl_cols[0]
        x = df[wl_col].to_numpy().astype(float)
        for col in df.columns:
            if col == wl_col:
                continue
            y_raw = df[col].to_numpy()
            y_clean = np.array([
                float(v) if v is not None and str(v).strip() != "" else np.nan
                for v in y_raw
            ])
            mask = ~np.isnan(y_clean)
            if mask.sum() > 1:
                y_interp = np.interp(x, x[mask], y_clean[mask])
                df = df.with_columns(pl.Series(col, y_interp))
        return df
    
    def sort_by_wavelength_desc(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Sort DataFrame by wavelength column in descending order.

        Args:
            df (pl.DataFrame): Input DataFrame.

        Returns:
            Sorted DataFrame
        """
        last_wl_col = None
        for col in df.columns:
            if self.is_wavelength_column(col):
                last_wl_col = col

        if last_wl_col:
            df = df.sort(last_wl_col)
        return df

    def load_csv(self, file_path: str):
        """
        Load a CSV file and store its raw contents.

        Args:
            file_path: Path to the CSV file
        """
        self.current_file = file_path
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                self.raw_content = f.read()
                # Auto-detect settings if needed
            if self.separator == ',' and self.decimal_sep == '.':
                self.separator = self.detect_separator(self.raw_content)
                self.decimal_sep = self.detect_decimal_sign(self.raw_content)           
        except Exception as e:
            self.raw_content = f"Error reading file: {e}"
            raise

    def import_data(self):
        """
        Main method to import and process the CSV data.

        Returns:
            The processed DataFrame
        """
        if not self.current_file or self.raw_content is None:
            raise ValueError("No file loaded")

        try:
            csv_data = io.StringIO(self.raw_content)

            # First pass: detect headers, but skip initial rows
            df_sample = pl.read_csv(
                csv_data,
                separator=self.separator if self.separator else ",",
                decimal_comma=(self.decimal_sep == ","),
                infer_schema_length=10,  # Read first 10 rows for analysis
                has_header=False,        # Don't assume headers - we'll detect them
                new_columns=None,         # Use default column names
                skip_rows=self.skip_initial_rows
            )

            # Detect header rows on the sample data (after skipping initial rows)
            headers = self.detect_header_rows_from_df(df_sample)

            # Second pass: read with proper configuration, also skipping initial rows
            csv_data.seek(0)
            if headers > 0:
                combined_headers = []
                for col in df_sample.columns:
                    str_values = [
                        str(df_sample[row, col]).strip()
                        for row in range(headers)
                        if df_sample[row, col] is not None and str(df_sample[row, col]).strip() != ""
                    ]
                    combined_headers.append(" ".join(str_values))

                # Ensure unique column names
                counts = Counter(combined_headers)
                seen = Counter()
                unique_headers = []
                for h in combined_headers:
                    if counts[h] > 1:
                        seen[h] += 1
                        unique_headers.append(f"{h}_{seen[h]}")
                    else:
                        unique_headers.append(h)

                # Reset file pointer and read with proper configuration
                csv_data.seek(0)
                df = pl.read_csv(
                    csv_data,
                    separator=self.separator if self.separator else ",",
                    decimal_comma=(self.decimal_sep == ","),
                    infer_schema_length=None,
                    has_header=False,  # We'll specify columns explicitly
                    new_columns=unique_headers,
                    skip_rows=self.skip_initial_rows + headers  # Skip both initial rows and header rows
                )
            else:
                csv_data.seek(0)
                df = pl.read_csv(
                    csv_data,
                    separator=self.separator if self.separator else ",",
                    decimal_comma=(self.decimal_sep == ","),
                    infer_schema_length=None,
                    has_header=False,
                    skip_rows=self.skip_initial_rows
                )

            # Remove end rows if specified
            total_rows = len(df)
            if self.skip_end_rows > 0 and total_rows > self.skip_end_rows:
                df = df.slice(0, total_rows - self.skip_end_rows)

            # Clean data: remove fully null/empty rows and columns
            # Remove rows that are fully null or empty strings
            df = df.filter(
                ~pl.all_horizontal(
                    [
                        (pl.col(c).is_null()) | (str(pl.col(c).cast(pl.Utf8)).strip() == "")
                        for c in df.columns
                    ]
                )
            )

            # Remove columns that are fully null or empty strings
            non_empty_cols = [
                c for c in df.columns
                if not df.select(
                    ((pl.col(c).is_null()) | (str(pl.col(c).cast(pl.Utf8)).strip() == ""))
                    .all()
                )[0, 0]
            ]
            df = df.select(non_empty_cols)

            # Apply unit conversions
            for col in df.columns:
                if self.manual_scaling:
                    factor = 1.0
                    # Check if wavelength column and get appropriate factor
                    if self.is_wavelength_column(col):
                        factor = self.wavelength_factors.get(self.wavelength_unit, 1.0)
                    else:
                        try:
                            df[col].cast(pl.Float64)  # This will raise exception for non-numeric columns
                            factor = self.data_type_factors.get(self.data_type_unit, 1.0)
                        except Exception:
                            pass

                    if factor != 1.0:
                        try:
                            df = df.with_columns(pl.col(col).cast(pl.Float64) * factor)
                        except Exception as e:
                            print(f"Skipping unit conversion for {col}: {e}")
                else:
                    # Original automatic scaling
                    factor = self.get_conversion_factor(col)
                    if factor != 1.0:
                        try:
                            df = df.with_columns(pl.col(col).cast(pl.Float64) * factor)
                        except Exception as e:
                            print(f"Skipping unit conversion for {col}: {e}")

            # Sort and interpolate
            if self.auto_sorting:  # Apply auto-sorting only when enabled
                df = self.sort_by_wavelength_desc(df)
            df = self.interpolate_missing(df)

            self.dataframe = df
            return df

        except Exception as e:
            raise ValueError(f"Import error ({self.separator}/{self.decimal_sep}): {e}") from e