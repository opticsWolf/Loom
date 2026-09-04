# -*- coding: utf-8 -*-
"""
Loom: Weaving the mathematics of light in thin film systems
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""

import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Union

def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML config file into a plain dict."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_yaml(data: Any, path: Union[str, Path]) -> None:
    """Write a plain dict to a YAML config file."""
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a JSON config file into a plain dict."""
    with open(path, 'r') as f:
        return json.load(f)

def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """Write a plain dict to a JSON config file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)