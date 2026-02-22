# -*- coding: utf-8 -*-
# Loom: Weaving the mathematics of light in thin film systems
#
# Copyright (c) 2026 opticsWolf
#
# SPDX-License-Identifier: LGPL-3.0-or-later

[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "Loom"
version = "0.1.0"
authors = [
  { name="opticsWolf", email="opticswolf@protonmail.com" },
]
description = "A high-performance optical engine utilizing a Scattering Matrix algorithm."
readme = "README.md"
requires-python = ">=3.10"
license = "LGPL-3.0-or-later"
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
    "Operating System :: OS Independent",
    "Topic :: Scientific/Engineering :: Physics",
]
dependencies = [
    "numpy>=1.22.0",
    "numba>=0.56.0",
    "scipy>=1.8.0",
]

[project.urls]
"Homepage" = "https://github.com/opticsWolf/loom"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"

[tool.mypy]
python_version = "3.10"
strict = true
ignore_missing_imports = true