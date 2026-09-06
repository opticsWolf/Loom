"""pipeline_from_config: typed configs -> native (DesignStack, contrast_map)."""
import numpy as np
import pytest

from navette.config.builders import pipeline_from_config
from navette.config.models import (
    KonstantParams, MaterialDefinition, LayerConfig, GroupConfig,
    NamedStructureConfig,
)

WL = np.array([500.0, 600.0])

def _mat(code, n):
    return MaterialDefinition(name=code, code=code, model="Konstant",
                              params=KonstantParams(n=n))

def _film(code, d, **kw):
    return LayerConfig(material_code=code, thickness_nm=d, **kw)

LIB = [_mat("air", 1.0), _mat("L", 1.45), _mat("H", 2.1), _mat("sub", 1.52)]

def test_builds_stack_and_contrast():
    films = [_film("L", 100.0), _film("H", 60.0)]
    # Half-space rows carry a dummy thickness (gt=0 guard); the driver
    # pins them at 0.0 regardless.
    air = LayerConfig(material_code="air", thickness_nm=1.0, layer_type=0,
                      optimize=False, needle=False)
    sub = LayerConfig(material_code="sub", thickness_nm=1.0, layer_type=2,
                      optimize=False, needle=False)
    s = NamedStructureConfig(label="t", layers=[air, *films, sub])
    stack, cmap = pipeline_from_config(s, LIB, WL, contrast={"H": "L"})
    assert stack.film_count() == 2
    assert set(cmap) == {"H"}

def test_per_film_flags_land_and_override():
    films = [_film("L", 100.0, optimize=False, needle=False, inhomogen=True),
             _film("H", 60.0)]
    s = NamedStructureConfig(label="t", layers=films)
    stack, _ = pipeline_from_config(s, LIB, WL,
                                    per_film_flags={"H": {"optimize": False}})
    # Graded + pinned (optimize/needle False) expands WITH the profile:
    # 1 L-slice-span + 1 H film, not 2 flat films.
    assert stack.film_count() > 2

def test_missing_rows_fall_back_to_constants():
    s = NamedStructureConfig(label="t", layers=[_film("L", 100.0)])
    stack, _ = pipeline_from_config(s, LIB, WL)
    assert stack.film_count() == 1

def test_duplicate_film_code_raises():
    s = NamedStructureConfig(label="t",
                             layers=[_film("L", 100.0), _film("L", 50.0)])
    with pytest.raises(ValueError, match="duplicate film"):
        pipeline_from_config(s, LIB, WL)

def test_unknown_material_raises():
    s = NamedStructureConfig(label="t", layers=[_film("X", 100.0)])
    with pytest.raises(KeyError, match="'X'"):
        pipeline_from_config(s, LIB, WL)

def test_two_ambients_raise():
    a = LayerConfig(material_code="air", thickness_nm=1.0, layer_type=0)
    s = NamedStructureConfig(label="t", layers=[a, a, _film("L", 10.0)])
    with pytest.raises(ValueError, match="ambient"):
        pipeline_from_config(s, LIB, WL)

def test_group_rides_along():
    g = GroupConfig(name="H")
    films = [_film("L", 100.0), _film("H", 60.0)]
    s = NamedStructureConfig(label="t", layers=films, groups=[g])
    stack, _ = pipeline_from_config(s, LIB, WL)
    assert stack.film_count() == 2
