# -*- coding: utf-8 -*-
"""Phase 1 program documents: envelope gate, partial + full restore,
prefix namespaces, legacy flat files."""
import numpy as np
import pytest

from navette.config import load_document, load_program
from navette.config.program import PROGRAM_SCHEMA_VERSION

WL = np.array([400., 600., 800.])
EXAMPLE = "src/navette/config/example_program.yaml"


def test_version_gate(tmp_path):
  import json
  from navette._structure import gate_document
  def gate(doc):
    p = tmp_path / "d.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return load_document(str(p))
  with pytest.raises(ValueError, match="unsupported"):
    gate({"kind": "materials", "schema_version": 99, "materials": []})
  with pytest.raises(ValueError, match="unsupported"):
    gate({"kind": "materials", "materials": []})  # missing
  with pytest.raises(ValueError, match="unknown"):
    gate({"kind": "pipeline", "schema_version": 1})
  kind, name, payload = gate({"schema_version": 1, "kind": "groups", "groups": []})
  assert (kind, name, payload) == ("groups", None, [])


def test_full_restore():
  prog = load_program(EXAMPLE, WL)
  assert prog.name == "AR demo"
  assert prog.materials.contains("L") and prog.materials.contains("H")
  assert set(prog.groups) == {"H"}
  assert set(prog.structures) == {"ar"}
  st = prog.structures["ar"]
  assert [layer.material for layer in st.layer_list] == ["L", "H"]
  assert prog.architect is not None
  assert len(prog.architect) == 1
  assert prog.architect.blocks[0].label == "main"
  # Graded layer survived the trip with its flags.
  graded = st.layer_list[1]
  assert graded.inhomogen and not graded.optimize and not graded.needle


def test_prefix_namespace():
  a = load_program(EXAMPLE, WL)
  b = load_program(EXAMPLE, WL, prefix="run2_")
  assert set(b.structures) == {"run2_ar"}
  assert b.materials.contains("run2_L")
  assert [layer.material for layer in
          b.structures["run2_ar"].layer_list] == ["run2_L", "run2_H"]
  assert set(b.groups) == {"run2_H"}
  # Unprefixed load is unaffected.
  assert set(a.structures) == {"ar"}


def test_partial_restore_and_context():
  kind, name, payload = load_document(EXAMPLE)
  assert kind == "program" and name == "AR demo"
  import navette.config.program as P
  mats = P.load_materials(payload["materials"], WL)
  assert mats.contains("H")
  groups = P.load_groups(payload["groups"])
  assert set(groups) == {"H"}
  # Structures resolve against a context provider (file section absent).
  single = {"label": "solo", "layers": [{"material_code": "H", "thickness_nm": 10.0}]}
  st = P.load_structure(single, mats, groups)
  assert [layer.material for layer in st.layer_list] == ["H"]
  with pytest.raises(KeyError, match="needs materials"):
    P.load_structure(single, None, groups)


def test_missing_ref_names_section():
  import navette.config.program as P
  mats = P.load_materials([
    {"name": "H", "code": "H", "model": "Konstant", "params": {"n": 2.0, "k": 0.0}}], WL)
  bad = {"label": "x", "layers": [{"material_code": "NOPE", "thickness_nm": 5.0}]}
  with pytest.raises(KeyError, match="NOPE"):
    P.load_structure(bad, mats, {})


def test_legacy_flat_files(tmp_path):
  import yaml
  flat_mat = {"materials": [
    {"name": "H", "code": "H", "model": "Konstant", "params": {"n": 2.0, "k": 0.0}}]}
  p = tmp_path / "m.yaml"
  p.write_text(yaml.safe_dump(flat_mat))
  kind, _, payload = load_document(str(p))
  assert kind == "materials"
  import navette.config.program as P
  assert P.load_materials(payload, WL).contains("H")
  flat_stack = {"layers": [{"material_code": "H", "thickness_nm": 5.0}]}
  q = tmp_path / "s.yaml"
  q.write_text(yaml.safe_dump(flat_stack))
  kind, _, payload = load_document(str(q))
  assert kind == "structure"
  prog = load_program(str(q), WL,
                      context={"materials": P.load_materials(flat_mat["materials"], WL)})
  assert set(prog.structures) == {"stack"}


def _assert_shared_blocks(prog, label):
  named = prog.structures[label]
  block_shell = prog.architect.blocks[0].structure
  assert block_shell._inner.core_id() == named._inner.core_id()
  # Edits propagate through the shared handle.
  before = prog.architect.get_global_layer_count()
  named._inner.append_layer(named._inner.layer_list[0])
  try:
    assert prog.architect.get_global_layer_count() == before + 1
  finally:
    named._inner.remove_layer(len(named._inner.layer_list) - 1)


def test_restored_blocks_alias_structures():
  # Whole-document native path.
  _assert_shared_blocks(load_program(EXAMPLE, WL), "ar")


def test_context_path_blocks_alias_structures():
  # Section-wise context path keeps the same invariant.
  from navette.config.program import load_materials
  kind, name, payload = load_document(EXAMPLE)
  mats = load_materials(payload["materials"] if kind == "program"
                        else payload, WL)
  prog = load_program(EXAMPLE, WL, context={"materials": mats})
  _assert_shared_blocks(prog, "ar")
