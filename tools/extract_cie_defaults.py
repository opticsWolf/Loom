"""Extract embedded color defaults from `src/navette/data/CIE/` into
`rust/navette/data/*.json`. Run when the upstream CIE sources change;
`tools/check_cie_sync.py` guards against unextracted drift."""
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import cie_defaults as cd

cd.DATA.mkdir(parents=True, exist_ok=True)
for name, extract in cd.TARGETS:
    (cd.DATA / name).write_text(cd.dump_canonical(extract()), encoding="utf-8")
    print(f"wrote rust/navette/data/{name}")
