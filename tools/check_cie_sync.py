"""CI sync guard: re-extract the embedded color defaults from
`src/navette/data/CIE/` and byte-compare against `rust/navette/data/`.
Exit 1 on any drift (loud, not silent). Stdlib only."""
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import cie_defaults as cd

drift = 0
for name, extract in cd.TARGETS:
    target = cd.DATA / name
    if not target.exists():
        print(f"MISSING rust/navette/data/{name} (run extract_cie_defaults.py)")
        drift += 1
        continue
    want = cd.dump_canonical(extract())
    got = target.read_text(encoding="utf-8")
    if want != got:
        print(f"DRIFT rust/navette/data/{name} (re-run extract_cie_defaults.py)")
        drift += 1
print("cie sync OK" if not drift else f"cie sync FAILED ({drift} file(s))")
sys.exit(1 if drift else 0)
