"""Final2 bookkeeping (0 live calls): stability table, R7 comparison, call ledger.

usage:
  python scripts/final2_tools.py calls <run_root> <label>     # append to runs/final2/CALLS.txt
  python scripts/final2_tools.py compare <ids> <rootA...> -- <rootB...>
  python scripts/final2_tools.py stability <root1> <root2>

Calls are counted from EVERY trial record on disk (provider-error reruns
included), not only the merged one, so CALLS.txt is a hard upper bound.
"""
from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
LEDGER = REPO / "runs/final2/CALLS.txt"


def run_calls(root: pathlib.Path) -> int:
    total = 0
    for p in pathlib.Path(root).glob("job_*/runs/*/*/trial_record.json"):
        stats = json.loads(p.read_text()).get("api_stats") or {}
        total += sum(int((stats.get(r) or {}).get("attempts", 0)) for r in ("A", "B"))
    return total


def ledger_total() -> int:
    if not LEDGER.exists():
        return 0
    total = 0
    for line in LEDGER.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and not line.startswith("#") and parts[0] != "TOTAL":
            total += int(parts[1])
    return total


def add_calls(label: str, n: int) -> int:
    lines = [l for l in (LEDGER.read_text().splitlines() if LEDGER.exists() else [])
             if not l.startswith("TOTAL")]
    if not lines:
        lines = ["# final2 live API calls (A+B attempts incl. retries/reruns); hard cap 2000"]
    lines.append(f"{label:<40} {n}")
    LEDGER.write_text("\n".join(lines) + "\n")
    total = ledger_total()
    LEDGER.write_text(LEDGER.read_text() + f"TOTAL{'':<35} {total}\n")
    return total


def rows(root) -> dict[str, dict]:
    data = json.loads((pathlib.Path(root) / "rows.json").read_text())
    return {r["trial"]: r for r in data["rows"]}


def compare(ids: list[str], side_a: list[str], side_b: list[str]) -> dict:
    """R7: mean correct count over ``ids`` for each side (runs missing a
    trial are skipped for that trial and reported)."""
    def side(roots):
        tables = [rows(r) for r in roots]
        counts = [sum(bool(t[i]["correct"]) for i in ids if i in t) for t in tables]
        per = {i: "".join("✔" if t[i]["correct"] else "✘" for t in tables if i in t) for i in ids}
        calls = [sum(t[i]["calls"] for i in ids if i in t) for t in tables]
        return {"counts": counts, "mean": sum(counts) / len(counts), "per_trial": per, "calls": calls}
    a, b = side(side_a), side(side_b)
    return {"A": a, "B": b, "delta": b["mean"] - a["mean"]}


def main(argv):
    cmd = argv[0]
    if cmd == "calls":
        n = run_calls(pathlib.Path(argv[1]))
        print(f"{argv[2]}: {n} calls; ledger total {add_calls(argv[2], n)}")
    elif cmd == "add":
        print(f"{argv[1]}: {argv[2]} calls; ledger total {add_calls(argv[1], int(argv[2]))}")
    elif cmd == "compare":
        ids = argv[1].split(",")
        cut = argv.index("--")
        res = compare(ids, argv[2:cut], argv[cut + 1:])
        for i in ids:
            print(f"{i:<32} {res['A']['per_trial'][i]:>4}  ->  {res['B']['per_trial'][i]}")
        print(f"mean A {res['A']['mean']:.2f} {res['A']['counts']} calls {res['A']['calls']}")
        print(f"mean B {res['B']['mean']:.2f} {res['B']['counts']} calls {res['B']['calls']}")
        print(f"delta {res['delta']:+.2f}")
    elif cmd == "stability":
        a, b = rows(argv[1]), rows(argv[2])
        for tid in sorted(a):
            n = int(a[tid]["correct"]) + int(b[tid]["correct"])
            print(f"{tid:<34} {n}/2 {'stable-pass' if n == 2 else 'unstable' if n == 1 else 'stable-fail'}")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
