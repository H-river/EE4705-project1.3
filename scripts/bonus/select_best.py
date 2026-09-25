#!/usr/bin/env python
"""Pick a run's checkpoint: max VAL success, ties -> latest step; link runs/bonus/best/<label>.
    python scripts/bonus/select_best.py <run_name> [label]"""
import csv, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
run = sys.argv[1]; label = sys.argv[2] if len(sys.argv) > 2 else run
rows = list(csv.DictReader((ROOT / "runs/bonus/curves" / f"{run}.csv").open()))
best = max(rows, key=lambda r: (float(r["rate"]), int(r["step"])))
link = ROOT / "runs/bonus/best" / label
if link.is_symlink() or link.exists():
    link.unlink()
link.symlink_to(f"../train/{run}/checkpoints/{int(best['step']):06d}/pretrained_model")
print(f"{label} -> {run} step {best['step']} VAL {best['success']}/{best['n']}")
