#!/usr/bin/env python
"""Stage 6 (bonus): manipulation-mode trial sets for the four cells.

eval/trials/bonus_<cell>/<cell>_<i>.yaml, i = 0..n-1, scenes from
``sample_scene(rng([seed, i]), **CELLS[cell])`` — the SAME scenes and seeds
as scripts/bonus/eval_grasp.py (C1 1001, C2 1002, C3 1003, C4 1004), so the
skill-level and full-executor numbers are on identical layouts.

    python scripts/bonus/make_cells.py --n 30
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/bonus"))

import collect_grasp_demos as cg  # noqa: E402
from eval_grasp import CELLS, DEFAULT_SEED  # noqa: E402

INSTRUCTION = {"stone": "Move the gray stone to the red area.", "cube": "Move the blue cube to the red area.",
               "bottle": "Move the green bottle to the red area."}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--cells", default="C1,C2,C3,C4")
    args = ap.parse_args(argv)
    for cell in args.cells.split(","):
        out = ROOT / "eval/trials" / f"bonus_{cell.lower()}"
        out.mkdir(parents=True, exist_ok=True)
        seed = DEFAULT_SEED[cell]
        for i in range(args.n):
            rng = np.random.default_rng([seed, i])
            cls, objs, robot = cg.sample_scene(rng, **CELLS[cell])
            trial = {
                "id": f"{cell.lower()}_{i:02d}",
                "category": f"bonus_{cell}",
                "instruction": INSTRUCTION[cls],
                "scene": {"seed": seed * 100 + i,
                          "objects": [{"name": n, "pos": [round(v, 4) for v in p], "yaw": round(y, 4)}
                                      for n, p, y in objs],
                          "robot_init": {k: round(v, 4) for k, v in robot.items()}},
                "clarification_responses": [],
                "expected": {"target": cls, "region": "red_region", "feasible": True,
                             "events_required": ["GRASP", "MOVE_TO", "PLACE", "STOP"]},
            }
            (out / f"{trial['id']}.yaml").write_text(yaml.safe_dump(trial, sort_keys=False))
        print(f"{cell}: {args.n} trials -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
