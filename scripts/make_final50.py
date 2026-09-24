"""Build eval/trials/final50: the 35 existing e2e trials (f01..f35) plus 15 new
ones (f36..f50).  Deterministic; rerunning overwrites the directory.

expected.outcome is the scoring label for the final run: ``success`` (claimed
and achieved), ``reject`` (REFUSED) or ``clarify`` (a clarification is asked,
the scripted answer consumed, and the task then claimed and achieved).
"""
from __future__ import annotations

import math
import pathlib
import random
import shutil

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
TRIALS = ROOT / "eval" / "trials"
OUT = TRIALS / "final50"
SOURCES = ["smoke", "student_c", "student_c_v2", "student_c_evaluation"]

Z = {"stone": 0.88, "stone2": 0.88, "cube": 0.88, "bottle": 0.915}
BASE = {"stone": (0.40, -0.15), "cube": (0.40, 0.15), "bottle": (0.55, -0.35)}
REGION_XY = (0.40, 0.30)
TABLE_X, TABLE_Y = (0.25, 0.65), (-0.5, 0.5)  # core.scene_geometry.table_aabb()
EDGE_MARGIN = 0.06
INSTRUCTION = {"stone": "Move the gray stone to the red area.",
               "cube": "Move the blue cube to the red area.",
               "bottle": "Move the green bottle to the red area."}


def expected(target, outcome="success", **extra):
    e = {"target": target, "region": "red_region", "feasible": outcome != "reject",
         "search_required": False, "clarification_required": outcome == "clarify",
         "events_required": ["GRASP", "PLACE"], "outcome": outcome}
    if outcome == "reject":
        e = {"feasible": False, "outcome": "reject"}
    e.update(extra)
    return e


def obj(name, x, y, yaw=0.0):
    d = {"name": name, "pos": [round(x, 3), round(y, 3), Z[name]]}
    if yaw:
        d["yaw"] = round(yaw, 3)
    return d


def jittered_scene(rng):
    """±8 cm around the standard layout, on the table, ≥ 10 cm apart, off the region."""
    while True:
        placed = {}
        for name, (x, y) in BASE.items():
            placed[name] = (x + rng.uniform(-0.08, 0.08), y + rng.uniform(-0.08, 0.08))
        pts = list(placed.values())
        on_table = all(TABLE_X[0] + EDGE_MARGIN <= x <= TABLE_X[1] - EDGE_MARGIN
                       and TABLE_Y[0] + EDGE_MARGIN <= y <= TABLE_Y[1] - EDGE_MARGIN for x, y in pts)
        apart = all(math.dist(a, b) >= 0.10 for i, a in enumerate(pts) for b in pts[i + 1:])
        off_region = all(math.dist(p, REGION_XY) >= 0.14 for p in pts)
        if on_table and apart and off_region:
            return placed


def write(trial):
    path = OUT / f"{trial['id']}.yaml"
    path.write_text(yaml.safe_dump(trial, sort_keys=False))


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    n = 0
    for src in SOURCES:
        for path in sorted((TRIALS / src).glob("*.yaml")):
            n += 1
            trial = yaml.safe_load(path.read_text())
            trial["id"] = f"f{n:02d}_{trial['id']}"
            trial["source_trial"] = f"{src}/{path.name}"
            e = trial.setdefault("expected", {})
            if not e.get("feasible", True):
                e["outcome"] = "reject"
            elif e.get("clarification_required"):
                e["outcome"] = "clarify"
            else:
                e["outcome"] = "success"
            write(trial)
    assert n == 35, n

    # f36..f41: scene variations (seeds 501..506).
    for k, target in enumerate(["stone", "cube", "bottle", "stone", "cube", "bottle"]):
        seed = 501 + k
        rng = random.Random(seed)
        placed = jittered_scene(rng)
        yaw0 = round(rng.uniform(-0.3, 0.3), 3)
        n += 1
        write({"id": f"f{n:02d}_scene_var_{seed}_{target}", "category": "final_scene_variation",
               "instruction": INSTRUCTION[target],
               "scene": {"seed": seed,
                         "objects": [obj(nm, x, y, rng.uniform(-0.5, 0.5)) for nm, (x, y) in placed.items()],
                         "robot_init": {"x": 0.0, "y": 0.0, "yaw": yaw0}},
               "clarification_responses": [], "expected": expected(target)})

    # f42..f46: paraphrases on the standard smoke_1 layout.
    std = [obj(nm, x, y) for nm, (x, y) in BASE.items()]
    paraphrases = [("could you move that rock onto the red marker", "stone"),
                   ("the grey one goes on the red patch", "stone"),
                   ("please put the blue block in the red zone", "cube"),
                   ("take the green bottle and set it down on the red square", "bottle"),
                   ("I'd like the cube moved over to the red target", "cube")]
    for k, (text, target) in enumerate(paraphrases):
        n += 1
        write({"id": f"f{n:02d}_paraphrase_{k + 1}_{target}", "category": "final_instruction_paraphrase",
               "instruction": text,
               "scene": {"seed": 511 + k, "objects": std, "robot_init": {"x": 0.0, "y": 0.0, "yaw": 0.0}},
               "clarification_responses": [], "expected": expected(target)})

    # f47..f48: two stones, the instruction names the colour.
    two = [("Move the gray stone to the red area.", "stone",
            [obj("stone", 0.45, -0.05), obj("stone2", 0.38, -0.22), obj("cube", 0.35, 0.12)]),
           ("Put the dark red stone on the red area.", "stone2",
            [obj("stone", 0.36, -0.20), obj("stone2", 0.46, -0.08), obj("bottle", 0.55, -0.35)])]
    for k, (text, target, objects) in enumerate(two):
        n += 1
        write({"id": f"f{n:02d}_two_stones_colour_{target}", "category": "final_two_stones_colour",
               "instruction": text,
               "scene": {"seed": 521 + k, "objects": objects, "robot_init": {"x": 0.0, "y": 0.0, "yaw": 0.0}},
               "clarification_responses": [], "expected": expected(target)})

    # f49: impossible request (stacking on another object is unsupported).
    n += 1
    write({"id": f"f{n:02d}_reject_stack", "category": "final_reject",
           "instruction": "Stack the blue cube on top of the green bottle.",
           "scene": {"seed": 531, "objects": std, "robot_init": {"x": 0.0, "y": 0.0, "yaw": 0.0}},
           "clarification_responses": [], "expected": expected(None, "reject")})

    # f50: two stones, no colour -> ask; the scripted answer picks the gray one.
    n += 1
    write({"id": f"f{n:02d}_clarify_two_stones", "category": "final_clarify",
           "instruction": "put the stone on the red area",
           "scene": {"seed": 541, "objects": [obj("stone", 0.36, -0.08), obj("stone2", 0.47, -0.20),
                                               obj("bottle", 0.56, -0.38)],
                     "robot_init": {"x": 0.0, "y": 0.0, "yaw": 0.0}},
           "clarification_responses": ["the gray one"], "expected": expected("stone", "clarify")})
    assert n == 50, n
    print(f"wrote {n} trials to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
