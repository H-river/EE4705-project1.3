# Owner: Student B
"""E4: clarification chains. Four ambiguous instructions, each with a scripted
answer: B's plan() -> question -> answer -> replan(), timed per call.

    python -m eval.b_clarification_chain --out runs/final/e4 [--figs docs/night_run/figs]

Scenes are hand-built public A-style descriptions (no simulator, no oracle).
Writes <out>/chains.json and, with --figs, b_clarification_timeline.png.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
from dataclasses import replace

from core.types import ExecutionContext, GroundedObject, GroundStatus, PlanStatus, SceneDescription
from planner.config import QwenPlannerConfig
from planner.student_b import StudentBPlanner


def _obj(i, name, colour, pos):
    return GroundedObject(i, name, GroundStatus.LOCALIZED, bbox_xyxy=(100, 100, 140, 140), pos_world=pos,
                          confidence=.95, attributes={"color": colour}, frame_id=1)


def _scene(objects):
    region = GroundedObject("a1", "red_region", GroundStatus.LOCALIZED, bbox_xyxy=(300, 100, 460, 200),
                            pos_world=(.40, .30, .853), confidence=.95, kind="region", frame_id=1,
                            attributes={"color": "red"}, region_half_extents_xy=(.08, .08))
    return SceneDescription(objects=objects, regions=[region], frame_id=1)


TWO_STONES = [_obj("a2", "stone", "gray", (.40, -.15, .875)), _obj("a3", "stone", "dark_red", (.50, -.02, .875)),
              _obj("a4", "cube", "blue", (.35, .12, .875))]
THREE_KINDS = [_obj("a2", "stone", "gray", (.40, -.15, .875)), _obj("a4", "cube", "blue", (.40, .15, .875)),
               _obj("a5", "bottle", "green", (.55, -.35, .905))]

CASES = [
    {"id": "c1_stone_gray", "instruction": "move the stone to the red area", "objects": TWO_STONES,
     "answer": "the gray one", "expect_object": "a2"},
    {"id": "c2_rock_dark_red", "instruction": "put the rock on the red marker", "objects": TWO_STONES,
     "answer": "the dark red one", "expect_object": "a3"},
    {"id": "c3_object_cube", "instruction": "move the object to the red area", "objects": THREE_KINDS,
     "answer": "the blue cube", "expect_object": "a4"},
    {"id": "c4_where_cube", "instruction": "pick up the blue cube and put it down somewhere", "objects": THREE_KINDS,
     "answer": "on the red area", "expect_object": "a4"},
]


def run_case(case, config, out):
    planner = StudentBPlanner(replace(config, audit_dir=str(out / case["id"] / "audit")))
    scene = _scene(case["objects"])
    steps, t0 = [], time.monotonic()
    start = time.monotonic()
    plan = planner.plan(case["instruction"], scene)
    steps.append({"kind": "plan", "t0": start - t0, "t1": time.monotonic() - t0, "status": plan.status.value,
                  "question": plan.clarification_question})
    if plan.status is PlanStatus.NEEDS_CLARIFICATION:
        steps.append({"kind": "answer", "t0": steps[-1]["t1"], "t1": steps[-1]["t1"], "text": case["answer"]})
        start = time.monotonic()
        plan = planner.replan(case["instruction"], scene, [], ExecutionContext(scene), case["answer"])
        steps.append({"kind": "plan", "t0": start - t0, "t1": time.monotonic() - t0, "status": plan.status.value,
                      "question": plan.clarification_question})
    goal = planner.goal or {}
    asked = any(s.get("question") for s in steps if s["kind"] == "plan")
    ok = asked and plan.status is PlanStatus.READY and goal.get("object_id") == case["expect_object"] \
        and goal.get("region_id") == "a1"
    return {"id": case["id"], "instruction": case["instruction"], "answer": case["answer"], "steps": steps,
            "final_status": plan.status.value, "goal": goal, "asked": asked, "correct": ok,
            "actions": [a.skill.value for a in plan.actions], "api_stats": planner.client.stats.as_dict()}


def timeline_figure(chains, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from eval.b_metrics import INK, INK2, PALETTE, SURFACE
    fig, ax = plt.subplots(figsize=(8.0, 3.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c2")
    ax.tick_params(colors=INK2, labelsize=8, left=False)
    for y, c in enumerate(reversed(chains)):
        plans = [s for s in c["steps"] if s["kind"] == "plan"]
        for k, s in enumerate(plans):
            colour = PALETTE[0] if k == 0 else PALETTE[2]
            ax.barh(y, s["t1"] - s["t0"], left=s["t0"], height=0.36, color=colour)
            if s.get("question"):  # the final status is labelled at the bar's right end
                ax.text(s["t0"], y + 0.30, "asks: " + s["question"][:90], fontsize=6.5, color=INK, va="bottom")
        answers = [s for s in c["steps"] if s["kind"] == "answer"]
        for s in answers:
            ax.plot([s["t0"]], [y], marker="o", markersize=8, color=PALETTE[1], markeredgecolor=SURFACE, markeredgewidth=2)
            ax.text(s["t0"], y - 0.34, f"answer: “{s['text']}”", fontsize=6.5, color=INK2, va="top")
        mark = "✔" if c["correct"] else "✘"
        ax.text(max(p["t1"] for p in plans) + 0.2, y, f"{mark} {c['final_status']} {c['goal'].get('object_id', '')}",
                fontsize=7.5, color=INK, va="center")
    ax.set_yticks(range(len(chains)), [f"{c['id']}\n“{c['instruction'][:34]}”" for c in reversed(chains)], fontsize=7)
    ax.set_xlabel("seconds from the first planning call (blue: first plan, green: plan after the answer)",
                  color=INK2, fontsize=8)
    ax.set_title("Clarification chains: question → scripted answer → plan", color=INK, fontsize=10, loc="left")
    ax.set_ylim(-0.7, len(chains) - 0.2)
    xmax = max(s["t1"] for c in chains for s in c["steps"])
    ax.set_xlim(0, xmax * 1.35)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--figs", type=pathlib.Path, default=None)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    config = QwenPlannerConfig.from_env()
    chains = [run_case(c, config, args.out) for c in CASES]
    (args.out / "chains.json").write_text(json.dumps(chains, indent=2, ensure_ascii=False))
    for c in chains:
        print(f"{'OK ' if c['correct'] else 'BAD'} {c['id']}: " + " -> ".join(
            (s["status"] + (f" ASK '{s['question']}'" if s.get("question") else "")) if s["kind"] == "plan"
            else f"ANSWER '{s['text']}'" for s in c["steps"]))
    if args.figs:
        args.figs.mkdir(parents=True, exist_ok=True)
        timeline_figure(chains, args.figs / "b_clarification_timeline.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
