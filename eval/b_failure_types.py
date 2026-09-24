"""Final2 6.2: Student B failure-type histogram from every audit on disk (0 live calls).

    python -m eval.b_failure_types [--runs runs] [--md docs/validation/B_FAILURE_TYPES.md]
                                   [--fig docs/night_run/figs/b_failure_types.png]

Types: WRONG_GOAL (wrong or unknown class / changed goal), WRONG_BINDING
(right class, wrong or missing instance), HALLUCINATED_INSTANCE (an ID that is
not in the scene), INVALID_PARAMS (schema / action-field / plan-structure
violations), WRONG_STATUS (READY vs SEARCH vs CLARIFICATION vs INFEASIBLE).

Three sources, counted separately:
1. labelled benchmark cases (every result.json + case.json under --runs):
   first-response errors and final errors against the expected answer;
2. contract validation errors of every B audit (repairs; no label needed);
3. e2e trials whose failure the final tables attribute to B.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
from collections import Counter

TYPES = ("WRONG_GOAL", "WRONG_BINDING", "HALLUCINATED_INSTANCE", "INVALID_PARAMS", "WRONG_STATUS")


def classify(error: str, scene_ids: set[str] | None = None) -> str:
    e = error
    m = re.match(r"(object|region)_id: expected (\S*), got (\S*)", e)
    if m:
        got = m.group(3).strip("'\"")
        if got and got != "None" and scene_ids is not None and got not in scene_ids:
            return "HALLUCINATED_INSTANCE"
        return "WRONG_BINDING"
    if e.startswith(("status:", "Unexpected search", "READY needs", "NEEDS_SEARCH may only",
                     "Non-action status", "READY cannot", "READY must")):
        return "WRONG_STATUS"
    if "Unknown perceived" in e or "not in scene" in e:
        return "HALLUCINATED_INSTANCE"
    if e.startswith(("Unknown object class", "Unknown region class", "Original goal must keep")) \
            or "supported classes" in e or "Held object differs" in e:
        return "WRONG_GOAL"
    if ("color mismatch" in e or "ID/class/role mismatch" in e or "Ambiguous object must not be bound" in e
            or "must target the goal" in e or "must carry the goal" in e):
        return "WRONG_BINDING"
    return "INVALID_PARAMS"


def benchmark(root: pathlib.Path) -> Counter:
    c = Counter()
    for path in root.rglob("result.json"):
        case = path.with_name("case.json")
        try:
            result, spec = json.loads(path.read_text()), json.loads(case.read_text())
        except (OSError, ValueError):
            continue
        scene = spec.get("scene") or {}
        ids = {g.get("instance_id") for g in scene.get("objects", []) + scene.get("regions", [])}
        for stage in result.get("stages", []):
            for err in stage.get("errors") or []:
                c[classify(err, ids)] += 1
    return c


def validation(root: pathlib.Path) -> Counter:
    c, seen = Counter(), set()
    for path in root.rglob("*.json"):
        if path.name in ("result.json", "case.json", "summary.json") or "cache" in path.parts:
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not (isinstance(data, dict) and "system_prompt" in data and "responses" in data):
            continue
        key = data.get("audit_path") or str(path)
        if key in seen or data.get("response_source") not in ("live", "cache"):
            continue
        seen.add(key)
        ids = {g.get("instance_id") for g in ((data.get("input") or {}).get("scene") or {}).get("instances", [])}
        for r in data["responses"]:
            err = r.get("validation_error")
            if err:
                for part in err.split("; "):
                    c[classify(part, ids)] += 1
    return c


def e2e(root: pathlib.Path) -> Counter:
    c = Counter()
    for path in root.rglob("rows.json"):
        try:
            rows = json.loads(path.read_text())["rows"]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        for r in rows:
            if r.get("correct") or r.get("module") != "B":
                continue
            cause = r.get("cause", "")
            c["INVALID_PARAMS" if "rejected by the contract" in cause else "WRONG_STATUS"] += 1
    return c


def figure(sources: dict[str, Counter], out: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, hue = "#0b0b0b", "#52514e", "#2a78d6"
    fig, axes = plt.subplots(1, len(sources), figsize=(4.2 * len(sources), 2.8), sharey=True)
    for ax, (title, counts) in zip(axes, sources.items()):
        values = [counts.get(t, 0) for t in TYPES]
        ax.barh(range(len(TYPES)), values, color=hue, height=0.6)
        for y, v in enumerate(values):
            ax.text(v, y, f" {v}", va="center", color=ink, fontsize=9)
        ax.set_title(f"{title} (n={sum(values)})", color=ink, fontsize=10, loc="left")
        ax.set_yticks(range(len(TYPES)), [t.replace("_", " ").lower() for t in TYPES], color=muted, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, max(values + [1]) * 1.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(muted)
        ax.spines["bottom"].set_color(muted)
        ax.tick_params(axis="x", colors=muted, labelsize=8)
    fig.suptitle("Student B failure types (all audits on disk)", color=ink, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=pathlib.Path, default=pathlib.Path("runs"))
    ap.add_argument("--md", type=pathlib.Path, default=pathlib.Path("docs/validation/B_FAILURE_TYPES.md"))
    ap.add_argument("--fig", type=pathlib.Path, default=pathlib.Path("docs/night_run/figs/b_failure_types.png"))
    a = ap.parse_args(argv)
    sources = {"labelled benchmark errors": benchmark(a.runs),
               "contract validation errors": validation(a.runs),
               "B-attributed e2e failures": e2e(a.runs)}
    figure(sources, a.fig)
    lines = ["# Student B failure types", "",
             f"Generated by `python -m eval.b_failure_types` from every audit under `{a.runs}/` (0 live calls). "
             "Definitions in the module docstring. Figure: `" + str(a.fig) + "`.", "",
             "| type | " + " | ".join(sources) + " |", "|---|" + "---|" * len(sources)]
    for t in TYPES:
        lines.append(f"| {t} | " + " | ".join(str(c.get(t, 0)) for c in sources.values()) + " |")
    lines.append("| total | " + " | ".join(str(sum(c.values())) for c in sources.values()) + " |")
    a.md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
