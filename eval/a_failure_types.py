"""Student A failure-type histogram from every eval artifact on disk (0 live calls).

    python -m eval.a_failure_types [--runs runs] [--md docs/validation/A_FAILURE_TYPES.md]
                                   [--fig docs/night_run/figs/a_failure_types.png]

Types: REGION_OBJECT_CONFLATION (wrong kind classified, or the right kind but
the wrong instance bound -- e.g. a region and an object sharing near-identical
image coordinates), DUPLICATE_INSTANCE_MISS (one of two same-class instances
not reported -- the model misses an object that is genuinely present),
SCENE_DESCRIPTION_ERROR (describe()'s free-text caption misses a present
object or names one that is not there), LOCALIZATION_FAILURE (a genuinely
visible target left UNLOCALIZED, or lost during an episode), VERIFICATION_
MISMATCH (A's own PLACE verification disagrees with the oracle, in either
direction), SCHEMA_OR_API_ERROR (an exception, a non-live response source, or
a provider/content-filter failure that produced no usable answer at all).

Three sources, counted separately, mirroring eval.b_failure_types:
1. eval.grounding per-case result files (every ``a_*.json`` case record under
   --runs, from ``eval.grounding run``): target-selection misses classified by
   expected vs. predicted kind and by the localization flag;
2. eval.a_scene_description / eval.a_vqa_check summaries (every
   ``scene_description_summary.json`` / ``vqa_summary.json`` under --runs):
   per-case missed/hallucinated object mentions;
3. e2e trials whose failure the final tables (eval/final_table.py) attribute
   to A -- read from every ``rows.json`` under --runs.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
from collections import Counter

TYPES = ("REGION_OBJECT_CONFLATION", "DUPLICATE_INSTANCE_MISS", "SCENE_DESCRIPTION_ERROR",
         "LOCALIZATION_FAILURE", "VERIFICATION_MISMATCH", "SCHEMA_OR_API_ERROR")

_CASE_ID_RE = re.compile(r"^a_\d+$")


def classify_grounding(record: dict) -> str | None:
    """A single eval.grounding per-case record -> failure type, or None if correct."""
    if record.get("error"):
        return "SCHEMA_OR_API_ERROR"
    sources = record.get("response_sources") or []
    if any(s != "live" for s in sources):
        return "SCHEMA_OR_API_ERROR"
    if record.get("target_selection_correct"):
        return None
    expected, predicted = record.get("expected_kind"), record.get("predicted_kind")
    if expected != predicted:
        if "ambiguous" in (expected, predicted):
            return "DUPLICATE_INSTANCE_MISS"
        return "REGION_OBJECT_CONFLATION"
    if not record.get("metric_localized", True):
        return "LOCALIZATION_FAILURE"
    return "REGION_OBJECT_CONFLATION"  # right kind, right localization flag, wrong instance bound


def grounding(root: pathlib.Path) -> Counter:
    c, seen = Counter(), set()
    for path in root.rglob("a_*.json"):
        if not _CASE_ID_RE.match(path.stem):
            continue
        if path in seen:
            continue
        seen.add(path)
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if "expected_kind" not in record:
            continue  # not an eval.grounding per-case file (e.g. a captured scene folder)
        kind = classify_grounding(record)
        if kind:
            c[kind] += 1
    return c


def description(root: pathlib.Path) -> Counter:
    c = Counter()
    for name in ("scene_description_summary.json", "vqa_summary.json"):
        for path in root.rglob(name):
            try:
                summary = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            for case in summary.get("per_case", []):
                if case.get("missed") or case.get("hallucinated"):
                    c["SCENE_DESCRIPTION_ERROR"] += 1
    return c


def e2e(root: pathlib.Path) -> Counter:
    c = Counter()
    for path in root.rglob("rows.json"):
        try:
            rows = json.loads(path.read_text())["rows"]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        for r in rows:
            if r.get("correct") or r.get("module") not in ("A", "A/C"):
                continue
            cause = r.get("cause", "")
            if "verification" in cause or "false claim" in cause:
                c["VERIFICATION_MISMATCH"] += 1
            elif "target lost" in cause or "not localized" in cause or "search exhausted" in cause:
                c["LOCALIZATION_FAILURE"] += 1
    return c


def figure(sources: dict[str, Counter], out: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, hue = "#0b0b0b", "#52514e", "#2a78d6"
    fig, axes = plt.subplots(1, len(sources), figsize=(4.6 * len(sources), 2.8), sharey=True)
    if len(sources) == 1:
        axes = [axes]
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
    fig.suptitle("Student A failure types (all eval artifacts on disk)", color=ink, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=pathlib.Path, default=pathlib.Path("runs"))
    ap.add_argument("--md", type=pathlib.Path, default=pathlib.Path("docs/validation/A_FAILURE_TYPES.md"))
    ap.add_argument("--fig", type=pathlib.Path, default=pathlib.Path("docs/night_run/figs/a_failure_types.png"))
    a = ap.parse_args(argv)
    sources = {"grounding target-selection misses": grounding(a.runs),
               "scene-description / VQA misses": description(a.runs),
               "A-attributed e2e failures": e2e(a.runs)}
    figure(sources, a.fig)
    lines = ["# Student A failure types", "",
             f"Generated by `python -m eval.a_failure_types` from every eval artifact under `{a.runs}/` "
             "(0 live calls). Definitions in the module docstring. Figure: `" + str(a.fig) + "`.", "",
             "| type | " + " | ".join(sources) + " |", "|---|" + "---|" * len(sources)]
    for t in TYPES:
        lines.append(f"| {t} | " + " | ".join(str(c.get(t, 0)) for c in sources.values()) + " |")
    lines.append("| total | " + " | ".join(str(sum(c.values())) for c in sources.values()) + " |")
    a.md.parent.mkdir(parents=True, exist_ok=True)
    a.md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
