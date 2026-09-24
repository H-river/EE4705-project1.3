"""Aggregate Student B planner audits into docs/validation/B_METRICS.md.

    python -m eval.b_metrics [--runs runs/night] [--out docs/validation/B_METRICS.md]
                             [--variation runs/night/planning_variation]
                             [--trials-root runs/final/run_1] [--figs docs/night_run/figs]

v2 adds: plan-status conversions (repair, REJECTED, READY-with-SEARCH ->
NEEDS_SEARCH), B memory usage (how often B planned from memory, the age
distribution, and the trial outcome of episodes that used it vs not, from the
``--trials-root`` e2e records), and figures (``--figs``).

Reads every B audit JSON (files carrying ``system_prompt`` and ``responses``)
under ``--runs``. Only responses whose source is ``live`` or ``cache`` count
(fixtures are ignored). Call types come from the planner input:
post-clarification (``clarification`` set), held-object replan
(``execution_context.held_instance_id`` set), other replan
(``history_total`` > 0) and initial plan.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
from collections import Counter, defaultdict


def load_audits(root: pathlib.Path) -> list[dict]:
    audits, seen = [], set()
    for path in sorted(root.rglob("*.json")):
        if "cache" in path.parts:
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not (isinstance(data, dict) and "system_prompt" in data and "responses" in data):
            continue
        key = data.get("audit_path") or str(path)
        if key in seen:
            continue
        seen.add(key)
        if data.get("response_source") not in ("live", "cache"):
            continue
        data["_file"] = str(path)
        audits.append(data)
    return audits


def call_type(audit: dict) -> str:
    inp = audit.get("input", {})
    if inp.get("clarification"):
        return "post-clarification"
    if (inp.get("execution_context") or {}).get("held_instance_id"):
        return "held-object replan"
    if inp.get("history_total", 0) > 0:
        return "other replan"
    return "initial plan"


def _rate(n: int, d: int) -> str:
    return f"{n}/{d} = {n / d:.1%}" if d else "n/a (0)"


def _stats(values: list[float]) -> str:
    if not values:
        return "n/a"
    return f"mean {statistics.mean(values):.1f}, median {statistics.median(values):.1f}"


def summarise(audits: list[dict]) -> str:
    n = len(audits)
    first_valid = sum(bool(a.get("first_pass_valid")) for a in audits)
    repaired = [a for a in audits if a.get("repair_count", 0) > 0]
    repaired_ok = sum(bool(a.get("accepted")) for a in repaired)
    accepted = sum(bool(a.get("accepted")) for a in audits)
    norm_total = sum(a.get("normalization_count", 0) or 0 for a in audits)
    norm_kinds: Counter = Counter()
    rejected_first: Counter = Counter()
    per_type = defaultdict(lambda: {"n": 0, "pt": [], "ct": [], "lat": [], "cached": 0})
    for a in audits:
        responses = a.get("responses") or []
        for r in responses:
            for item in r.get("normalizations") or []:
                norm_kinds[str(item.get("kind", item) if isinstance(item, dict) else item)[:60]] += 1
        if responses and responses[0].get("validation_error"):
            rejected_first[str(responses[0]["validation_error"])[:80]] += 1
        t = per_type[call_type(a)]
        t["n"] += 1
        for r in responses:
            resp = r.get("response") or {}
            if not resp:
                continue
            if resp.get("cached"):
                t["cached"] += 1
                continue
            t["pt"].append(resp.get("prompt_tokens", 0))
            t["ct"].append(resp.get("completion_tokens", 0))
            t["lat"].append(resp.get("latency_s", 0.0))
    lines = [
        f"- planner calls (audits): {n}",
        f"- first-pass-valid rate: {_rate(first_valid, n)}",
        f"- repair rate (calls that needed a repair): {_rate(len(repaired), n)}",
        f"- repair success rate (accepted after repair): {_rate(repaired_ok, len(repaired))}",
        f"- accepted overall: {_rate(accepted, n)}",
        f"- normalisations applied: {norm_total} in total",
        "",
        "**Validator ablation**: first responses the contract rejected before repair."
        " Without the validator these plans would have gone to the robot as they were:"
        f" **{sum(rejected_first.values())}/{n}**.",
        "",
        "| first-response rejection reason | count |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in rejected_first.most_common()] or ["| (none) | 0 |"]
    if norm_kinds:
        lines += ["", "| normalisation | count |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in norm_kinds.most_common()]
    lines += ["", "| call type | calls | live responses | cached | prompt tokens | completion tokens | latency s |",
              "|---|---|---|---|---|---|---|"]
    for name in ("initial plan", "held-object replan", "post-clarification", "other replan"):
        t = per_type.get(name)
        if not t:
            continue
        lines.append(f"| {name} | {t['n']} | {len(t['lat'])} | {t['cached']} | {_stats(t['pt'])} | "
                     f"{_stats(t['ct'])} | {_stats(t['lat'])} |")
    return "\n".join(lines) + "\n"


def status_of(audit: dict) -> str:
    plan = audit.get("compiled_plan") or {}
    return str(plan.get("status") or ("ERROR" if not audit.get("accepted") else "?"))


def conversions(audits: list[dict]) -> dict:
    """Counts of the ways B turned a raw model answer into its final status."""
    out = Counter()
    for a in audits:
        status = status_of(a)
        out["status:" + status] += 1
        if a.get("repair_count", 0) > 0:
            out["repaired -> accepted" if a.get("accepted") else "repaired -> not accepted"] += 1
        if status == "REJECTED":
            out["contract failure -> REJECTED (replan)"] += 1
        for r in a.get("responses") or []:
            for item in r.get("normalizations") or []:
                if isinstance(item, dict) and item.get("change") == "READY with SEARCH -> NEEDS_SEARCH":
                    out["READY with SEARCH -> NEEDS_SEARCH"] += 1
    return dict(out)


def _episode_outcomes(trials_root: pathlib.Path) -> dict:
    """plan reason text -> (trial id, correct) from e2e trial records."""
    index = {}
    for rec_path in trials_root.rglob("trial_record.json"):
        if "merged" in rec_path.parts:
            continue
        try:
            rec = json.loads(rec_path.read_text())
        except (OSError, ValueError):
            continue
        ok = bool(rec.get("claimed_success")) and rec.get("actual_success") is True
        for e in rec.get("events", []):
            if e.get("type") == "plan" and e.get("reason"):
                index[(rec.get("instruction"), e["reason"])] = (rec.get("trial_id"), ok)
    return index


def memory_usage(audits: list[dict], trials_root=None) -> dict:
    used = [a for a in audits if a.get("used_memory_for")]
    ages = [int(a.get("memory_age_frames") or 0) for a in used]
    result = {"calls": len(audits), "memory_calls": len(used), "ages": ages,
              "memory_statuses": dict(Counter(status_of(a) for a in used))}
    if trials_root is not None:
        index = _episode_outcomes(pathlib.Path(trials_root))
        by_trial = {}
        for a in audits:
            reason = (a.get("compiled_plan") or {}).get("reason")
            key = ((a.get("input") or {}).get("instruction"), reason)
            if key in index:
                tid, ok = index[key]
                by_trial.setdefault(tid, [ok, False])
                by_trial[tid][1] |= bool(a.get("used_memory_for"))
        mem = [ok for ok, m in by_trial.values() if m]
        fresh = [ok for ok, m in by_trial.values() if not m]
        result.update(trials_matched=len(by_trial), memory_trials=len(mem), memory_trials_ok=sum(mem),
                      fresh_trials=len(fresh), fresh_trials_ok=sum(fresh))
    return result


def memory_section(m: dict) -> str:
    lines = [f"- B calls that planned from memory: {_rate(m['memory_calls'], m['calls'])}",
             f"- memory age (frames): {_stats([float(a) for a in m['ages']])}"
             + (f", max {max(m['ages'])}" if m["ages"] else ""),
             "- status of memory plans: " + (", ".join(f"{k} {v}" for k, v in m["memory_statuses"].items()) or "none")]
    if "trials_matched" in m:
        lines += [f"- e2e trials matched to audits: {m['trials_matched']}",
                  f"- trials with ≥ 1 memory plan: claimed∧achieved {_rate(m['memory_trials_ok'], m['memory_trials'])}",
                  f"- trials without: claimed∧achieved {_rate(m['fresh_trials_ok'], m['fresh_trials'])}",
                  "- caveat: B only needs memory in episodes where the region left the view (SEARCH, "
                  "re-grasp, long carries), so these two rates are not a controlled comparison."]
    return "\n".join(lines) + "\n"


PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4, fixed order
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"


def _axes(title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c2")
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.grid(axis="x", color="#ecebe6", linewidth=0.6)
    ax.set_axisbelow(True)
    return fig, ax, plt


def figures(audits: list[dict], m: dict, out: pathlib.Path) -> list[pathlib.Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    conv = conversions(audits)
    statuses = sorted((k.split(":", 1)[1], v) for k, v in conv.items() if k.startswith("status:"))
    fig, ax, plt = _axes("B final plan status per call")
    labels, values = [s for s, _ in statuses], [v for _, v in statuses]
    ax.barh(labels, values, color=PALETTE[0], height=0.6)
    for y, v in enumerate(values):
        ax.text(v, y, f" {v}", va="center", fontsize=8, color=INK2)
    ax.set_xlabel("planner calls", color=INK2, fontsize=8)
    fig.tight_layout(); paths.append(out / "b_status.png"); fig.savefig(paths[-1]); plt.close(fig)

    per = defaultdict(list)
    for a in audits:
        for r in a.get("responses") or []:
            resp = r.get("response") or {}
            if resp and not resp.get("cached"):
                per[call_type(a)].append(float(resp.get("latency_s", 0.0)))
    names = [n for n in ("initial plan", "other replan", "held-object replan", "post-clarification") if per.get(n)]
    fig, ax, plt = _axes("B response latency by call type (live responses)")
    ax.boxplot([per[n] for n in names], orientation="horizontal", tick_labels=[f"{n} (n={len(per[n])})" for n in names], widths=0.5,
               medianprops={"color": PALETTE[1], "linewidth": 2}, boxprops={"color": PALETTE[0]},
               whiskerprops={"color": PALETTE[0]}, capprops={"color": PALETTE[0]},
               flierprops={"markeredgecolor": PALETTE[0], "markersize": 4})
    ax.set_xlabel("seconds", color=INK2, fontsize=8)
    fig.tight_layout(); paths.append(out / "b_latency.png"); fig.savefig(paths[-1]); plt.close(fig)

    fig, ax, plt = _axes(f"Age of positions B planned from memory (n={len(m['ages'])})")
    if m["ages"]:
        ax.hist(m["ages"], bins=range(0, max(m["ages"]) + 3, 2), color=PALETTE[2], rwidth=0.85)
    ax.set_xlabel("frames since the region was last LOCALIZED", color=INK2, fontsize=8)
    ax.set_ylabel("memory plans", color=INK2, fontsize=8)
    fig.tight_layout(); paths.append(out / "b_memory_age.png"); fig.savefig(paths[-1]); plt.close(fig)
    return paths


def variation_table(run_dir: pathlib.Path) -> str:
    """Per-category goal/status accuracy from an eval.b_benchmark output folder."""
    summary = json.loads((run_dir / "summary.json").read_text())
    per = defaultdict(lambda: [0, 0, 0])  # n, goal_ok, status_ok
    for case in summary.get("cases", []):
        c = per[case.get("category", "?")]
        c[0] += 1
        errors = [str(e) for e in case.get("errors") or []]
        planned = bool(case.get("stages"))  # no stages = planner raised, nothing to score
        goal_errors = [e for e in errors if e.startswith(("object_id", "region_id", "Ambiguous"))]
        c[1] += planned and not goal_errors
        c[2] += planned and not any(e.startswith("status") for e in errors)
    lines = ["| category | cases | goal accuracy | status accuracy |", "|---|---|---|---|"]
    for cat, (n, g, s) in sorted(per.items()):
        lines.append(f"| {cat} | {n} | {_rate(g, n)} | {_rate(s, n)} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Student B planner audit metrics")
    parser.add_argument("--runs", type=pathlib.Path, default=pathlib.Path("runs/night"))
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("docs/validation/B_METRICS.md"))
    parser.add_argument("--variation", type=pathlib.Path, default=None)
    parser.add_argument("--trials-root", type=pathlib.Path, default=None,
                        help="e2e run root whose trial records give the outcome of memory vs fresh episodes")
    parser.add_argument("--figs", type=pathlib.Path, default=None, help="write PNG figures here")
    args = parser.parse_args(argv)
    audits = load_audits(args.runs)
    m = memory_usage(audits, args.trials_root)
    text = [f"# Student B metrics\n\nSource: every B audit under `{args.runs}` (live or cached responses only).\n",
            summarise(audits), "\n## Conversions\n",
            "\n".join(f"- {k}: {v}" for k, v in sorted(conversions(audits).items())) + "\n",
            "\n## Memory (B EpisodeMemory)\n", memory_section(m)]
    if args.figs is not None:
        text += ["\n## Figures\n"] + [f"![{p.stem}]({os.path.relpath(p, args.out.parent)})\n"
                                         for p in figures(audits, m, args.figs)]
    if args.variation is not None:
        text += ["\n## Variation suite (`eval/trials/student_b_variation`)\n", variation_table(args.variation)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(text))
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
