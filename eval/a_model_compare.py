# Owner: Student A
"""Compare Qwen-VL tiers from eval.grounding output folders.

    python -m eval.a_model_compare runs/a_tiers --out docs/validation/A_MODEL_COMPARE.md \
        [--figs docs/night_run/figs]

Expects one folder per (model, run) under ``root``, each an ordinary
``eval.grounding run`` output directory (contains ``summary.json`` and the
per-case ``a_*.json`` files) named ``<model>_<run>`` -- e.g.
``qwen3-vl-plus_eval``, ``qwen3-vl-max_eval``, or ``qwen3-vl-plus_run2`` for a
repeated capture with the same model. Set ``EE4705_VLM_MODEL`` to the tier
under test before each ``eval.grounding run`` and name the output folder to
match, the same way ``eval.b_model_compare`` expects ``<model>_eval`` /
``<model>_var`` from ``eval.b_benchmark``.

Cost is reported as live tokens; the repo has no price list, so multiply by
the console's per-token prices for money.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib


def load(root: pathlib.Path) -> list[dict]:
    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and (p / "summary.json").exists()):
        model, _, run = d.name.rpartition("_")
        model = model or d.name
        run = run or "eval"
        s = json.loads((d / "summary.json").read_text())
        cases = [json.loads(p.read_text()) for p in sorted(d.glob("a_*.json"))
                  if p.name not in ("dataset_snapshot.json",)]
        cases = [c for c in cases if "expected_kind" in c]
        stats = s.get("client_stats", {}) or {}
        live = stats.get("live_requests", 0)
        rows.append({
            "model": model, "run": run, "n": s.get("cases", len(cases)), "correct": s.get("correct", 0),
            "accuracy": s.get("target_selection_accuracy", 0.0),
            "mean_position_error_m": s.get("mean_position_error_m"),
            "error_cases": s.get("error_cases", 0),
            "latency_mean_s": (stats.get("total_latency_s", 0.0) / live) if live else 0.0,
            "calls": stats.get("attempts", 0), "live_requests": live, "cache_hits": stats.get("cache_hits", 0),
            "prompt_tokens": stats.get("prompt_tokens", 0), "completion_tokens": stats.get("completion_tokens", 0),
            "errors": [c["error"] for c in cases if c.get("error")],
        })
    tier = {"flash": 0, "plus": 1, "max": 2}
    rows.sort(key=lambda r: (next((v for k, v in tier.items() if f"-{k}" in r["model"]), 9), r["model"], r["run"]))
    return rows


def table(rows: list[dict]) -> str:
    lines = ["| model | run | correct | accuracy | mean pos. error (m) | latency mean (s) | calls | prompt tok | completion tok |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        err = f"{r['mean_position_error_m']:.3f}" if r["mean_position_error_m"] is not None else "n/a"
        lines.append(f"| {r['model']} | {r['run']} | {r['correct']}/{r['n']} ({r['accuracy']:.1%}) | "
                     f"{r['accuracy']:.1%} | {err} | {r['latency_mean_s']:.1f} | "
                     f"{r['calls']} | {r['prompt_tokens']:,} | {r['completion_tokens']:,} |")
    return "\n".join(lines) + "\n"


def figure(rows: list[dict], path: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from eval.b_metrics import INK, INK2, PALETTE, SURFACE
    models = list(dict.fromkeys(r["model"] for r in rows))
    runs = list(dict.fromkeys(r["run"] for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for ax, key, title, fmt in ((axes[0], "accuracy", "Target-selection accuracy (eval.grounding)", "{:.0%}"),
                                (axes[1], "latency_mean_s", "Mean latency per live call (s)", "{:.1f}")):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c9c8c2")
        ax.tick_params(colors=INK2, labelsize=7)
        ax.set_title(title, color=INK, fontsize=9, loc="left")
        width = 0.7 / max(len(runs), 1)
        for k, run in enumerate(runs):
            vals = [next((r[key] for r in rows if r["model"] == m and r["run"] == run), 0) for m in models]
            xs = [i + (k - (len(runs) - 1) / 2) * (width + 0.02) for i in range(len(models))]
            ax.bar(xs, vals, width=width, color=PALETTE[k % len(PALETTE)], label=run)
            for x, v in zip(xs, vals):
                ax.text(x, v, fmt.format(v), ha="center", va="bottom", fontsize=6.5, color=INK2)
        ax.set_xticks(range(len(models)), [m.replace("qwen3-vl-", "").replace("qwen3.7-", "") for m in models])
        ax.grid(axis="y", color="#ecebe6", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylim(0, 1.25)
    if len(runs) > 1:
        axes[0].legend(fontsize=7, frameon=False, loc="upper left", ncol=len(runs))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--figs", type=pathlib.Path, default=None)
    a = ap.parse_args(argv)
    rows = load(a.root)
    text = ["# Student A: Qwen-VL tier comparison\n",
            f"Source: `{a.root}` (`eval.grounding run`, one folder per model/run). "
            "Cost = live tokens; the repo has no price list.\n", table(rows)]
    fails = {f"{r['model']} {r['run']}": r["errors"] for r in rows if r["errors"]}
    if fails:
        text.append("\n## Cases with an exception or non-live response\n")
        text += [f"- **{k}**: " + "; ".join(e[:120] for e in v) for k, v in fails.items()]
    if a.figs:
        a.figs.mkdir(parents=True, exist_ok=True)
        figure(rows, a.figs / "a_model_compare.png")
        text.append(f"\n![a_model_compare]({os.path.relpath(a.figs / 'a_model_compare.png', a.out.parent)})\n")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(text) + "\n")
    (a.out.with_suffix(".json")).write_text(json.dumps(rows, indent=2))
    print(table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
