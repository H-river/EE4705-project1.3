# Owner: Student B
"""E2: compare Qwen tiers from eval.b_benchmark output folders.

    python -m eval.b_model_compare runs/final/e2 --out docs/validation/B_MODEL_COMPARE.md \
        [--figs docs/night_run/figs]

Expects <root>/<model>_eval and <root>/<model>_var benchmark folders (see
runs/final/e2/run.sh). Cost is reported as live tokens; the repo has no
price list, so multiply by the console's per-token prices for money.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics


def load(root: pathlib.Path) -> list[dict]:
    rows = []
    for d in sorted(p for p in root.iterdir() if (p / "summary.json").exists()):
        model, suite = d.name.rsplit("_", 1)
        s = json.loads((d / "summary.json").read_text())
        results = [json.loads(p.read_text()) for p in d.glob("*/result.json")]
        stats = [r.get("api_stats", {}) for r in results]
        rows.append({
            "model": model, "suite": suite, "n": s["n_cases"], "correct": s["n_correct"],
            "accuracy": s["accuracy"], "first_response_accuracy": s["first_response_accuracy"],
            "latency_mean_s": statistics.mean(r["latency_s"] for r in results),
            "latency_median_s": statistics.median(r["latency_s"] for r in results),
            "calls": sum(x.get("attempts", 0) for x in stats),
            "prompt_tokens": sum(x.get("prompt_tokens", 0) for x in stats),
            "completion_tokens": sum(x.get("completion_tokens", 0) for x in stats),
            "errors": [e for r in results for e in r.get("errors", [])],
        })
    tier = {"flash": 0, "plus": 1, "max": 2}
    rows.sort(key=lambda r: (next((v for k, v in tier.items() if f"-{k}" in r["model"]), 9), r["model"], r["suite"]))
    return rows


def table(rows: list[dict]) -> str:
    lines = ["| model | suite | correct | first-response | latency mean / median (s) | calls | prompt tok | completion tok |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['suite']} | {r['correct']}/{r['n']} ({r['accuracy']:.1%}) | "
                     f"{r['first_response_accuracy']:.1%} | {r['latency_mean_s']:.1f} / {r['latency_median_s']:.1f} | "
                     f"{r['calls']} | {r['prompt_tokens']:,} | {r['completion_tokens']:,} |")
    return "\n".join(lines) + "\n"


def figure(rows: list[dict], path: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from eval.b_metrics import INK, INK2, PALETTE, SURFACE
    models = list(dict.fromkeys(r["model"] for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for ax, key, title, fmt in ((axes[0], "accuracy", "Accuracy (32-case + 20-case suites)", "{:.0%}"),
                                (axes[1], "latency_mean_s", "Mean latency per case (s)", "{:.1f}")):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c9c8c2")
        ax.tick_params(colors=INK2, labelsize=7)
        ax.set_title(title, color=INK, fontsize=9, loc="left")
        width = 0.36
        for k, suite in enumerate(("eval", "var")):
            vals = [next((r[key] for r in rows if r["model"] == m and r["suite"] == suite), 0) for m in models]
            xs = [i + (k - 0.5) * (width + 0.02) for i in range(len(models))]
            ax.bar(xs, vals, width=width, color=PALETTE[k], label={"eval": "32-case", "var": "20-case variation"}[suite])
            for x, v in zip(xs, vals):
                ax.text(x, v, fmt.format(v), ha="center", va="bottom", fontsize=6.5, color=INK2)
        ax.set_xticks(range(len(models)), [m.replace("qwen3.7-", "").rsplit("-", 3)[0] for m in models])
        ax.grid(axis="y", color="#ecebe6", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylim(0, 1.25)
    axes[0].legend(fontsize=7, frameon=False, loc="upper left", ncol=2)
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
    text = ["# Student B: Qwen tier comparison (E2)\n",
            f"Source: `{a.root}` (`eval.b_benchmark`, cache off, B thinking off, 2 workers). "
            "Cost = live tokens; the repo has no price list.\n", table(rows)]
    fails = {f"{r['model']} {r['suite']}": r["errors"] for r in rows if r["errors"]}
    if fails:
        text.append("\n## Failed checks\n")
        text += [f"- **{k}**: " + "; ".join(e[:120] for e in v) for k, v in fails.items()]
    if a.figs:
        a.figs.mkdir(parents=True, exist_ok=True)
        figure(rows, a.figs / "b_model_compare.png")
        text.append(f"\n![b_model_compare]({os.path.relpath(a.figs / 'b_model_compare.png', a.out.parent)})\n")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(text) + "\n")
    (a.out.with_suffix(".json")).write_text(json.dumps(rows, indent=2))
    print(table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
