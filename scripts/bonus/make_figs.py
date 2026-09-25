#!/usr/bin/env python
"""Bonus figures: VAL success vs training step (one line per run) -> docs/bonus/figs/.

    python scripts/bonus/make_figs.py --runs act_base diffusion_base --out curves_main.png \
        [--scripted-val 20/20] [--title ...]
Palette: validated categorical slots 1-4 (dataviz reference palette); scripted = gray dashed reference.
"""

from __future__ import annotations

import argparse
import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load(name: str):
    with (ROOT / "runs/bonus/curves" / f"{name}.csv").open() as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["step"]))
    return [int(r["step"]) for r in rows], [100 * float(r["rate"]) for r in rows]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Closed-loop VAL grasp success vs training step (20 episodes)")
    ap.add_argument("--scripted-val", type=float, default=None, help="scripted success rate in %% on VAL")
    args = ap.parse_args(argv)
    labels = args.labels or args.runs
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    for i, (run, lab) in enumerate(zip(args.runs, labels)):
        x, y = load(run)
        c = SERIES[i % len(SERIES)]
        ax.plot(x, y, color=c, lw=2, marker="o", ms=5, mec=SURF, mew=1.5, label=lab, zorder=3)
        best = max(range(len(y)), key=lambda k: (y[k], x[k]))  # tie -> latest step (selection rule)
        ax.annotate(f"{lab}: selected {x[best] // 1000}k ({y[best]:.0f} %)", (x[-1], y[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK2)
    if args.scripted_val is not None:
        ax.axhline(args.scripted_val, color=INK2, lw=1.2, ls="--", zorder=2)
        ax.text(ax.get_xlim()[0], args.scripted_val + 1.5, f"scripted {args.scripted_val:.0f} %", fontsize=8,
                color=INK2)
    ax.set_ylim(0, 105)
    ax.set_xlabel("training step", color=INK2, fontsize=9)
    ax.set_ylabel("grasp success (%)", color=INK2, fontsize=9)
    ax.set_title(args.title, color=INK, fontsize=10, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    if len(args.runs) >= 2:
        ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK2)
    fig.tight_layout()
    out = ROOT / "docs/bonus/figs" / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURF)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
