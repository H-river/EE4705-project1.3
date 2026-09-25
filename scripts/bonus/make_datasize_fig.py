#!/usr/bin/env python
"""8.1 figure: ACT success vs number of demos (500, 1k, 2k, 5k) -> docs/bonus/figs/datasize.png (+ csv)."""
import json, pathlib, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/bonus"))
import manip_metrics
from make_figs import SERIES, INK, INK2, GRID, SURF
RUNS = [(500, "act_500"), (1000, "act_1k"), (2000, "act"), (5000, "act_5k")]


def skill(label, cell):
    return json.loads((ROOT / "runs/bonus/eval" / f"{label}_{cell}.jsonl").read_text().splitlines()[0])["summary"]


rows = []
for n, label in RUNS:
    m = manip_metrics.run_metrics(ROOT / "runs/bonus/manip" / f"{label}_C1")
    rows.append((n, 100 * m["grasp_ok"] / m["n"], 100 * skill(label, "C1")["rate"], 100 * skill(label, "C2")["rate"]))
(ROOT / "docs/bonus/data/datasize.csv").write_text(
    "demos,full_C1_pct,skill_C1_pct,skill_C2_pct\n" + "".join(f"{r[0]},{r[1]:.1f},{r[2]:.1f},{r[3]:.1f}\n" for r in rows))
fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
xs = [r[0] for r in rows]
for k, (name, c) in enumerate([("full executor C1", SERIES[0]), ("skill level C1", SERIES[1]), ("skill level C2 (±20 cm)", SERIES[2])]):
    ys = [r[k + 1] for r in rows]
    ax.plot(xs, ys, color=c, lw=2, marker="o", ms=6, mec=SURF, mew=1.5, label=name, zorder=3)
    ax.annotate(name, (xs[-1], ys[-1]), xytext=(8, (k - 1) * 6), textcoords="offset points", va="center",
                fontsize=8, color=INK2, annotation_clip=False)
ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels(["500", "1k", "2k", "5k"])
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_ylim(70, 103); ax.grid(axis="y", color=GRID, lw=0.8)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=8)
ax.set_xlabel("training demonstrations (log scale)", color=INK2, fontsize=9)
ax.set_ylabel("grasp success (%)", color=INK2, fontsize=9)
ax.set_title("ACT: grasp success vs number of demos (30 episodes per point)", color=INK, fontsize=10, loc="left")
ax.legend(frameon=False, fontsize=8, loc="lower left", labelcolor=INK2)
fig.tight_layout(); fig.savefig(ROOT / "docs/bonus/figs/datasize.png", facecolor=SURF)
print(rows)
