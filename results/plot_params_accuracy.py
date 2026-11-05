"""Draw the parameters-against-accuracy chart (slide 7): python results/plot_params_accuracy.py (from the repo root).

Our tuned runs (hold-out top-1, 224 px, 10 epochs) against the brief's reference results. Scores are checked against results/results.csv.
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = "results/hypothesis_charts/params_vs_accuracy.png"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
OURS, REF = "#2a78d6", "#8a8983"
ZERO_X = 0.03   # where 0% is drawn on the log axis

# (run, label, trainable % of the backbone, top-1)
OURS_PTS = [("abl_linear_probe", "Linear probe\n(head only)", 0, 62.62),
            ("imp_bitfit", "BitFit\n(biases)", 0.12, 71.30),
            ("imp_trivialaug", "LoRA + AdaptFormer", 4.998, 74.28),
            ("full_ft_llrd_10ep", "Full fine-tuning", 100, 76.06)]
REF_PTS = [("LoRA r=8", 0.34, 68.3), ("Full fine-tuning", 100, 70.8)]

rows = {r["name"]: r for r in csv.DictReader(open("results/results.csv", encoding="utf8"))}
for run, _, pct, top1 in OURS_PTS:
    assert float(rows[run]["holdout_top1"]) == top1, run


def x(pct):
    return pct if pct > 0 else ZERO_X


fig, ax = plt.subplots(figsize=(10, 5.6), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.set_xscale("log")
ax.set_xlim(0.018, 250)
ax.set_ylim(60, 79)

xs = [x(p[2]) for p in OURS_PTS]
ys = [p[3] for p in OURS_PTS]
ax.plot(xs, ys, color=OURS, lw=2, zorder=2)
ax.scatter(xs, ys, s=90, color=OURS, edgecolor=SURFACE, linewidth=2, zorder=3)
offsets = {"abl_linear_probe": (14, 0),"imp_bitfit": (0, 12), "imp_trivialaug": (0, 12), "full_ft_llrd_10ep": (0, 12)}
for run, label, pct, top1 in OURS_PTS:
    dx, dy = offsets[run]
    ax.annotate(f"{label}\n{top1:.2f}", (x(pct), top1), xytext=(dx, dy), textcoords="offset points",
                ha="center" if dx == 0 else "left", va="bottom" if dy > 0 else "center", fontsize=9.5, color=INK, linespacing=1.15)

for label, pct, top1 in REF_PTS:
    dy = 12 if pct < 1 else -12
    ax.scatter(pct, top1, s=80, facecolor=SURFACE, edgecolor=REF, linewidth=2, zorder=3)
    ax.annotate(f"{label}\n{top1:.1f}", (pct, top1), xytext=(0, dy), textcoords="offset points",
                ha="center", va="bottom" if dy > 0 else "top", fontsize=9, color=MUTED, linespacing=1.15)

# the gains the slide talks about
ax.annotate("", xy=(0.12, 71.30), xytext=(0.12, 62.62), arrowprops=dict(arrowstyle="-", color=MUTED, lw=1, ls=(0, (2, 2))))
ax.text(0.135, 66.6, "+8.68\n74% of the adapters'\ngain over the probe", fontsize=8.5, color=MUTED, va="center", linespacing=1.15)
ax.text(0.9, 73.3, "+2.98 for 40x the parameters", fontsize=8.5, color=MUTED, ha="center", rotation=0)
ax.axvline(5, color=GRID, lw=1, ls=(0, (4, 3)), zorder=0)
ax.text(5.4, 60.4, "5% limit", fontsize=8, color=MUTED, va="bottom")

ax.set_xticks([ZERO_X, 0.1, 1, 5, 10, 100])
ax.set_xticklabels(["0", "0.1", "1", "5", "10", "100"])
ax.minorticks_off()
ax.set_xlabel("Trainable parameters, % of the backbone (log scale; head not counted)", fontsize=9, color=MUTED)
ax.set_ylabel("Hold-out top-1 (%)", fontsize=9, color=MUTED)
ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
ax.grid(color=GRID, lw=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(GRID)

fig.suptitle("Most of the adaptation is cheap", x=0.02, ha="left", fontsize=13, color=INK, fontweight="bold", y=0.97)
fig.text(0.02, 0.905, "Same recipe, 10 epochs, 224 px. The biases alone recover three quarters of what 5% of adapters add.", fontsize=9.5, color=MUTED)
fig.legend(handles=[Line2D([], [], marker="o", color=OURS, lw=2, ms=8, label="Ours (hold-out)"),
                    Line2D([], [], marker="o", ls="", ms=8, markerfacecolor=SURFACE, markeredgecolor=REF, markeredgewidth=2, label="Brief's reference")],
           loc="upper right", ncol=2, frameon=False, fontsize=9, labelcolor=MUTED, bbox_to_anchor=(0.98, 0.985))
fig.text(0.02, 0.015, "Full fine-tuning: full_ft_llrd_10ep (layer-wise lr decay, no TrivialAugment). The brief's results are on the val set, ours on our hold-out: compare shapes, not exact values.",
         fontsize=7.5, color=MUTED)
fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.14)
fig.savefig(OUT, dpi=150, facecolor=SURFACE)
