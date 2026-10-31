"""Draw the hypothesis charts in EXPERIMENT_LOG.md: python results/plot_hypotheses.py (from the repo root).

One PNG per group of hypotheses, one panel per hypothesis, one row per comparison: tested run against its baseline, hold-out top-1.
Scores at 224 px are checked against results/results.csv; other test sizes come from the run sections of the log.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = "results/hypothesis_charts"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
TESTED, BASE = "#2a78d6", "#8a8983"
NOISE = 0.5

R = {"lora_r8_full": 68.78, "abl_lora_only": 70.76, "abl_adaptformer_only": 71.09, "abl_combined": 71.05,
     "abl_adaptformer_last6": 70.17, "imp_trivialaug": 74.28, "tune_label_smoothing": 72.17, "full_ft_ls": 73.98,
     "full_ft_baseline": 72.53, "tune_lr_low": 72.37, "tune_lr_high": 69.28, "tune_ls_lr2e4": 72.35, "tune_ls_lr1e4": 73.02,
     "tune_ls_lr5e5": 73.52, "tune_head_init": 69.98, "tune_ls_lr1e4_init43": 71.99, "imp_head_from_lp": 70.28,
     "tune_head_lr_half": 74.14, "tune_head_lr_quarter": 74.02, "tune_head_lr_double": 71.75, "abl_linear_probe": 62.62,
     "imp_droppath": 74.09, "imp_ema": 74.19, "imp_mixup": 73.97, "imp_trivialaug_fullbudget": 74.18, "full_ft_llrd": 76.26,
     "full_ft_llrd_10ep": 76.06, "full_ft_ls_headlow": 75.75, "full_ft_llrd_288": 79.69}
for row in csv.DictReader(open("results/results.csv", encoding="utf8")):
    if row["name"] in R:
        assert float(row["holdout_top1"]) == R[row["name"]], row["name"]
AT_288 = {"imp_trivialaug": 75.99, "imp_trivialaug_fullbudget": 75.99, "full_ft_llrd": 77.89, "full_ft_llrd_10ep": 77.67,
          "full_ft_ls_headlow": 77.43, "imp_trivialaug_llrd": 74.73}


def vs(run, base):
    return (run, R[run], base, R[base])


# file name: (heading, [(hypothesis, verdict, [(tested label, tested score, baseline label, baseline score)])])
GROUPS = {
    "h1_h4_design": ("Where should the 5% go?", [
        ("H1: more adapter parameters beat the small LoRA", "Supported", [vs(r, "lora_r8_full") for r in ("abl_lora_only", "abl_adaptformer_only", "abl_combined")]),
        ("H2: adapting attention and FFN beats either alone", "Rejected", [vs("abl_combined", "abl_lora_only"), vs("abl_combined", "abl_adaptformer_only")]),
        ("H3: AdaptFormer alone can match the combined design, tuned", "Inconclusive", [("imp_adaptformer_only", 73.78, "imp_trivialaug", R["imp_trivialaug"])]),
        ("H4: the 5% is best spent on the last 6 blocks", "Rejected", [vs("abl_adaptformer_last6", "abl_adaptformer_only")]),
    ]),
    "h5_h8_overconfidence": ("Is over-confidence the limit?", [
        ("H5: label smoothing helps", "Supported", [vs("tune_label_smoothing", "abl_combined"), vs("full_ft_ls", "full_ft_baseline")]),
        ("H6: a lower learning rate helps, a higher one hurts", "Supported", [vs("tune_lr_low", "abl_combined"), vs("tune_lr_high", "abl_combined")]),
        ("H7: label smoothing and a lower rate add up", "Rejected", [vs("tune_ls_lr2e4", "tune_label_smoothing"), vs("tune_ls_lr2e4", "tune_lr_low")]),
        ("H8: with label smoothing, a still lower rate keeps helping", "Supported down to 1e-4", [vs("tune_ls_lr1e4", "tune_ls_lr2e4"), vs("tune_ls_lr5e5", "tune_ls_lr1e4")]),
    ]),
    "h9_h11_head": ("What does the head do?", [
        ("H9: a better head start gives a better final model", "Rejected", [vs("tune_head_init", "abl_combined"), vs("tune_ls_lr1e4_init43", "tune_ls_lr1e4"), vs("imp_head_from_lp", "imp_trivialaug")]),
        ("H10: the head should learn more slowly than the adapters", "Supported", [vs(r, "tune_ls_lr1e4") for r in ("tune_head_lr_half", "tune_head_lr_quarter", "tune_head_lr_double")]),
        ("H11: training the head gives most of the gain", "Supported", [("abl_linear_probe (head only)", R["abl_linear_probe"], "zero-shot", 35.70), ("tune_ls_lr1e4 (head + adapters)", R["tune_ls_lr1e4"], "abl_linear_probe", R["abl_linear_probe"])]),
    ]),
    "h12_h14_recipe": ("Does more regularisation or training time help?", [
        ("H12: a stronger regulariser adds accuracy", "Rejected", [vs(r, "tune_head_lr_half") for r in ("imp_droppath", "imp_ema", "imp_mixup", "imp_trivialaug")]),
        ("H13: layer-wise lr decay lifts the 5% design", "Rejected", [("imp_trivialaug_llrd", 72.78, "imp_trivialaug", R["imp_trivialaug"]), ("imp_trivialaug_llrd at 288 px", AT_288["imp_trivialaug_llrd"], "imp_trivialaug at 288 px", AT_288["imp_trivialaug"])]),
        ("H14: the whole budget improves on 10 epochs", "Rejected", [vs("imp_trivialaug_fullbudget", "imp_trivialaug"), vs("full_ft_llrd", "full_ft_llrd_10ep")]),
    ]),
    "h15_h16_tracks": ("Full fine-tuning and the track comparison", [
        ("H15: layer-wise lr decay helps full fine-tuning", "Supported, modestly", [vs("full_ft_llrd", "full_ft_ls_headlow")]),
        ("H16: tuned, the 5% design matches full fine-tuning", "Rejected", [
            ("5%, 10 epochs, 224 px", R["imp_trivialaug"], "full, 10 epochs, 224 px", R["full_ft_llrd_10ep"]),
            ("5%, 10 epochs, 288 px", AT_288["imp_trivialaug"], "full, 10 epochs, 288 px", AT_288["full_ft_llrd_10ep"]),
            ("5%, whole budget, 224 px", R["imp_trivialaug_fullbudget"], "full, whole budget, 224 px", R["full_ft_llrd"]),
            ("5%, whole budget, 288 px", AT_288["imp_trivialaug_fullbudget"], "full, whole budget, 288 px", AT_288["full_ft_llrd"])]),
    ]),
    "h17_h19_pixels": ("Do more pixels or more models help?", [
        ("H17: testing above the training size helps", "Supported", [(f"{r} at 288 px", AT_288[r], "the same model at 224 px", R[r]) for r in ("imp_trivialaug", "full_ft_llrd", "full_ft_llrd_10ep", "full_ft_ls_headlow")]),
        ("H18: averaging two models clears 1 point", "Rejected", [("ensemble at 288 px", 78.30, "full_ft_llrd at 288 px", 77.91)]),
        ("H19: training at a higher resolution beats only testing at it", "Supported", [("full_ft_llrd_288, tested at 288 px", R["full_ft_llrd_288"], "full_ft_llrd_10ep at 288 px", AT_288["full_ft_llrd_10ep"]), ("full_ft_llrd_288, tested at 352 px", 80.45, "full_ft_llrd_10ep at 288 px", AT_288["full_ft_llrd_10ep"])]),
    ]),
}


def delta_label(d):
    d = round(d, 2)
    tag = "  noise" if abs(d) < NOISE else "  at noise bar" if abs(d) == NOISE else ""
    return f"{d:+.2f}{tag}"


def panel(ax, title, verdict, rows):
    lo, hi = min(min(r[1], r[3]) for r in rows), max(max(r[1], r[3]) for r in rows)
    pad = max((4 - (hi - lo)) / 2, 0.08 * (hi - lo))
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    for y, (run, score, base, base_score) in enumerate(rows):
        ax.plot([base_score, score], [y, y], color=TESTED, lw=2, alpha=0.35, zorder=1, solid_capstyle="butt")
        ax.scatter(base_score, y, s=64, facecolor=SURFACE, edgecolor=BASE, linewidth=2, zorder=2)
        ax.scatter(score, y, s=70, color=TESTED, edgecolor=SURFACE, linewidth=1, zorder=3)
        ax.text(-0.015, y - 0.16, run, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=9, color=INK)
        ax.text(-0.015, y + 0.2, "vs " + base, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8, color=MUTED)
        ax.text(1.015, y, delta_label(score - base_score), transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=9, color=INK)
    ax.set_yticks([])
    ax.tick_params(axis="x", colors=MUTED, labelsize=8, length=0)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_facecolor(SURFACE)
    ax.set_title(f"{title}  ({verdict})", loc="left", x=-0.527, fontsize=10.5, color=INK, fontweight="bold", pad=8)


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, (heading, panels) in GROUPS.items():
        heights = [len(p[2]) * 0.62 + 0.3 for p in panels]
        fig, axes = plt.subplots(len(panels), 1, figsize=(10, sum(heights) + 0.75 * len(panels) + 1.0), gridspec_kw={"height_ratios": heights}, facecolor=SURFACE)
        h = fig.get_figheight()
        for ax, (title, verdict, rows) in zip(axes, panels):
            panel(ax, title, verdict, rows)
        fig.subplots_adjust(left=0.31, right=0.86, top=1 - 0.85 / h, bottom=0.6 / h, hspace=0.75)
        fig.suptitle(heading, x=0.02, ha="left", fontsize=13, color=INK, fontweight="bold", y=1 - 0.2 / h)
        fig.legend(handles=[Line2D([], [], marker="o", ls="", ms=8, color=TESTED, label="Tested run"),
                            Line2D([], [], marker="o", ls="", ms=8, markerfacecolor=SURFACE, markeredgecolor=BASE, markeredgewidth=2, label="Baseline")],
                   loc="upper right", ncol=2, frameon=False, fontsize=9, labelcolor=MUTED, bbox_to_anchor=(0.99, 1 - 0.08 / h))
        fig.text(0.02, 0.15 / h, "Hold-out top-1 (%). Each panel has its own axis range. Right column: tested minus baseline; under 0.5 points is noise.", fontsize=8, color=MUTED)
        fig.savefig(f"{OUT}/{name}.png", dpi=150, facecolor=SURFACE)
        plt.close(fig)


if __name__ == "__main__":
    main()
