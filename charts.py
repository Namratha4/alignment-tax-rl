"""Three charts for the writeup. Reads results/summary.csv, results/raw/*.json, results/training_curves.csv.
Writes charts/*.png. Run `python aggregate.py` first (it fetches and caches the eval results).

Encoding is the same in every chart: orange = naive (beta 0), blue = KL penalty, darker blue = larger beta.
Palette validated with the dataviz skill's validate_palette (categorical pair and ordinal ramp both pass)."""
import csv
import json
import math
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SURFACE, INK, INK2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e8e7e3", "#c9c8c3"
NAIVE = "#eb6834"
BETA = {0.01: "#86b6ef", 0.04: "#2a78d6", 0.2: "#104281"}  # ordinal blue ramp: bigger beta, darker

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": AXIS,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.axisbelow": True, "font.size": 10,
})


def titles(fig, title, subtitle):
    fig.text(0.06, 0.965, title, fontsize=15, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.06, 0.915, subtitle, fontsize=10, color=INK2, ha="left", va="top")


# ---------- chart 1: the share of completions that hit the length cap, over training ----------
def chart_clipped():
    curves = defaultdict(lambda: ([], []))
    for r in csv.DictReader(open("results/training_curves.csv")):
        curves[r["run"]][0].append(int(r["step"]))
        curves[r["run"]][1].append(float(r["clipped_ratio"]))

    panels = [
        ("LoRA, learning rate 2e-4", [
            ("kl_b0.04_lr2e-4_s300", "KL β = 0.04", BETA[0.04]),
            ("naive_b0_lr2e-4_s300", "Naive (β = 0)", NAIVE)]),
        ("Full-parameter, learning rate 1e-5", [
            ("full_b0.01_lr1e-5_s300", "KL β = 0.01", BETA[0.01]),
            ("full_b0.04_lr1e-5_s300", "KL β = 0.04", BETA[0.04]),
            ("full_b0.2_lr1e-5_s300", "KL β = 0.2", BETA[0.2]),
            ("full_b0_lr1e-5_s300", "Naive (β = 0)", NAIVE)]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.22, wspace=0.08)
    for ax, (ptitle, runs) in zip(axes, panels):
        for run, label, color in runs:
            x, y = curves[run]
            ax.plot(x, y, color=color, lw=2.2, label=label, solid_capstyle="round")
        ax.set_title(ptitle, fontsize=11, color=INK, loc="left", pad=8)
        ax.set_xlim(0, 300)
        ax.set_ylim(0, 1.1)
        ax.set_xlabel("Training step")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=len(runs), frameon=False, fontsize=9,
                  labelcolor=INK, handlelength=1.6, columnspacing=1.2)
    axes[0].set_ylabel("Share of completions hitting the 512-token cap")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))

    xs, ys = curves["naive_b0_lr2e-4_s300"]
    first = next(s for s, v in zip(xs, ys) if v > 0.5)
    axes[0].text(150, 1.045, f"naive: over 50% capped from step {first}, never recovers",
                 ha="center", fontsize=9, color=INK)
    xs, ys = curves["full_b0_lr1e-5_s300"]
    above = [s for s, v in zip(xs, ys) if v > 0.5]
    recovered = next(s for s, v in zip(xs, ys) if s > max(above) and v < 0.1)
    axes[1].text(298, 0.84, f"naive: over 50% capped from\nstep {min(above)} to {max(above)} (with a dip),\nnormal again by step {recovered}",
                 ha="right", va="center", fontsize=9, color=INK)
    titles(fig, "A KL penalty stopped the model from forgetting how to stop",
           "Without it, completions run to the length cap; with it (even β = 0.01), they almost always end normally. One run per setting.")
    fig.savefig("charts/1_clipped_ratio.png", dpi=200)
    plt.close(fig)


# ---------- chart 2: the KL knob ----------
def metric(stem):
    r = json.load(open(f"results/raw/{stem}.json"))["results"]
    g, i = r["gsm8k"], r["ifeval"]
    mm = [v for t, v in r.items() if t.startswith("mmlu_")]
    return {
        "GSM8K": (g["exact_match,flexible-extract"], g["exact_match_stderr,flexible-extract"]),
        "IFEval": (i["prompt_level_loose_acc,none"], i["prompt_level_loose_acc_stderr,none"]),
        "MMLU": (sum(v["acc,none"] for v in mm) / len(mm), math.sqrt(sum(v["acc_stderr,none"] ** 2 for v in mm)) / len(mm)),
    }


def chart_knob():
    base = metric("v2_baseline")
    stems = {0.0: "v2_full_b0_lr1e-5_s300_ckpt300", 0.01: "v2_full_b0.01_lr1e-5_s300_ckpt300",
             0.04: "v2_full_b0.04_lr1e-5_s300_ckpt300", 0.2: "v2_full_b0.2_lr1e-5_s300_ckpt300"}
    runs = {b: metric(s) for b, s in stems.items()}
    kl = defaultdict(list)
    for r in csv.DictReader(open("results/training_curves.csv")):
        if r["kl"] not in ("", "None", None) and r["run"].startswith("full_b0."):
            kl[r["run"]].append(float(r["kl"]))
    kl_by_beta = {b: sum(kl[f"full_b{b}_lr1e-5_s300"][-6:]) / 6 for b in (0.01, 0.04, 0.2)}

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.84, bottom=0.12, hspace=0.42, wspace=0.28)
    xpos = {0.0: 0, 0.01: 1, 0.04: 2, 0.2: 3}
    ticklabels = ["0\n(naive)", "0.01", "0.04", "0.2"]

    def delta(task, b):
        v, se = runs[b][task]
        bv, bse = base[task]
        return (v - bv) * 100, math.sqrt(se ** 2 + bse ** 2) * 100

    ax = axes[0, 0]
    xs = [xpos[b] for b in (0.01, 0.04, 0.2)]
    ys = [kl_by_beta[b] for b in (0.01, 0.04, 0.2)]
    ax.plot(xs, ys, color=BETA[0.04], lw=2, marker="o", ms=9, mec=SURFACE, mew=2)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(9, 6), fontsize=9, color=INK)
    ax.text(0, max(ys) * 0.5, "not logged\nat β = 0", ha="center", va="center", fontsize=9, color=INK2)
    ax.set_title("Measured distance from the base model (KL)", fontsize=10.5, loc="left", color=INK, pad=8)
    ax.set_ylim(0, max(ys) * 1.2)

    for ax, task, title in ((axes[0, 1], "GSM8K", "GSM8K change vs base (points)"),
                            (axes[1, 0], "IFEval", "IFEval change vs base (points)"),
                            (axes[1, 1], "MMLU", "MMLU change vs base (points)")):
        d = {b: delta(task, b) for b in runs}
        ax.axhline(0, color=INK2, lw=0.9)
        xs = [xpos[b] for b in (0.01, 0.04, 0.2)]
        ax.plot(xs, [d[b][0] for b in (0.01, 0.04, 0.2)], color=BETA[0.04], lw=2, zorder=3)
        for b in (0.01, 0.04, 0.2):
            ax.errorbar(xpos[b], d[b][0], yerr=d[b][1], color=BETA[0.04], capsize=3, lw=1.2, zorder=3)
            ax.plot(xpos[b], d[b][0], "o", color=BETA[0.04], ms=9, mec=SURFACE, mew=2, zorder=4)
        ax.errorbar(0, d[0.0][0], yerr=d[0.0][1], color=NAIVE, capsize=3, lw=1.2, zorder=3)
        ax.plot(0, d[0.0][0], "o", color=NAIVE, ms=9, mec=SURFACE, mew=2, zorder=4)
        ax.set_title(title, fontsize=10.5, loc="left", color=INK, pad=8)
        if task == "MMLU":
            ax.annotate("picks answer “A” on\n55–62% of questions", (1, d[0.01][0]), textcoords="offset points",
                        xytext=(16, 4), fontsize=9, color=INK, va="center")
        lo = min(d[b][0] - d[b][1] for b in d)
        hi = max(d[b][0] + d[b][1] for b in d)
        pad = (hi - lo) * 0.08
        ax.set_ylim(lo - pad, hi + pad)

    for ax in axes.flat:
        ax.set_xticks(range(4))
        ax.set_xticklabels(ticklabels)
        ax.set_xlim(-0.5, 3.5)
        ax.set_xlabel("KL coefficient β")
    handles = [Line2D([], [], marker="o", ls="none", color=NAIVE, ms=9, mec=SURFACE, mew=2, label="Naive (β = 0)"),
               Line2D([], [], marker="o", ls="-", color=BETA[0.04], ms=9, mec=SURFACE, mew=2, lw=2, label="KL penalty")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=2, frameon=False, fontsize=9.5,
               labelcolor=INK)
    titles(fig, "Turning the KL knob (β): less drift, a mixed capability picture",
           "Full-parameter training, learning rate 1e-5, step 300. Bars: ±1 SE of the difference (unpaired, so conservative). One run per setting.")
    fig.savefig("charts/2_kl_knob.png", dpi=200)
    plt.close(fig)


# ---------- chart 3: trade-off scatter ----------
def chart_tradeoff():
    rows = {r["run"]: r for r in csv.DictReader(open("results/summary.csv"))}
    pts = [  # key in summary.csv, display name, method, family, label offset (points) and alignment
        ("LoRA naive 5e-5", "LoRA naive, lr 5e-5", "naive", "lora", (9, 7, "left")),
        ("LoRA naive 2e-4", "LoRA naive, lr 2e-4", "naive", "lora", (9, 5, "left")),
        ("LoRA KL.04 2e-4", "LoRA KL β=0.04, lr 2e-4", "kl", "lora", (9, -13, "left")),
        ("full naive @300", "Full naive, lr 1e-5", "naive", "full", (11, -3, "left")),
        ("full KL.01 @300", "Full KL β=0.01 (MMLU answer-letter bias)", "kl", "full", (11, -3, "left")),
        ("full KL.04 @300", "Full KL β=0.04", "kl", "full", (-9, -15, "right")),
        ("full KL.2 @300", "Full KL β=0.2", "kl", "full", (10, -13, "left")),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.85, bottom=0.2)
    ax.grid(True, axis="both")
    ax.axhline(100, color=INK2, lw=0.9)
    ax.axvline(0, color=INK2, lw=0.9)
    for key, name, method, family, (dx, dy, ha) in pts:
        x = float(rows[key]["gsm8k_delta"]) * 100
        y = float(rows[key]["retention"]) * 100
        ax.plot(x, y, marker="o" if family == "lora" else "D", ms=11 if family == "lora" else 10,
                color=NAIVE if method == "naive" else BETA[0.04], mec=SURFACE, mew=2, ls="none", zorder=4)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=9.5, color=INK)
    ax.set_xlim(-13, 10)
    ax.set_ylim(72, 112)
    ax.set_xlabel("GSM8K change vs base (points; 5-shot eval, n = 200)")
    ax.set_ylabel("Retention vs base (%)\n(MMLU, IFEval, HumanEval averaged)")
    ax.text(9.7, 110.5, "better ↗", ha="right", va="top", fontsize=10, color=INK2)
    handles = [
        Line2D([], [], marker="o", ls="none", color=NAIVE, ms=10, mec=SURFACE, mew=2, label="Naive (β = 0)"),
        Line2D([], [], marker="o", ls="none", color=BETA[0.04], ms=10, mec=SURFACE, mew=2, label="KL penalty"),
        Line2D([], [], marker="o", ls="none", color=INK2, ms=10, mec=SURFACE, mew=2, label="LoRA"),
        Line2D([], [], marker="D", ls="none", color=INK2, ms=9, mec=SURFACE, mew=2, label="Full-parameter")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4, frameon=False, fontsize=9.5,
              labelcolor=INK)
    titles(fig, "GSM8K change vs. retention of other skills, by run",
           "Final checkpoint of each run. Each score has about ±3 to 4 points of sampling error; one training run per setting.")
    fig.savefig("charts/3_tradeoff.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    chart_clipped()
    chart_knob()
    chart_tradeoff()
    print("wrote charts/1_clipped_ratio.png, 2_kl_knob.png, 3_tradeoff.png")
