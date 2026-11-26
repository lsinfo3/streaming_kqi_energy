"""Fig. 4: live single-stream monitoring (W2), energy with inference minus the reference without
inference, per candidate and view, with Welch 95 % intervals over the 30 runs of each series.

Source: data/w2_live/ (per-run logs). Usage:  python compact_measurements.py [output_dir]
(default: code/figures). The paper uses a figure height of 1.75 in.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ART = Path(__file__).resolve().parents[2]
DATA = ART / "data"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
DEVS = [("raspberry_pi", "Raspberry Pi"), ("jetson_nano_orin", "Jetson")]
VIEWS = [("all", "ALL"), ("gt_100", "ALL>100B"), ("uplink_only", "UL"), ("uplink_only_gt_100", "UL>100B")]
C3 = ["#2a78d6", "#eb6834", "#1baf7a"]
# two live runs outside the shortlist rule, selected by mistake (MEASUREMENTS.md section 8): not reported
EXCLUDED = {("raspberry_pi", "buffer_health", "uplink_only", "lgbm", "efficiency"),
            ("jetson_nano_orin", "stalling", "uplink_only", "rf", "performance")}
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
                     "font.serif": ["Times New Roman", "Times", "STIXGeneral"], "mathtext.fontset": "stix",
                     "font.size": 7.5, "axes.linewidth": 0.5, "xtick.major.width": 0.5,
                     "ytick.major.width": 0.5, "axes.edgecolor": "#52514e", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b"})


def clean(x):
    x = np.asarray(x, float)
    if len(x) < 4:
        return x
    q1, q3 = np.percentile(x, [25, 75]); i = q3 - q1
    return x[(x >= q1 - 1.5 * i) & (x <= q3 + 1.5 * i)]


def load(sub):
    out = {}
    for f in sorted(DATA.glob(f"w2_live/*/live/{sub}/*/*/*/*_stats.csv")):
        sel = "performance" if f.stem.endswith("performance_stats") else "efficiency"
        out[(f.parts[-7], f.parts[-4], f.parts[-3], f.parts[-2], sel)] = pd.read_csv(f)
    return out


def live(H=1.75):
    me, ref = load("model_execution"), load("feature_extraction")
    fig, axs = plt.subplots(2, 1, figsize=(3.45, H), sharex=True, sharey=True)
    VARC = {"performance": C3[1], "efficiency": C3[0]}
    for a, (dev, dname) in enumerate(DEVS):
        ax = axs[a]
        for vi, (v, _) in enumerate(VIEWS):
            keys = sorted((k for k in me if k[0] == dev and k[2] == v and k not in EXCLUDED),
                          key=lambda t: (t[1] != "stalling", t[3], t[4] != "efficiency"))  # as in the paper: ea before pf
            for j, key in enumerate(keys):
                d = me[key]
                e1, e0 = clean(d.tf_energy), clean(ref[key].tf_energy)
                diff = e1.mean() - e0.mean()
                se = np.sqrt(e1.var(ddof=1) / len(e1) + e0.var(ddof=1) / len(e0))
                ci = stats.t.ppf(0.975, min(len(e1), len(e0)) - 1) * se
                xx = vi - 0.33 + 0.66 * (j + 0.5) / len(keys)
                valid = d.predictions.mean() > 100
                col = VARC[key[4]] if valid else "#a3a29e"
                stall = key[1] == "stalling"
                ax.errorbar(xx, diff, yerr=ci, fmt="o" if stall else "^", ms=3.6, color=col, mec=col,
                            mfc=col if stall else "white", mew=0.8, elinewidth=0.6, capsize=0, zorder=3)
        ax.axhline(0, color="#52514e", lw=0.5, zorder=1)
        ax.yaxis.set_major_locator(plt.MultipleLocator(50))
        ax.grid(True, axis="y", lw=0.3, color="#d9d8d4"); ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.set_title(dname, loc="left", fontsize=7.5, pad=3)
        for vi in range(1, 4):
            ax.axvline(vi - 0.5, color="#d9d8d4", lw=0.4)
    axs[1].set_xticks(range(4)); axs[1].set_xticklabels([vl for _, vl in VIEWS])
    axs[1].set_xlim(-0.5, 3.5)
    h = [plt.Line2D([], [], color=C3[1], marker="s", ls="", ms=4, label="performance-first"),
         plt.Line2D([], [], color=C3[0], marker="s", ls="", ms=4, label="energy-aware"),
         plt.Line2D([], [], color="#a3a29e", marker="s", ls="", ms=4, label="almost no predictions"),
         plt.Line2D([], [], color="#52514e", marker="o", ls="", ms=3.6, label="stalling"),
         plt.Line2D([], [], color="#52514e", marker="^", ls="", ms=3.6, mfc="white", label="buffer health")]
    fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, fontsize=6.5, handletextpad=0.2, columnspacing=0.8)
    fig.supylabel(r"$\Delta E$ vs. reference [J]", x=0.012, y=0.45, fontsize=7.5)
    fig.tight_layout(rect=(0.045, 0, 1, 1 - 0.375 / H), pad=0.3, h_pad=0.4)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "live_delta.pdf")
    print("saved", OUT / "live_delta.pdf")


if __name__ == "__main__":
    live()
