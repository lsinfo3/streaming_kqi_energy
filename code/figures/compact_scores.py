"""Fig. 2: full-test performance by traffic view and model family.

AP and F1 for stalling, R^2 and MAE for buffer health, performance-first (filled) and energy-aware
(open) configuration per family and view.

Source: data/derived/repro_scores.csv (scores recomputed from the exported prediction arrays; MAE in s).
Usage:  python compact_scores.py [output_dir]   (default: ../figures, i.e. code/figures)
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ART = Path(__file__).resolve().parents[2]
DATA = ART / "data" / "derived" / "repro_scores.csv"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
FAMS = ["rf", "lgbm", "ffnn", "cnn", "lstm", "transformer"]
VIEWS = ["all", "gt_100", "uplink_only", "uplink_only_gt_100"]
C3 = ["#2a78d6", "#eb6834", "#1baf7a"]
FAM = {"rf": (C3[0], "o", "RF"), "lgbm": (C3[1], "o", "LGBM"), "ffnn": (C3[2], "o", "FFNN"),
       "cnn": (C3[0], "^", "CNN"), "lstm": (C3[1], "^", "LSTM"), "transformer": (C3[2], "^", "TF")}
VLAB = {"all": "ALL", "gt_100": "ALL>100B", "uplink_only": "UL", "uplink_only_gt_100": "UL>100B"}
GRID, INK = "#d9d8d4", "#52514e"
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
                     "font.serif": ["Times New Roman", "Times", "STIXGeneral"], "mathtext.fontset": "stix",
                     "font.size": 7.5, "axes.linewidth": 0.5, "xtick.major.width": 0.5,
                     "ytick.major.width": 0.5, "axes.edgecolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.labelcolor": "#0b0b0b"})


def load():
    d = pd.read_csv(DATA)
    p = d.cfg.str.extract(r"^(?:stalling|buffer_health)_(.+)_(rf|lgbm|ffnn|cnn|lstm|transformer)_(performance|efficiency)$")
    d["view"], d["fam"], d["var"] = p[0], p[1], p[2].map({"performance": "pf", "efficiency": "ea"})
    d["test"] = d.ap.where(d.kqi == "stalling", d.r2)          # AP or R^2 [%]
    d["test2"] = d.f1.where(d.kqi == "stalling", d.mae_json)  # F1 [%] or MAE [s]
    return d


def scores_fig():
    df = load()
    panels = [("stalling", "test", "AP [%]"), ("stalling", "test2", "F1 [%]"),
              ("buffer_health", "test", "$R^2$ [%]"), ("buffer_health", "test2", "MAE [s]")]
    fig, axs = plt.subplots(1, 4, figsize=(7.16, 1.53))
    for ax, (k, col, lab) in zip(axs, panels):
        for vi, v in enumerate(VIEWS):
            for fi, f in enumerate(FAMS):
                c, mk, _ = FAM[f]; x = vi + (fi - 2.5) * 0.13
                d = df[(df.kqi == k) & (df.view == v) & (df.fam == f)].set_index("var")[col]
                ax.plot([x, x], [d.pf, d.ea], color=c, lw=0.6, alpha=0.7, zorder=1)
                ax.scatter(x, d.pf, s=12, marker=mk, color=c, edgecolor=c, lw=0.8, zorder=3)
                ax.scatter(x, d.ea, s=12, marker=mk, facecolor="white", edgecolor=c, lw=0.8, zorder=3)
        ax.set_xticks(range(4)); ax.set_xticklabels([VLAB[v].replace(">", "\n>") for v in VIEWS], fontsize=6.8)
        for vi in range(1, 4):
            ax.axvline(vi - 0.5, color=GRID, lw=0.4)
        ax.set_xlim(-0.5, 3.5); ax.set_ylabel(lab)
        ax.yaxis.set_major_locator(plt.MultipleLocator({"AP [%]": 10, "F1 [%]": 5, "$R^2$ [%]": 20, "MAE [s]": 5}[lab]))
        ax.set_title("Stalling" if k == "stalling" else "Buffer health", fontsize=7.5)
        ax.grid(True, axis="y", lw=0.3, color=GRID); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    h = [plt.Line2D([], [], color=c, marker=mk, ls="", ms=4, label=l) for c, mk, l in FAM.values()]
    h += [plt.Line2D([], [], color=INK, marker="o", ls="", ms=4, label="performance-first"),
          plt.Line2D([], [], color=INK, marker="o", ls="", ms=4, mfc="white", label="energy-aware")]
    fig.legend(handles=h, loc="upper center", ncol=8, frameon=False, fontsize=7, handletextpad=0.2, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.238 / 1.53), pad=0.3, w_pad=0.9)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "test_scores.pdf")
    print("saved", OUT / "test_scores.pdf")


if __name__ == "__main__":
    scores_fig()
