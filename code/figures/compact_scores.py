"""IEEE-size figures from the HPO databases and the full-test results (house style of plot_figures.py).
hpo_pareto.pdf      : Fig. 2  left: trials per search; right: example front with pf, knee, chord
                      (source: data_masters_thesis/measurements/model_hpo/*.db, read with sqlite3)
test_scores.pdf     : Fig. 4  full-test AP, F1 (stalling), R^2, MAE (buffer health) per family and view
                      (source: data_masters_thesis/general_data/*_predictions/*_results.json via scores_table.csv)
Run: python3 hpo_figures.py   [preview_dir]   (writes PDFs to ../figures, optional PNG previews)
"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "noms_paper_v0" / "figures_src"))
from hpo_data import FAMS, VIEWS

HERE = Path(__file__).resolve().parents[2] / "noms_paper_v0" / "figures_src"
OUT = Path(__file__).resolve().parents[1] / "figures"
C3 = ["#2a78d6", "#eb6834", "#1baf7a"]
FAM = {"rf": (C3[0], "o", "RF"), "lgbm": (C3[1], "o", "LGBM"), "ffnn": (C3[2], "o", "FFNN"),
       "cnn": (C3[0], "^", "CNN"), "lstm": (C3[1], "^", "LSTM"), "transformer": (C3[2], "^", "TF")}
VLAB = {"all": "ALL", "gt_100": "ALL>100B", "uplink_only": "UL", "uplink_only_gt_100": "UL>100B"}
VMK = {"all": "o", "gt_100": "s", "uplink_only": "^", "uplink_only_gt_100": "D"}
GRID, INK, MUTED = "#d9d8d4", "#52514e", "#a3a29e"
BOUND = {"rf": 15000, "lgbm": 15000}          # J; neural networks 2500 J (selection.tex)
EXAMPLE = ("stalling", "cnn", "all")
plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.linewidth": 0.5,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "axes.edgecolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.labelcolor": "#0b0b0b"})


def style(ax, grid="both"):
    ax.grid(True, axis=grid, lw=0.3, color=GRID); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)


PREVIEW = Path(sys.argv[1]) if len(sys.argv) > 1 else None   # optional folder for PNG previews


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    if PREVIEW: fig.savefig(PREVIEW / f"{name}.png", dpi=300)


def scores_fig():
    df = pd.read_csv(HERE / "scores_table.csv")
    panels = [("stalling", "test", "AP [%]", 100), ("stalling", "test2", "F1 [%]", 100),
              ("buffer_health", "test", "$R^2$ [%]", 100), ("buffer_health", "test2", "MAE [s]", 1)]
    fig, axs = plt.subplots(1, 4, figsize=(7.16, 1.53))
    for ax, (k, col, lab, mul) in zip(axs, panels):
        for vi, v in enumerate(VIEWS):
            for fi, f in enumerate(FAMS):
                c, mk, _ = FAM[f]; x = vi + (fi - 2.5) * 0.13
                d = df[(df.kqi == k) & (df.view == v) & (df.fam == f)].set_index("var")[col] * mul
                ax.plot([x, x], [d.pf, d.ea], color=c, lw=0.6, alpha=0.7, zorder=1)
                ax.scatter(x, d.pf, s=12, marker=mk, color=c, edgecolor="white", lw=0.3, zorder=3)
                ax.scatter(x, d.ea, s=12, marker=mk, facecolor="white", edgecolor=c, lw=0.8, zorder=3)
        ax.set_xticks(range(4)); ax.set_xticklabels([VLAB[v].replace(">", "\n>") for v in VIEWS], fontsize=6.8)
        for vi in range(1, 4): ax.axvline(vi - 0.5, color=GRID, lw=0.4)
        ax.set_xlim(-0.5, 3.5); ax.set_ylabel(lab)
        ax.yaxis.set_major_locator(plt.MultipleLocator({"AP [%]": 10, "F1 [%]": 5, "$R^2$ [%]": 20, "MAE [s]": 5}[lab]))  # original ticks
        ax.set_title("Stalling" if k == "stalling" else "Buffer health", fontsize=7.5)
        style(ax, "y")
    h = [plt.Line2D([], [], color=c, marker=mk, ls="", ms=4, label=l) for c, mk, l in FAM.values()]
    h += [plt.Line2D([], [], color=INK, marker="o", ls="", ms=4, label="performance-first"),
          plt.Line2D([], [], color=INK, marker="o", ls="", ms=4, mfc="white", label="energy-aware")]
    fig.legend(handles=h, loc="upper center", ncol=8, frameon=False, fontsize=7, handletextpad=0.2, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.238 / 1.53), pad=0.3, w_pad=0.9)
    save(fig, "test_scores")


