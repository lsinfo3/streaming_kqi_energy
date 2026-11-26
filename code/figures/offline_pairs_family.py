"""Fig. 3: W1 energy vs. score, performance-first -> energy-aware pairs.

One panel per model family (columns) and KQI (rows). Colour = device, marker = traffic view,
filled = performance-first, open = energy-aware.

Source: data/derived/repro_w1.csv (mode "offline": W1 energy E in J, AP / R2 on the W1 subset).
Usage:  python offline_pairs_family.py [output_dir]   (default: code/figures)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

ART = Path(__file__).resolve().parents[2]
DATA = ART / "data" / "derived" / "repro_w1.csv"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent

C3 = ["#2a78d6", "#eb6834", "#1baf7a"]          # same palette as compact_measurements.py
# device colours differ from Fig. 4, where blue/orange mean energy-aware/performance-first
DEVS = [("raspberry_pi", "Raspberry Pi", C3[2]), ("jetson_nano_orin", "Jetson", "#7a4fb5")]
FAMS = [("rf", "RF"), ("lgbm", "LGBM"), ("ffnn", "FFNN"), ("cnn", "CNN"), ("lstm", "LSTM"), ("transformer", "TF")]
VIEWS = [("all", "ALL", "o"), ("gt_100", "ALL>100B", "s"), ("uplink_only", "UL", "^"),
         ("uplink_only_gt_100", "UL>100B", "D")]
KQIS = [("stalling", "ap", "Stalling AP [%]", (58, 93)), ("buffer_health", "r2", "Buffer $R^2$ [%]", (45, 95))]

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
                     "font.serif": ["Times New Roman", "Times", "STIXGeneral"], "mathtext.fontset": "stix",
                     "font.size": 7.5, "axes.linewidth": 0.5, "xtick.major.width": 0.5,
                     "ytick.major.width": 0.5, "axes.edgecolor": "#52514e", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b"})


def main():
    d = pd.read_csv(DATA)
    d = d[d["mode"] == "offline"]
    H = 2.15  # figure height in inches (legend and axis-label margins below are absolute)
    fig, axs = plt.subplots(2, 6, figsize=(7.16, H), sharex=True, sharey="row")
    for r, (kq, col_score, ylab, ylim) in enumerate(KQIS):
        for c, (fam, fname) in enumerate(FAMS):
            ax = axs[r, c]
            for dev, _, colr in DEVS:
                for v, _, mk in VIEWS:
                    s = d[(d.dev == dev) & (d.kqi == kq) & (d.fam == fam) & (d.view == v)].set_index("sel")
                    pf, ea = s.loc["performance"], s.loc["efficiency"]
                    ypf, yea = pf[col_score] * 100, ea[col_score] * 100
                    ax.plot([pf.E, ea.E], [ypf, yea], color=colr, lw=0.6, alpha=0.6, zorder=1)
                    ax.plot(pf.E, ypf, mk, color=colr, mec="white", mew=0.3, ms=3.6, zorder=3)
                    ax.plot(ea.E, yea, mk, mfc="white", mec=colr, mew=0.8, ms=3.4, zorder=3)
            ax.set_xscale("log"); ax.set_xlim(18, 1.2e4); ax.set_xticks([1e2, 1e3, 1e4])
            ax.set_ylim(*ylim); ax.yaxis.set_major_locator(MultipleLocator(10))
            ax.grid(True, which="major", lw=0.3, color="#d9d8d4"); ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(length=2, pad=1.5)
            if r == 0:
                ax.set_title(fname, fontsize=8, pad=2)
            if c == 0:
                ax.set_ylabel(ylab)
    fig.text(0.5, 0.0294 / H, "Energy for 100,000 predictions (W1) [J]", ha="center")
    h = [Line2D([], [], color=colr, lw=1.2, label=name) for _, name, colr in DEVS]
    h += [Line2D([], [], ls="", marker=mk, color="#52514e", ms=4, label=lab) for _, lab, mk in VIEWS]
    # filled/open shown as a short pair segment so it does not read as the ALL circle
    h += [Line2D([], [], color="#9a9893", lw=0.8, marker="o", markevery=[0], ms=4, label="performance-first"),
          Line2D([], [], color="#9a9893", lw=0.8, marker="o", markevery=[1], mfc="white", mec="#9a9893",
                 ms=4, label="energy-aware")]
    fig.legend(handles=h, loc="upper center", ncol=8, frameon=False, fontsize=7, handletextpad=0.3,
               columnspacing=1.0, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0.0858 / H, 1, 1 - 0.196 / H), pad=0.3, w_pad=0.4, h_pad=0.5)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "offline_pairs_family.pdf")
    print("saved", OUT / "offline_pairs_family.pdf")


if __name__ == "__main__":
    main()
