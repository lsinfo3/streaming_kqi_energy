"""IEEE-size figures for the paper draft.
offline_pairs.pdf : W1 energy vs score, pf -> ea pairs (source: master_thesis/results/models/selected/*.csv)
live_delta.pdf    : W2 energy with inference minus feature-preparation reference, from the per-run logs
                    (data_masters_thesis/measurements/model_execution/*/live), Welch 95 % intervals
"""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from pathlib import Path
HERE = Path(__file__).resolve().parents[2] / "noms_paper_v0" / "figures_src"
SEL = HERE.parents[1] / "master_thesis/results/models/selected"
OUT = Path(__file__).resolve().parents[1] / "figures"
DEVS = [("raspberry_pi", "Raspberry Pi"), ("jetson_nano_orin", "Jetson")]
VIEWS = [("all", "ALL"), ("gt_100", "ALL>100B"), ("uplink_only", "UL"), ("uplink_only_gt_100", "UL>100B")]
KQIS = [("stalling", "Stalling AP [%]"), ("buffer_health", "Buffer $R^2$ [%]")]
C3 = ["#2a78d6", "#eb6834", "#1baf7a"]            # validated all-pairs (dataviz validate_palette.js)
FAM = {"rf": (C3[0], "o", "RF"), "lgbm": (C3[1], "o", "LGBM"), "ffnn": (C3[2], "o", "FFNN"),
       "cnn": (C3[0], "^", "CNN"), "lstm": (C3[1], "^", "LSTM"), "transformer": (C3[2], "^", "TF")}
plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.linewidth": 0.5,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "axes.edgecolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b"})

def offline():
    fig, axs = plt.subplots(2, 2, figsize=(7.16, 2.7), sharex="row", sharey="row")
    for r, (kq, ylab) in enumerate(KQIS):
        for c, (dev, dname) in enumerate(DEVS):
            ax = axs[r, c]
            for v, _ in VIEWS:
                df = pd.read_csv(SEL / dev / f"{kq}_{v}.csv", index_col=0)
                for fam, (col, mk, _) in FAM.items():
                    pf = df.loc[f"{kq}_{v}_{fam}_performance"]; ea = df.loc[f"{kq}_{v}_{fam}_efficiency"]
                    ax.plot([pf.energy, ea.energy], [pf.score * 100, ea.score * 100], color=col, lw=0.6, alpha=0.6, zorder=1)
                    ax.scatter(pf.energy, pf.score * 100, s=16, marker=mk, color=col, edgecolor="white", lw=0.4, zorder=3)
                    ax.scatter(ea.energy, ea.score * 100, s=16, marker=mk, facecolor="white", edgecolor=col, lw=0.9, zorder=3)
            ax.set_xscale("log")
            ax.grid(True, which="major", lw=0.3, color="#d9d8d4"); ax.set_axisbelow(True)
            for s in ("top", "right"): ax.spines[s].set_visible(False)
            if r == 0: ax.set_title(dname, fontsize=8)
            if c == 0: ax.set_ylabel(ylab)
            if r == 1: ax.set_xlabel("Energy for 100,000 predictions (W1) [J]")
    axs[1, 0].set_ylim(bottom=45)
    from matplotlib.ticker import MultipleLocator
    for ax in axs.flat: ax.yaxis.set_major_locator(MultipleLocator(10))   # same 10-pp ticks in both rows
    h = [plt.Line2D([], [], color=col, marker=mk, ls="", ms=4.5, label=lab) for col, mk, lab in FAM.values()]
    h += [plt.Line2D([], [], color="#52514e", marker="o", ls="", ms=4.5, label="performance-first"),
          plt.Line2D([], [], color="#52514e", marker="o", ls="", ms=4.5, mfc="white", label="energy-aware")]
    fig.legend(handles=h, loc="upper center", ncol=8, frameon=False, fontsize=7, handletextpad=0.2, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.225 / 2.7), pad=0.3, w_pad=0.8, h_pad=0.6)
    fig.savefig(OUT / "offline_pairs.pdf")

# two live runs outside the shortlist rule, selected by mistake (thesis author, Nov 2026): not reported
EXCLUDED_LIVE = {("pi", "buffer_health", "uplink_only", "lgbm", "ea"), ("jn", "stalling", "uplink_only", "rf", "pf")}

def live():
    """W2 from the per-run logs: mean energy with inference minus the reference, Welch 95 % interval (30 runs each)."""
    import sys; sys.path.insert(0, str(HERE.parent / "model"))
    from logdata import execution, clean
    from scipy import stats
    x = execution("live"); k = ["dev", "kqi", "view", "fam", "variant"]
    me = x[x["sub"] == "model_execution"]; ref = x[x["sub"] == "feature_extraction"]
    fig, axs = plt.subplots(2, 1, figsize=(3.45, 2.0), sharex=True, sharey=True)
    VARC = {"pf": C3[1], "ea": C3[0]}
    for a, (dev, dname) in enumerate(DEVS):
        ax = axs[a]; dv = {"raspberry_pi": "pi", "jetson_nano_orin": "jn"}[dev]
        for vi, (v, vl) in enumerate(VIEWS):
            cands = me[(me.dev == dv) & (me.view == v)].groupby(k)
            keys = sorted((t for t in cands.groups if t not in EXCLUDED_LIVE),
                          key=lambda t: (t[1] != "stalling", t[3], t[4]))
            for j, key in enumerate(keys):
                d = cands.get_group(key); r = ref.set_index(k).loc[[key]]
                e1, e0 = clean(d.tf_energy), clean(r.tf_energy)
                diff = e1.mean() - e0.mean(); se = np.sqrt(e1.var(ddof=1) / len(e1) + e0.var(ddof=1) / len(e0))
                ci = stats.t.ppf(0.975, min(len(e1), len(e0)) - 1) * se
                xx = vi - 0.33 + 0.66 * (j + 0.5) / len(keys)
                valid = d.predictions.mean() > 100
                col = VARC[key[4]] if valid else "#a3a29e"
                stall = key[1] == "stalling"
                ax.errorbar(xx, diff, yerr=ci, fmt="o" if stall else "^", ms=3.6, color=col, mec=col,
                            mfc=col if stall else "white", mew=0.8, elinewidth=0.6, capsize=0, zorder=3)
        ax.axhline(0, color="#52514e", lw=0.5, zorder=1)
        ax.yaxis.set_major_locator(plt.MultipleLocator(50))   # original ticks 0, 50, 100
        ax.grid(True, axis="y", lw=0.3, color="#d9d8d4"); ax.set_axisbelow(True)
        for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
        
        ax.set_title(dname, loc="left", fontsize=7.5, pad=3)
        for vi in range(1, 4): ax.axvline(vi - 0.5, color="#d9d8d4", lw=0.4)
    axs[1].set_xticks(range(4)); axs[1].set_xticklabels([vl for _, vl in VIEWS])
    axs[1].set_xlim(-0.5, 3.5)
    h = [plt.Line2D([], [], color=C3[1], marker="s", ls="", ms=4, label="performance-first"),
         plt.Line2D([], [], color=C3[0], marker="s", ls="", ms=4, label="energy-aware"),
         plt.Line2D([], [], color="#a3a29e", marker="s", ls="", ms=4, label="almost no predictions"),
         plt.Line2D([], [], color="#52514e", marker="o", ls="", ms=3.6, label="stalling"),
         plt.Line2D([], [], color="#52514e", marker="^", ls="", ms=3.6, mfc="white", label="buffer health")]
    fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, fontsize=6.5, handletextpad=0.2, columnspacing=0.8)
    fig.supylabel(r"$\Delta E$ vs. reference [J]", x=0.012, y=0.45, fontsize=7.5)
    fig.tight_layout(rect=(0.045, 0, 1, 1 - 0.375 / 2.0), pad=0.3, h_pad=0.4)
    fig.savefig(OUT / "live_delta.pdf")
