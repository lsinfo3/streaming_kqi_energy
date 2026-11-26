"""Reproduce the live single-stream monitoring results (workload W2, Section IV-D and Fig. 4).

For every live candidate, the five-minute energy with inference is compared with its reference run
without inference (feature preparation only). Means are taken after the 1.5-IQR outlier filter.

Usage:  python repro_w2.py [../../data]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "data"
DEVS = {"raspberry_pi": "Raspberry Pi", "jetson_nano_orin": "Jetson"}
# two live runs outside the shortlist rule, selected by mistake (MEASUREMENTS.md section 8): not reported
EXCLUDED = {("raspberry_pi", "buffer_health", "uplink_only", "lgbm", "efficiency"),
            ("jetson_nano_orin", "stalling", "uplink_only", "rf", "performance")}
VALID_PREDICTIONS = 100        # candidates with fewer predictions per run on average are "almost no predictions"


def clean(x):
    x = np.asarray(x, float)
    if len(x) < 4:
        return x
    q1, q3 = np.percentile(x, [25, 75]); i = q3 - q1
    return x[(x >= q1 - 1.5 * i) & (x <= q3 + 1.5 * i)]


def load(sub):
    rows = []
    for f in sorted(DATA.glob(f"w2_live/*/live/{sub}/*/*/*/*_stats.csv")):
        dev, kqi, view, fam = f.parts[-7], f.parts[-4], f.parts[-3], f.parts[-2]
        sel = "performance" if f.stem.endswith("performance_stats") else "efficiency"
        rows.append(((dev, kqi, view, fam, sel), pd.read_csv(f)))
    return dict(rows)


def baseline_power(dev):
    """P0: whole-device power while capturing one stream without dissection (W3), mean over the four views."""
    p = []
    for v in ("header_only", "header_only_gt_100", "header_only_uplink_only", "header_only_uplink_only_gt_100"):
        x = pd.read_csv(DATA / "w3_capture" / dev / v / f"{v}_1_overlaid_stats.csv")
        e = x.tf_energy.to_numpy(); keep = np.isin(e, clean(e))
        p.append((x.tf_energy / x.tf_duration)[keep].mean())
    return float(np.mean(p))


def w1_per_prediction():
    """c_m = (E_W1 - P0 T_W1) / 100,000: inference energy per prediction above the baseline (Table IV)."""
    w = pd.read_csv(DATA / "derived" / "repro_w1.csv")
    w = w[w["mode"] == "offline"]
    p0 = {d: baseline_power(d) for d in DEVS}
    return {(r.dev, r.kqi, r.view, r.fam, r.sel): (r.E - p0[r.dev] * r.dur) / 100_000 for r in w.itertuples()}, p0


def main():
    me, ref = load("model_execution"), load("feature_extraction")
    w1, p0 = w1_per_prediction()
    rows = []
    for key, d in me.items():
        if key in EXCLUDED:
            continue
        e1, e0 = clean(d.tf_energy), clean(ref[key].tf_energy)
        rows.append(dict(dev=key[0], kqi=key[1], view=key[2], fam=key[3], sel=key[4],
                         dE=e1.mean() - e0.mean(), share=(e1.mean() - e0.mean()) / e0.mean() * 100,
                         total=e1.mean(), preds=d.predictions.mean(),
                         w1_live=w1[key] * d.predictions.mean(), w1_pp=w1[key]))
    t = pd.DataFrame(rows)
    t["valid"] = t.preds > VALID_PREDICTIONS
    print("== Section IV-D: live single-stream monitoring (W2) ==")
    for dev, name in DEVS.items():
        g = t[t.dev == dev]; v = g[g.valid]
        print(f"\n{name}: {len(g)} candidates, {len(v)} valid "
              f"(valid: {v.preds.min():.0f}-{v.preds.max():.0f} predictions per run)")
        inv = g[~g.valid]
        for r in inv.itertuples():
            print(f"  almost no predictions: {r.kqi} {r.view} {r.fam} {r.sel}, {r.preds:.1f} per run on average")
        for sel, lab in (("performance", "performance-first"), ("efficiency", "energy-aware")):
            s = v[v.sel == sel]
            print(f"  {lab:17s} n={len(s):2d}  adds {s.dE.min():5.1f} to {s.dE.max():5.1f} J "
                  f"({s.share.min():.1f} % to {s.share.max():.1f} % of the reference)")
        print(f"  largest valid increment: {v.dE.max():.1f} J ({v.share.max():.1f} %)")
        s = v[v.sel == "performance"]
        print(f"  performance-first: W1 inference energy per prediction (c_m, P0 = {p0[dev]:.2f} W) varies "
              f"{s.w1_pp.max() / s.w1_pp.min():.0f}-fold; c_m x live predictions is at most {v.w1_live.max():.1f} J")
    pf = t[t.valid & (t.sel == "performance")].pivot_table(index=["kqi", "view", "fam"], columns="dev", values="total")
    pf = pf.dropna()
    ratio = (pf.jetson_nano_orin / pf.raspberry_pi - 1) * 100
    print(f"\nIdentical performance-first models on both devices (n={len(pf)}): "
          f"the Jetson uses a median {ratio.median():.0f} % more total energy than the Raspberry Pi")


if __name__ == "__main__":
    main()
