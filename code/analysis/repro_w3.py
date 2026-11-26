"""Reproduce the multi-stream capture and dissection results (workload W3, Section IV-E).

* energy of five-minute capture without dissection at 100 streams, per traffic view;
* the load at which the capture rate declines, per view and device: a load counts as loss-free while
  its captured-to-sent packet ratio reaches 98 % of the loss-free reference ratio of the same view;
* deferred dissection of a stored five-minute, 100-stream ALL capture (duration and energy).

Means are taken after the 1.5-IQR outlier filter. Usage:  python repro_w3.py [../../data]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "data"
W3 = DATA / "w3_capture"
DEVS = {"raspberry_pi": "Raspberry Pi", "jetson_nano_orin": "Jetson"}
VIEWS = {"ALL": "header_only", "ALL>100B": "header_only_gt_100", "UL": "header_only_uplink_only",
         "UL>100B": "header_only_uplink_only_gt_100"}
REPLAY_LIMIT = 200          # reliable traffic generation of the replay workstation
LOSS_FREE = 0.98


def clean(x):
    x = np.asarray(x, float)
    if len(x) < 4:
        return x
    q1, q3 = np.percentile(x, [25, 75]); i = q3 - q1
    return x[(x >= q1 - 1.5 * i) & (x <= q3 + 1.5 * i)]


def runs(dev, mode, S, dissection):
    suf = "_dissection" if dissection else ""
    f = W3 / dev / f"{mode}{suf}" / f"{mode}_{S}_overlaid{suf}_stats.csv"
    return pd.read_csv(f) if f.exists() else None


def loads(dev, mode, dissection):
    suf = "_dissection" if dissection else ""
    out = []
    for f in (W3 / dev / f"{mode}{suf}").glob(f"{mode}_*_overlaid{suf}_stats.csv"):
        s = f.name[len(mode) + 1:].split("_")[0]
        if s.isdigit():
            out.append(int(s))
    return sorted(out)


def ratio(x):
    """Median captured-to-sent packet ratio. Where the capture counter is unavailable (-1, Jetson above 100
    streams), the packets that reached dissection are counted instead."""
    cap = x.captured_packets if (x.captured_packets > 0).all() else x.get("processed_packets")
    if cap is None or (cap <= 0).any():
        return float("nan")
    return float((cap / x.actual).median())


def onset(dev, view, mode):
    """Last loss-free load, first load below 98 % of the reference ratio, and the highest load measured."""
    # reference: the loss-free 100-stream run of the same view without dissection; for UL>100B the frame-length
    # filter keeps small packets that are only discarded after dissection, so its reference is the 100-stream
    # run with dissection (loss-free)
    ref = ratio(runs(dev, VIEWS[view], 100, view == "UL>100B"))
    measured = [S for S in loads(dev, mode, True) if 1 < S <= REPLAY_LIMIT]
    last_ok, rows = None, []
    for S in measured:
        x = runs(dev, mode, S, True); rr = ratio(x) / ref
        rows.append(f"{S}:{rr:.3f}" + ("" if len(x) >= 8 else f"[{len(x)} run{'s' if len(x) > 1 else ''}]"))
        if rr < LOSS_FREE:
            return last_ok, S, measured[-1], rows
        last_ok = S
    return last_ok, None, measured[-1], rows


def main():
    print("== Section IV-E: capture without dissection at 100 streams, five minutes [J] ==")
    for dev, name in DEVS.items():
        e = {v: clean(runs(dev, m, 100, False).tf_energy).mean() for v, m in VIEWS.items()}
        print(f"  {name:12s} " + "  ".join(f"{v} {x:.0f}" for v, x in e.items())
              + f"   (range {min(e.values()):.0f}-{max(e.values()):.0f})")

    print("\n== Section IV-E: load at which the capture rate declines (with dissection) ==")
    print("   relative capture ratio per load; loads with fewer than 8 runs are screening runs, marked [n runs]")
    for dev, name in DEVS.items():
        for view, mode in list(VIEWS.items()) + [("FULL", "complete_capture")]:
            ok, bad, top, rows = onset(dev, "ALL" if view == "FULL" else view, mode)
            label = "full-packet capture" if view == "FULL" else view
            verdict = (f"loss-free up to {ok}, below 98 % from {bad} streams" if bad
                       else f"no decline up to {top} streams")
            print(f"  {name:12s} {label:20s} {verdict}")
            print(f"  {'':12s} {'':20s} {' '.join(rows)}")

    print("\n== Section IV-E: deferred dissection of a five-minute, 100-stream ALL capture ==")
    for dev, name in DEVS.items():
        x = pd.read_csv(DATA / "processing" / dev / "header_only_static_processing"
                        / "header_only_100_static_processing_stats.csv")
        keep = np.isin(x.tf_energy, clean(x.tf_energy))
        E, T = x.tf_energy[keep].mean(), x.tf_duration[keep].mean()
        cap = runs(dev, "header_only", 100, False)
        Ec, Tc = clean(cap.tf_energy).mean(), cap.tf_duration.mean()
        print(f"  {name:12s} {T / 60:.1f} min, {E / 1000:.2f} kJ  "
              f"({T / Tc:.1f} x the capture duration, {E / Ec:.1f} x the capture energy)")


if __name__ == "__main__":
    main()
