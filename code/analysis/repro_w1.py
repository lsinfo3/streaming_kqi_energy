"""Reproduce the offline-inference (W1) energy results of the accompanying paper.

Reads the per-configuration measurement CSVs of workload W1 and reproduces:

  * Table II  - whole-device energy for 100,000 predictions, ALL view, both
                devices, performance-first and energy-aware
  * Section V-B - median energy-aware savings per device and per KQI, and the
                number of configuration pairs whose energy increases
  * Section V-C - the batch-1 to batch-32 energy factor, the baseline share of
                W1 energy, and the number of configurations finishing under 10 s
  * Table II footnote - the share of configurations whose 95 % confidence
                interval stays below 2 % of the mean

Each input CSV holds 30 repetitions of one configuration. Energy is the
whole-device AC reading `tf_energy` in joules. Means are taken after removing
values beyond 1.5 interquartile ranges, as in the paper.

Usage:
    python repro_w1.py ../../data/w1_offline
    python repro_w1.py ../../data/w1_offline --out w1_summary.csv
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

# Twelve-hour idle means reported in the paper, used for the baseline share.
IDLE_POWER_W = {"raspberry_pi": 2.56, "jetson_nano_orin": 4.27}
DEVICES = ("raspberry_pi", "jetson_nano_orin")
BATCH_MODES = {"offline": "batch 1", "offline_fixed_batch": "batch 32"}
# On the Raspberry Pi the batched stalling RF and LGBM runs do not reproduce the
# unbatched predictions, so the paper excludes them from batched comparisons.
EXCLUDED_BATCHED = {"device": "raspberry_pi", "kqi": "stalling", "families": ("rf", "lgbm")}


def iqr_filtered_mean(values):
    """Mean, outlier count and relative 95 % CI half-width after a 1.5-IQR filter."""
    q1, q3 = np.percentile(values, [25, 75])
    span = q3 - q1
    kept = values[(values >= q1 - 1.5 * span) & (values <= q3 + 1.5 * span)]
    ci_rel = 1.96 * kept.std(ddof=1) / np.sqrt(len(kept)) / kept.mean() * 100
    return kept.mean(), len(values) - len(kept), ci_rel


def load(root):
    """Collect one row per configuration and batch mode."""
    rows = []
    for device in DEVICES:
        for mode in BATCH_MODES:
            pattern = os.path.join(root, device, mode, "*", "*", "*", "*_stats.csv")
            for path in sorted(glob.glob(pattern)):
                parts = path.split(os.sep)
                kqi, view, family = parts[-4], parts[-3], parts[-2]
                selection = os.path.basename(path).replace("_stats.csv", "").split("_")[-1]
                df = pd.read_csv(path)
                energy, n_outliers, ci_rel = iqr_filtered_mean(df["tf_energy"].to_numpy())
                row = {
                    "device": device,
                    "batch": BATCH_MODES[mode],
                    "kqi": kqi,
                    "view": view,
                    "family": family,
                    "selection": selection,
                    "energy_j": energy,
                    "duration_s": df["tf_duration"].mean(),
                    "runs": len(df),
                    "outliers": n_outliers,
                    "ci_rel_pct": ci_rel,
                }
                for column in ("ap", "f1", "r2", "mae", "runtime", "n_parameters"):
                    if column in df:
                        row[column] = df[column].iloc[0]
                rows.append(row)
    if not rows:
        raise SystemExit(f"no *_stats.csv found under {root}")
    return pd.DataFrame(rows)


def report(df):
    one = df[df.batch == "batch 1"]
    thirtytwo = df[df.batch == "batch 32"]

    print(f"series loaded: {len(df)} ({len(one)} at batch 1, {len(thirtytwo)} at batch 32)\n")

    print("== Table II: whole-device energy for 100,000 predictions, ALL view [J] ==")
    for kqi in ("stalling", "buffer_health"):
        table = one[(one.kqi == kqi) & (one.view == "all")].pivot_table(
            index="family", columns=["device", "selection"], values="energy_j"
        )
        print(f"\n{kqi}")
        print(table.round(0).to_string())

    print("\n== Section V-B: median energy-aware saving, 48 pairs per device ==")
    for device in DEVICES:
        pairs = (
            one[one.device == device]
            .pivot_table(index=["kqi", "view", "family"], columns="selection", values="energy_j")
            .dropna()
        )
        saving = (pairs["efficiency"] - pairs["performance"]) / pairs["performance"] * 100
        print(
            f"  {device:<17} n={len(saving)}  median {saving.median():+.1f} %"
            f"  (stalling {saving.xs('stalling').median():+.1f} %,"
            f" buffer health {saving.xs('buffer_health').median():+.1f} %)"
            f"  increases: {(saving > 0).sum()}"
        )

    print("\n== Section V-C: batch-1 to batch-32 energy factor ==")
    for device in DEVICES:
        key = ["kqi", "view", "family", "selection"]
        a = one[one.device == device].set_index(key)["energy_j"]
        b = thirtytwo[thirtytwo.device == device].set_index(key)["energy_j"]
        joined = pd.concat([a.rename("b1"), b.rename("b32")], axis=1).dropna().reset_index()
        if device == EXCLUDED_BATCHED["device"]:
            mask = (joined.kqi == EXCLUDED_BATCHED["kqi"]) & joined.family.isin(
                EXCLUDED_BATCHED["families"]
            )
            joined = joined[~mask]  # predictions not preserved; see MEASUREMENTS.md section 8
        factor = (joined.b1 / joined.b32).median()
        print(
            f"  {device:<17} n={len(joined)}  median factor {factor:.2f}"
            f"  -> {100 * (1 - 1 / factor):.0f} % saving"
        )

    print("\n== Section V-C: baseline share of W1 energy, and short runs ==")
    for device in DEVICES:
        sub = one[one.device == device]
        share = 100 * (IDLE_POWER_W[device] * sub.duration_s / sub.energy_j).median()
        light = sub[(sub.selection == "efficiency") & sub.family.isin(["rf", "lgbm", "ffnn"])]
        print(
            f"  {device:<17} median baseline share {share:.0f} %"
            f"  | energy-aware tree/FFNN runs under 10 s: {(light.duration_s < 10).sum()}/{len(light)}"
            f"  | shortest run {sub.duration_s.min():.1f} s"
        )

    print("\n== Table II footnote: confidence intervals ==")
    for label, sub in (("batch 1", one), ("batch 32", thirtytwo)):
        print(
            f"  {label:<9} CI below 2 % of the mean for {(sub.ci_rel_pct < 2).mean() * 100:.1f} %"
            f" of configurations, worst {sub.ci_rel_pct.max():.2f} %"
            f" | outliers removed {sub.outliers.sum()}/{sub.runs.sum()}"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", help="directory holding <device>/<offline|offline_fixed_batch>/...")
    ap.add_argument("--out", help="write the per-configuration table to this CSV")
    args = ap.parse_args()

    df = load(args.root)
    report(df)
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
