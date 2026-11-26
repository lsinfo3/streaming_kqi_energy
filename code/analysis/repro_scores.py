"""Recompute the full-test-set prediction scores of the accompanying paper.

For every selected configuration the original pipeline exported the model's
predictions and the matching ground truth over the full test split, as flat
NumPy arrays. This script recomputes the reported metrics from those arrays:
average precision and F1 at the stored validation-optimal threshold for
stalling, R squared and mean absolute error for buffer health.

Reproduces, among others, the Section IV-A figures for the ALL view: LightGBM
performance-first at 89.4 % average precision and 79.4 % F1 for stalling, and
LSTM performance-first at R squared 93.2 % with a mean absolute error of 5.9 s
for buffer health.

Two conventions matter:

  * Buffer health is stored normalised, as the player value clipped to
    [0, 120000] ms and divided by 120000 (see
    master_thesis/src/processing/preprocessing.py, build_ground_truth_buffer_health).
    Multiply a mean absolute error in those units by 120 to obtain seconds.
  * The exported buffer-health predictions are raw regressor output and are NOT
    clipped to the valid [0, 1] range: individual configurations put up to a
    quarter of their predictions outside it. The paper's metrics are computed
    after clipping, so this script clips too. Clipping reproduces every stored
    r2 and mae field; scoring the raw arrays instead disagrees with the paper by
    up to 1.25 points of R squared.
  * The arrays hold no run identifier, so scores cannot be broken down by run
    or by video without the per-run index arrays the pipeline wrote next to each
    dataset.

The arrays total about 325 MB and are therefore NOT part of this repository.
Point --predictions at the `general_data` directory of the measurement data
release, which must contain stalling_predictions/ and buffer_health_predictions/.

Usage:
    python repro_scores.py --predictions /path/to/general_data --out scores.csv
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, mean_absolute_error, r2_score

BH_SCALE_S = 120.0  # buffer health is stored as seconds/120
TASKS = (("stalling", "stalling_predictions"), ("buffer_health", "buffer_health_predictions"))


def collect(root):
    rows = []
    for kqi, folder in TASKS:
        paths = sorted(glob.glob(os.path.join(root, folder, "*_predictions.npy")))
        if not paths:
            raise SystemExit(f"no prediction arrays found in {os.path.join(root, folder)}")
        for path in paths:
            name = os.path.basename(path).replace("_predictions.npy", "")
            directory = os.path.dirname(path)
            truth = np.load(os.path.join(directory, f"{name}_ground_truth.npy"))
            pred = np.load(path)
            with open(os.path.join(directory, f"{name}_results.json")) as fh:
                stored = json.load(fh)

            row = {
                "kqi": kqi,
                "configuration": name,
                "samples": len(pred),
                # The neural pipeline drops the final partial batch, so its
                # populations are exact multiples of the batch size while the
                # tree models' are not. See FEATURES.md section 8.
                "multiple_of_128": len(pred) % 128 == 0,
            }
            if kqi == "stalling":
                threshold = stored["f1_threshold"]
                row["ap_pct"] = 100 * average_precision_score(truth, pred)
                row["f1_pct"] = 100 * f1_score(truth, pred >= threshold)
                row["ap_pct_stored"] = 100 * stored["ap"]
                row["f1_pct_stored"] = 100 * stored["f1"]
                row["threshold"] = threshold
            else:
                # The exported arrays hold the raw, unclipped regressor output: up to
                # 25 % of values fall outside the valid [0, 1] range of the normalised
                # target. The paper's metrics are computed after clipping to that range,
                # and clipping reproduces every stored r2 field to within 1e-8. Clipped
                # is therefore authoritative; raw is kept for transparency.
                clipped = np.clip(pred, 0.0, 1.0)
                row["r2_pct"] = 100 * r2_score(truth, clipped)
                row["mae_s"] = BH_SCALE_S * mean_absolute_error(truth, clipped)
                row["r2_pct_raw"] = 100 * r2_score(truth, pred)
                row["mae_s_raw"] = BH_SCALE_S * mean_absolute_error(truth, pred)
                row["outside_range_pct"] = 100 * float(((pred < 0) | (pred > 1)).mean())
                row["r2_pct_stored"] = 100 * stored["r2"] if "r2" in stored else np.nan
                # The stored mae is already in seconds, unlike the arrays.
                row["mae_s_stored"] = stored.get("mae", np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def report(df):
    print(f"configurations scored: {len(df)}\n")

    print("== recomputed against the stored results.json ==")
    for recomputed, stored, label in (
        ("ap_pct", "ap_pct_stored", "stalling AP [%]"),
        ("f1_pct", "f1_pct_stored", "stalling F1 [%]"),
        ("r2_pct", "r2_pct_stored", "buffer-health R2 [%], clipped"),
        ("mae_s", "mae_s_stored", "buffer-health MAE [s], clipped"),
    ):
        if recomputed in df:
            diff = np.nanmax(np.abs(df[recomputed] - df[stored]))
            flag = "" if diff < 1e-4 else "   <-- UNEXPECTED, investigate"
            print(f"  {label:<32} largest absolute difference {diff:.3g}{flag}")
    if "r2_pct_raw" in df:
        bh = df[df.kqi == "buffer_health"]
        gap = np.nanmax(np.abs(bh.r2_pct_raw - bh.r2_pct_stored))
        print(
            f"\n  Clipping matters: scoring the arrays without clipping to [0, 1] would"
            f" disagree with\n  the paper by up to {gap:.3f} pp of R2. Up to"
            f" {bh.outside_range_pct.max():.1f} % of a configuration's predictions lie"
            f" outside\n  the valid range. Always clip before scoring buffer health."
        )

    print("\n== test population sizes ==")
    print(f"  range {df.samples.min()} to {df.samples.max()} samples")
    for kqi in df.kqi.unique():
        sub = df[df.kqi == kqi]
        print(
            f"  {kqi:<14} multiples of 128: {sub.multiple_of_128.sum()}/{len(sub)}"
            f" (the remainder are the tree models, which keep every sample)"
        )

    for kqi, columns in (
        ("stalling", ["configuration", "samples", "ap_pct", "f1_pct"]),
        ("buffer_health", ["configuration", "samples", "r2_pct", "mae_s"]),
    ):
        sub = df[(df.kqi == kqi) & df.configuration.str.contains("_all_")]
        print(f"\n== ALL view, {kqi} ==")
        print(sub[columns].round(2).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--predictions",
        required=True,
        help="directory containing stalling_predictions/ and buffer_health_predictions/",
    )
    ap.add_argument("--out", help="write the per-configuration scores to this CSV")
    args = ap.parse_args()

    df = collect(args.predictions)
    report(df)
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
