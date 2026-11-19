#!/usr/bin/env python3
"""Export the surviving feature sets of the two-stage feature selection.

Reads the 32 published permutation-importance files in
``data_masters_thesis/general_data/importances/``.  Each file holds

  ``features``          the set that survived the mutual-information filter
                        (the input of the permutation-importance stage),
  ``importances_mean``  / ``importances_std``  the 30-repeat statistics,
  ``selected_features`` the final surviving set, ranked by the normalised lower
                        bound of the 95 % confidence interval.

The full per-view, per-max-window feature count is recomputed from the
generator in ``make_feature_names.py`` rather than taken from the files, so the
removal percentages are traceable to both sources.

Writes two files into ``artifact/data/features/``:
  ``feature_selection_summary.csv``  one row per (task, view, window set)
  ``feature_selection_sets.csv``     one row per surviving feature, in rank order

Usage:  python3 export_selected_features.py [importances_dir] [out_dir]
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_feature_names import (AGGREGATION_FUNCTIONS, METRICS, VIEWS,  # noqa: E402
                                WINDOW_SETS, feature_names_from_config)

_REPO_IMP = Path(__file__).resolve().parents[2] / "data" / "importances"
IMP_DIR = _REPO_IMP if _REPO_IMP.exists() else Path(
    "data_masters_thesis/general_data/importances")
TASKS = {"classifier": "stalling", "regressor": "buffer_health"}


def full_count(view, window_set):
    _, uplink_only, gt_100 = VIEWS[view]
    return len(feature_names_from_config(
        {"uplink_only": uplink_only, "gt_100": gt_100,
         "aggregation_functions": AGGREGATION_FUNCTIONS, "metrics": METRICS},
        WINDOW_SETS[window_set]))


def main():
    imp_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else IMP_DIR
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else (
        Path(__file__).resolve().parents[2] / "data" / "features")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary, sets = [], []
    for task, kqi in TASKS.items():
        for view in VIEWS:
            for window_set in WINDOW_SETS:
                path = imp_dir / (f"permutation_importance_{view}_packets_"
                                  f"{window_set}_windows_{task}.json")
                with path.open() as fh:
                    d = json.load(fh)
                n_full = full_count(view, window_set)
                n_mi, n_pi = len(d["features"]), len(d["selected_features"])
                imp = dict(zip(d["features"], d["importances_mean"]))
                std = dict(zip(d["features"], d["importances_std"]))
                summary.append({
                    "kqi": kqi, "task": task, "view": VIEWS[view][0],
                    "view_code": view, "window_set": window_set,
                    "n_full": n_full, "n_after_mi": n_mi, "n_after_pi": n_pi,
                    "removed_mi_pct": round(100 * (n_full - n_mi) / n_full, 2),
                    "removed_pi_pct": round(100 * (n_mi - n_pi) / n_mi, 2),
                    "removed_total_pct": round(100 * (n_full - n_pi) / n_full, 2),
                    "source_file": path.name,
                })
                for rank, name in enumerate(d["selected_features"], start=1):
                    sets.append({
                        "kqi": kqi, "view": VIEWS[view][0], "view_code": view,
                        "window_set": window_set, "rank": rank, "name": name,
                        "importance_mean": imp[name], "importance_std": std[name],
                    })

    for rows, name in ((summary, "feature_selection_summary.csv"),
                       (sets, "feature_selection_sets.csv")):
        with (out_dir / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out_dir / name} ({len(rows)} rows)")

    for kqi in ("stalling", "buffer_health"):
        r = [s["removed_total_pct"] for s in summary if s["kqi"] == kqi]
        print(f"{kqi:14s} total removal {min(r):.1f}-{max(r):.1f} %")


if __name__ == "__main__":
    main()
