#!/usr/bin/env python3
"""Generate the canonical ordered feature-name lists for the four traffic views.

The two functions below are a verbatim port of the generators in the thesis
pipeline:

  master_thesis/src/processing/preprocessing.py
      feature_names_from_config      (lines 46-64)
      feature_indices_per_scenario   (lines 67-100)

They are copied rather than imported because ``preprocessing.py`` imports
``paths.py``, which hard-codes absolute paths of the machine the thesis was run
on.  An identical copy of ``feature_names_from_config`` also exists in
``master_thesis/src/feature_selection.py`` (lines 37-55); the two agree
character for character except for the ``missing_indicator`` argument that only
the ``preprocessing.py`` copy of ``feature_indices_per_scenario`` has.

Output: ``artifact/data/features/feature_names_<view-code>.csv`` with columns
``index,name``, where ``index`` is the position of the feature in the canonical
288-element ALL ordering (the column index into the stored
``dataset_n288_i1.0_w1-3-5-10-20-30_ffill.npy`` matrix) and the rows are in the
order in which the pipeline passes them to the model.

Usage:  python3 make_feature_names.py [output_dir]
"""

import csv
import sys
from pathlib import Path

# The configuration used by every stage of the pipeline.  Taken from
# feature_indices_per_scenario (preprocessing.py:68-83) and from
# build_features_from_config (preprocessing.py:306-321).
AGGREGATION_FUNCTIONS = ["mean", "median", "min", "max", "std"]
METRICS = ["iat", "packet_size", "packet_count", "total_bytes"]
WINDOWS = [1, 3, 5, 10, 20, 30]

# Paper name -> code name, and the (uplink_only, gt_100) flag pair each implies.
VIEWS = {
    "all": ("ALL", False, False),
    "gt_100": ("ALL>100B", False, True),
    "uplink_only": ("UL", True, False),
    "uplink_only_gt_100": ("UL>100B", True, True),
}

# The window sets a search could select, from the ``feature_window_size``
# categorical of the Optuna studies and from ``window_lengths`` in
# master_thesis/src/models/model_training.py:18-23.
WINDOW_SETS = {
    "up_to_5s": [1, 3, 5],
    "up_to_10s": [1, 3, 5, 10],
    "up_to_20s": [1, 3, 5, 10, 20],
    "up_to_30s": [1, 3, 5, 10, 20, 30],
}


def feature_names_from_config(config: dict, windows=(False,)):
    """Port of preprocessing.py:46-64 (unchanged logic)."""
    names = []
    for t in windows:
        window = f"{t}s_" if t else ""
        for i in range(1 if config["uplink_only"] else 2):
            prefix = f"{window}dl_" if i else f"{window}ul_"
            for j in range(1 if config["gt_100"] else 2):
                suffix = "" if j else "_gt_100"
                for metric in config["metrics"]:
                    if metric == "iat":
                        names += [f"{prefix}{agg}_iat{suffix}"
                                  for agg in config["aggregation_functions"]]
                    elif metric == "packet_size":
                        names += [f"{prefix}{agg}_size{suffix}"
                                  for agg in config["aggregation_functions"]]
                    elif metric == "packet_count":
                        names.append(f"{prefix}packet_count{suffix}")
                    elif metric == "total_bytes":
                        names.append(f"{prefix}total_bytes{suffix}")
    return names


def all_feature_names():
    """The canonical 288-element ALL ordering the stored matrix columns follow."""
    return feature_names_from_config(
        {"uplink_only": False, "gt_100": False,
         "aggregation_functions": AGGREGATION_FUNCTIONS, "metrics": METRICS},
        WINDOWS)


def feature_indices_per_scenario(scenario="all", windows=WINDOWS, feature_names=False):
    """Port of preprocessing.py:67-100, without the missing_indicator branch.

    Returns ``(indices, selected_features)``.  ``indices`` ends with
    ``len(all_features)`` -- the trailing sample-id column of the stored
    matrix, which ``get_dataset`` overwrites with the target.
    """
    _, uplink_only, gt_100 = VIEWS[scenario]
    config = {"uplink_only": uplink_only, "gt_100": gt_100,
              "aggregation_functions": AGGREGATION_FUNCTIONS, "metrics": METRICS}
    all_features = all_feature_names()
    selected = feature_names or feature_names_from_config(config, windows)
    indices = sorted(all_features.index(f) for f in selected)
    return indices + [len(all_features)], selected


def main(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_features = all_feature_names()
    assert len(all_features) == 288, len(all_features)
    assert len(set(all_features)) == 288

    for code, (paper, _, _) in VIEWS.items():
        indices, names = feature_indices_per_scenario(code)
        indices = indices[:-1]                      # drop the id/target column
        # The sorted index order and the generator order must agree, otherwise
        # the published list would not match the matrix the models were fed.
        assert [all_features[i] for i in indices] == names, code
        path = out_dir / f"feature_names_{code}.csv"
        with path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["index", "name"])
            w.writerows(zip(indices, names))
        per_window = len(names) // len(WINDOWS)
        print(f"{paper:9s} ({code:18s}) {per_window:3d} per window  "
              f"{len(names):3d} total  -> {path.name}")
        for set_name, wins in WINDOW_SETS.items():
            n = len(feature_names_from_config(
                {"uplink_only": VIEWS[code][1], "gt_100": VIEWS[code][2],
                 "aggregation_functions": AGGREGATION_FUNCTIONS,
                 "metrics": METRICS}, wins))
            print(f"{'':32s} {set_name:10s} {n:3d}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else Path(__file__).resolve().parents[2] / "data" / "features")
