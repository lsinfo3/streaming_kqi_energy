#!/usr/bin/env python3
"""Reconstruct the 95 selected configurations from the stored Optuna databases.

For each of the 48 searches (2 KQIs x 4 traffic views x 6 model families) this
re-runs the selection that
``master_thesis/src/config_controller.py:model_training`` (lines 483-540)
performed when it generated the retraining configs:

  score  = trial_user_attributes['ap'] (stalling) or ['r2'] (buffer health)
  energy = trial_values.value for objective 1 (the normalised energy objective)
  performance-first = argmax(score) over all completed trials
  energy-aware      = the Pareto-optimal trial with the largest
                      (score_norm - energy_norm) after min-max scaling both
                      objectives over the Pareto-optimal trials

Writes ``artifact/data/features/configurations.csv``.

The databases are opened read-only.  Nothing outside the output file is written.

Usage:  python3 reconstruct_configurations.py [db_dir] [importances_dir] [out_csv]
"""

import csv
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]            # the artifact root
EXT = Path("data_masters_thesis")   # the external source tree, if available


def _pick(in_repo, external):
    """Prefer the copy shipped in this repository, fall back to the archive."""
    return in_repo if in_repo.exists() else external


DB_DIR = _pick(REPO / "data" / "hpo", EXT / "measurements" / "model_hpo")
IMP_DIR = _pick(REPO / "data" / "importances", EXT / "general_data" / "importances")
# The full-test prediction arrays are 325 MB and are NOT shipped in git; they come
# from the release asset.  Without them the `test_population` column stays empty.
PRED_DIR = _pick(REPO / "data" / "predictions", EXT / "general_data")

# study_id 1..4 in every database, in this order (config_controller.py:495)
VIEW_CODES = ["all", "uplink_only", "gt_100", "uplink_only_gt_100"]
VIEW_PAPER = {"all": "ALL", "gt_100": "ALL>100B",
              "uplink_only": "UL", "uplink_only_gt_100": "UL>100B"}
FAMILIES = ["rf", "lgbm", "ffnn", "cnn", "lstm", "transformer"]
KINDS = ["stalling", "buffer_health"]
ROLE_DIR = {"performance-first": "performance", "energy-aware": "efficiency"}

BASE_PER_WINDOW = {"all": 48, "gt_100": 24, "uplink_only": 24,
                   "uplink_only_gt_100": 12}
N_WINDOWS = {"up_to_5s": 3, "up_to_10s": 4, "up_to_20s": 5, "up_to_30s": 6}

HP_COLUMNS = ["n_estimators", "max_depth_exponent", "max_leaves_exponent",
              "min_samples_leaf", "max_features", "num_leafs_exponent",
              "learning_rate", "batch_exponent", "dropout", "loss_weight",
              "n_layers", "layer_size_exponent", "n_kernels_exponent",
              "kernel_size", "hidden_size_exponent", "n_head_exponent",
              "dim_model_exponent", "dim_feedforward_exponent",
              "sequence_window_size"]


def pareto_front(score, energy):
    """Verbatim from config_controller.py:486-488."""
    is_dominated = np.any((score[:, None] <= score) & (energy[:, None] >= energy)
                          & ((score[:, None] < score) | (energy[:, None] > energy)),
                          axis=1)
    return np.where(~is_dominated)[0]


def n_selected_features(view, window_set, kind, imp_dir):
    """Size of the surviving feature set, from the published permutation-importance
    files -- the same files model_training.py:37-39 and :117-119 read."""
    task = "classifier" if kind == "stalling" else "regressor"
    path = imp_dir / (f"permutation_importance_{view}_packets_"
                      f"{window_set}_windows_{task}.json")
    with path.open() as fh:
        return len(json.load(fh)["selected_features"])


def decode_params(conn, trial_id):
    params = {}
    for name, value, dist_json in conn.execute(
            "SELECT param_name, param_value, distribution_json "
            "FROM trial_params WHERE trial_id = ?", (trial_id,)):
        dist = json.loads(dist_json)
        if dist["name"] == "CategoricalDistribution":
            params[name] = dist["attributes"]["choices"][int(value)]
        elif "Int" in dist["name"]:
            params[name] = int(value)
        else:
            params[name] = value
    return params


def test_population(kind, view, family, role, pred_dir):
    """Length of the stored full-test prediction array for this configuration."""
    name = f"{kind}_{view}_{family}_{ROLE_DIR[role]}"
    path = pred_dir / f"{kind}_predictions" / f"{name}_predictions.npy"
    if not path.exists():
        return ""
    return len(np.load(path, mmap_mode="r"))


def reconstruct(db_dir=DB_DIR, imp_dir=IMP_DIR, pred_dir=PRED_DIR):
    rows = []
    for kind in KINDS:
        score_key = "ap" if kind == "stalling" else "r2"
        for family in FAMILIES:
            db = db_dir / f"{kind}_{family}_optuna_study.db"
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            for study_id, view in enumerate(VIEW_CODES, start=1):
                q = conn.execute(f"""
                    SELECT tua.value_json, tv.value, t.trial_id, t.number
                    FROM trials t
                    JOIN trial_user_attributes tua
                      ON tua.trial_id = t.trial_id AND tua.key = '{score_key}'
                    JOIN trial_values tv
                      ON tv.trial_id = t.trial_id AND tv.objective = 1
                    WHERE t.study_id = {study_id}
                    ORDER BY t.trial_id""").fetchall()
                arr = np.array([[float(a), float(b), float(c), float(d)]
                                for a, b, c, d in q])
                score, energy, ids, numbers = arr.T
                idx_pf = int(np.argmax(score))
                front = pareto_front(score, energy)
                s, e = score[front], energy[front]
                s_norm = (s - s.min()) / (s.max() - s.min())
                e_norm = (e - e.min()) / (e.max() - e.min())
                idx_ea = int(front[int(np.argmax(s_norm - e_norm))])

                n_trials = conn.execute(
                    "SELECT count(*) FROM trials WHERE study_id = ?",
                    (study_id,)).fetchone()[0]
                states = dict(conn.execute(
                    "SELECT state, count(*) FROM trials WHERE study_id = ? "
                    "GROUP BY state", (study_id,)).fetchall())
                conv = conn.execute(
                    "SELECT value_json FROM study_user_attributes "
                    "WHERE study_id = ? AND key = 'converged_at_trial'",
                    (study_id,)).fetchone()

                for role, idx in (("performance-first", idx_pf),
                                  ("energy-aware", idx_ea)):
                    trial_id = int(ids[idx])
                    params = decode_params(conn, trial_id)
                    dataset = params["dataset"]
                    window_set = params.get("feature_window_size", "")
                    if dataset == "multiple_windows":
                        n_features = BASE_PER_WINDOW[view] * N_WINDOWS[window_set]
                        fs = False
                    elif dataset == "multiple_windows_fs":
                        n_features = n_selected_features(view, window_set, kind,
                                                         imp_dir)
                        fs = True
                    else:                     # dl_0.5s / dl_1s: one window only
                        n_features = BASE_PER_WINDOW[view]
                        fs = False
                        window_set = ""
                    row = {
                        "kqi": kind,
                        "view": VIEW_PAPER[view],
                        "view_code": view,
                        "family": family,
                        "role": role,
                        "config_name": f"{kind}_{view}_{family}_{ROLE_DIR[role]}",
                        "trial_id": trial_id,
                        "trial_number": int(numbers[idx]),
                        "score": score[idx],
                        "energy_objective": energy[idx],
                        "pareto_optimal": int(idx in set(front.tolist())),
                        "same_trial_both_roles": "",
                        "dataset": dataset,
                        "feature_window_size": window_set,
                        "feature_selection": int(fs),
                        "n_features": n_features,
                        "test_population": test_population(kind, view, family,
                                                           role, pred_dir),
                        "n_trials": n_trials,
                        "n_complete": states.get("COMPLETE", 0),
                        "n_pruned": states.get("PRUNED", 0),
                        "n_failed": states.get("FAIL", 0),
                        "n_pareto": len(front),
                        "converged_at_trial": int(conv[0]) if conv else "",
                    }
                    for c in HP_COLUMNS:
                        row[c] = params.get(c, "")
                    rows.append(row)
    # mark the search(es) where both criteria pick the same trial
    by_search = {}
    for r in rows:
        by_search.setdefault((r["kqi"], r["view"], r["family"]), []).append(r)
    for pair in by_search.values():
        if pair[0]["trial_id"] == pair[1]["trial_id"]:
            for r in pair:
                r["same_trial_both_roles"] = 1
        else:
            for r in pair:
                r["same_trial_both_roles"] = 0
    return rows


def main():
    db_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DB_DIR
    imp_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else IMP_DIR
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else (
        Path(__file__).resolve().parents[2] / "data" / "features"
        / "configurations.csv")
    rows = reconstruct(db_dir, imp_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- the four checks reported in CONFIGURATIONS.md
    n_ea = sum(1 for r in rows if r["role"] == "energy-aware")
    dup = sorted({(r["kqi"], r["family"], r["view"]) for r in rows
                  if r["same_trial_both_roles"] == 1})
    distinct = len(rows) - len(dup)
    limit = sorted({(r["kqi"], r["family"], r["view"], r["n_trials"]) for r in rows
                    if r["converged_at_trial"] == "" or r["converged_at_trial"] >= 200})
    conv = sorted({(r["kqi"], r["family"], r["view"], r["n_trials"]) for r in rows
                   if r["converged_at_trial"] != "" and r["converged_at_trial"] < 200})
    print(f"searches                      : {len(rows) // 2}")
    print(f"energy-aware configurations   : {n_ea}")
    print(f"searches with pf == ea        : {dup}")
    print(f"distinct configurations       : {distinct}")
    print(f"searches at the 200 limit     : {limit}")
    print(f"converged searches            : {len(conv)}, "
          f"{min(c[3] for c in conv)}-{max(c[3] for c in conv)} trials")
    print(f"total trials                  : {sum(r['n_trials'] for r in rows) // 2}")
    print(f"complete trials               : {sum(r['n_complete'] for r in rows) // 2}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
