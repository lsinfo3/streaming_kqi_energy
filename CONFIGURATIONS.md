# Configurations

Every configuration evaluated in *Energy-Aware Assessment and Energy Modeling for Encrypted
Video KQI Prediction at the Edge*, reconstructed from the stored Optuna
search databases. This answers "what exactly did you run" and makes the energy-aware knee
selection checkable.

Machine-readable form: **`data/features/configurations.csv`** (96 rows, 42 columns).
Regenerate with `code/features/reconstruct_configurations.py`, which opens every database
read-only and prefers the copies shipped in `data/hpo/` and `data/importances/`. The only
column it cannot fill from the repository alone is `test_population`, which needs the
full-test prediction arrays from the release asset; the committed CSV has it filled. The
feature sets are described in `FEATURES.md`.

**Sources.**

| Short name | Path |
|---|---|
| `model_hpo/` | `data_masters_thesis/measurements/model_hpo/*.db` — 12 SQLite databases, 4 studies each; shipped in this repository as `data/hpo/` |
| `importances/` | `data_masters_thesis/general_data/importances/`; shipped as `data/importances/` |
| `selected_models/` | `data_masters_thesis/general_data/selected_models/{raspberry_pi,jetson_nano_orin}/*.csv`; shipped as `data/selected_models/` |
| `*_predictions/` | `data_masters_thesis/general_data/{stalling,buffer_health}_predictions/` — 325 MB, **not** shipped in git |
| `config_controller.py` | `master_thesis/src/config_controller.py` |
| `tree_models.py` | `master_thesis/src/models/tree_models.py` |
| `tree_modes_process.py` | `master_thesis/src/models/tree_modes_process.py` |
| `neural_models.py` | `master_thesis/src/models/neural_models.py` |
| `model_training.py` | `master_thesis/src/models/model_training.py` |
| `machine_learning_helpers.py` | `master_thesis/src/models/machine_learning_helpers.py` |

---

## 1. 48 searches, 96 role entries, 95 distinct configurations

Two KQIs × four traffic views × six model families = **48 model-selection problems**
("searches"). Each search contributes two selections, *performance-first* (pf) and
*energy-aware* (ea), so there are **96 role entries**.

In exactly one search — **buffer health, FFNN, ALL>100B** — both criteria pick the same
Optuna trial (trial 25). The number of **distinct hyperparameter configurations is therefore
95**, and that is the figure the paper uses. The coinciding trial was nevertheless retrained
twice, once for each role, and the two retrainings stopped early at 31 and 100 epochs and gave
different full-test R² values of 80.7 % and 84.2 % (`methodology.tex:57`; recomputed here
from `buffer_health_gt_100_ffnn_{performance,efficiency}_results.json` as 80.68 % and
84.19 %). Every downstream artefact — exported ONNX model, W1 energy measurement, stored
prediction array — therefore exists 96 times.

**So: 96 is the right count for a per-role listing, 95 for distinct hyperparameter
settings.** `configurations.csv` has 96 rows; the column `same_trial_both_roles` is 1 on the
two rows of the coinciding search and 0 elsewhere.

One database holds the six families' searches for one KQI per file (`{kqi}_{family}_optuna_study.db`),
and each file holds four studies, `study_id` 1 = ALL, 2 = UL, 3 = ALL>100B, 4 = UL>100B
(`config_controller.py:495`). Note that this order is *not* the order used in the paper's
tables.

---

## 2. How a configuration was selected

### 2.1 The two objectives

Each trial returns a pair (`tree_models.py:66-69`, `neural_models.py:206`):

```python
# trees
return max(score, 0), min(energy / 15_000, 1)      # energy = inf_ws_cpu_energy  (RAPL, J)
# neural networks
return max(0, score), min(energy / 2500, 1)        # energy = inf_ws_gpu_energy  (NVML, J)
```

`score` is the raw average precision (stalling) or R² (buffer health) on the **full**
validation split, stored unclipped as the trial user attribute `ap` or `r2`. Objective 0 in
`trial_values` is that value after the `max(., 0)` clip; objective 1 is the normalised energy.
The normalising bounds of 15 kJ (trees) and 2.5 kJ (neural) and the clip at 1 are exactly what
the paper states. The columns `score` and `energy_objective` of `configurations.csv` hold the
`ap`/`r2` user attribute and objective 1 respectively.

One configuration sits on the clip: buffer health, LSTM, ALL>100B, pf has
`energy_objective = 1.0000`, so its true validation energy was at or above 2.5 kJ and is not
recoverable from the objective alone (it is recoverable from the trial's
`inf_ws_gpu_energy` user attribute).

### 2.2 Search procedure

`optuna.samplers.TPESampler(seed=1234, multivariate=True)` with
`directions=["maximize", "minimize"]` — Optuna's multi-objective TPE, i.e. MOTPE
(`tree_models.py:140-147`, `neural_models.py:220-228`).

Stopping is `HypervolumeConvergenceCallback(reference_point=[0, 1], patience=30,
min_delta=0.001, min_trails=50, max_trails=200)` (`machine_learning_helpers.py:182-239`),
i.e. stop when the Pareto-front hypervolume has improved by less than 0.1 % for 30 consecutive
trials, with a floor of 50 and a ceiling of 200 trials. `study.optimize(..., n_trials=200)`
imposes a second ceiling.

Trial budget per candidate:

| | Trees (RF, LGBM) | Neural (FFNN, CNN, LSTM, TF) |
|---|---|---|
| Training data | a random 20 % draw of the training split, first half used for fitting (so 10 %), second half as the LGBM early-stopping set (`tree_modes_process.py:34-36`) | a random 10 % draw of the training windows (`neural_models.py:138`) |
| Scoring data | full validation split (`tree_modes_process.py:38`) | full validation split (`neural_models.py:139`) |
| Epochs | — | at most 20, with early stopping (`neural_models.py:188`) |
| Inference repetitions for the energy objective | 20 (`tree_modes_process.py:62`) | 5 (`neural_models.py:193`) |
| Time budget | 360 s per trial (`tree_models.py:48`) | 300 s training, 120 s inference (`neural_models.py:188-195`) |

All of this matches the paper's §III-C except the tree subset, where the paper says "a random
10 % subset of the training data": the code draws 20 % and uses half of it for fitting, so 10 %
is correct for the fit and a further 10 % is used for LightGBM early stopping.

### 2.3 The two selection criteria

The selection is `config_controller.py:483-540`, reproduced verbatim in
`code/features/reconstruct_configurations.py`:

```python
def pareto_front(score, energy):
    is_dominated = np.any((score[:, None] <= score) & (energy[:, None] >= energy)
                          & ((score[:, None] < score) | (energy[:, None] > energy)), axis=1)
    return np.where(~is_dominated)[0]

idx_score = np.argmax(score)                       # performance-first
pareto_indices = pareto_front(score, energy)
score, energy = score[pareto_indices], energy[pareto_indices]
score_norm = (score - score.min()) / (score.max() - score.min())
energy_norm = (energy - energy.min()) / (energy.max() - energy.min())
idx_combined = pareto_indices[np.argmax(score_norm - energy_norm)]   # energy-aware (knee)
```

* **performance-first** = the completed trial with the highest validation score, over all
  completed trials (not only the Pareto-optimal ones). In all 48 searches the arg-max trial
  turns out to be Pareto-optimal anyway (`pareto_optimal` column, verified below).
* **energy-aware** = the Pareto-optimal trial maximising `score_norm - energy_norm`.

`argmax(score_norm - energy_norm)` is identical to the paper's description, "the trial with
the largest distance above the diagonal connecting the lowest-energy and highest-scoring
trials in the scaled objective space". On a Pareto front the lowest-energy trial is also the
lowest-scoring one and the highest-scoring trial is also the highest-energy one, so after
min-max scaling the two endpoints are exactly (0, 0) and (1, 1); the diagonal is
`energy_norm = score_norm` and the signed distance above it is
`(score_norm - energy_norm)/sqrt(2)`. The arg-max is unchanged by the constant factor.

Only completed trials enter: the SQL joins `trial_user_attributes` on the `ap`/`r2` key and
`trial_values` on objective 1, and pruned or failed trials have neither.

---

## 4. Per-search summary

`Trials`, `compl.`, `pruned`, `failed` are row counts in `trials` for that `study_id`.
`Stop` is `converged` if the study carries a `converged_at_trial` attribute below 200 and
**limit** otherwise. `Pareto` is the number of Pareto-optimal completed trials.
`pf trial` / `ea trial` are zero-based `trials.number` values (`configurations.csv` also
carries the primary key `trial_id`). `score` is `ap` or `r2` on the validation split;
`E` is the normalised energy objective.

| KQI | Family | View | Trials | compl. | pruned | failed | Stop | Pareto | pf trial | ea trial | pf score | ea score | pf E | ea E |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S | RF | ALL | 137 | 133 | 4 | 0 | converged | 13 | 108 | 81 | 0.8551 | 0.8310 | 0.0776 | 0.0064 |
| S | RF | ALL>100B | 79 | 79 | 0 | 0 | converged | 12 | 38 | 64 | 0.8436 | 0.8425 | 0.0713 | 0.0142 |
| S | RF | UL | 110 | 109 | 1 | 0 | converged | 15 | 79 | 91 | 0.8448 | 0.8356 | 0.0576 | 0.0101 |
| S | RF | UL>100B | 77 | 76 | 1 | 0 | converged | 16 | 46 | 70 | 0.7705 | 0.7539 | 0.0836 | 0.0076 |
| S | LGBM | ALL | 119 | 107 | 5 | 7 | converged | 10 | 83 | 80 | 0.8688 | 0.8375 | 0.1629 | 0.0146 |
| S | LGBM | ALL>100B | 132 | 124 | 3 | 5 | converged | 13 | 79 | 98 | 0.8644 | 0.8381 | 0.1216 | 0.0101 |
| S | LGBM | UL | 152 | 144 | 2 | 6 | converged | 16 | 120 | 55 | 0.8534 | 0.8468 | 0.2217 | 0.0519 |
| S | LGBM | UL>100B | 133 | 120 | 0 | 13 | converged | 15 | 98 | 130 | 0.7938 | 0.7685 | 0.1192 | 0.0181 |
| S | FFNN | ALL | 53 | 53 | 0 | 0 | converged | 6 | 10 | 23 | 0.6969 | 0.6816 | 0.1028 | 0.0748 |
| S | FFNN | ALL>100B | 76 | 76 | 0 | 0 | converged | 6 | 45 | 36 | 0.6926 | 0.6543 | 0.0755 | 0.0713 |
| S | FFNN | UL | 72 | 72 | 0 | 0 | converged | 5 | 41 | 70 | 0.6955 | 0.6449 | 0.0713 | 0.0699 |
| S | FFNN | UL>100B | 136 | 136 | 0 | 0 | converged | 9 | 71 | 101 | 0.5625 | 0.4998 | 0.0961 | 0.0678 |
| S | CNN | ALL | 109 | 109 | 0 | 0 | converged | 12 | 77 | 25 | 0.8010 | 0.7599 | 0.2535 | 0.1103 |
| S | CNN | ALL>100B | 78 | 78 | 0 | 0 | converged | 11 | 5 | 12 | 0.8037 | 0.7250 | 0.2994 | 0.1043 |
| S | CNN | UL | 135 | 134 | 1 | 0 | converged | 17 | 100 | 29 | 0.7982 | 0.7160 | 0.1813 | 0.1011 |
| S | CNN | UL>100B | 118 | 118 | 0 | 0 | converged | 15 | 86 | 42 | 0.7233 | 0.6859 | 0.1783 | 0.1282 |
| S | LSTM | ALL | 112 | 110 | 1 | 1 | converged | 10 | 60 | 74 | 0.8330 | 0.8028 | 0.4854 | 0.1328 |
| S | LSTM | ALL>100B | 154 | 147 | 7 | 0 | converged | 15 | 39 | 69 | 0.8467 | 0.7942 | 0.3499 | 0.1099 |
| S | LSTM | UL | 53 | 51 | 2 | 0 | converged | 12 | 6 | 14 | 0.8267 | 0.8043 | 0.5570 | 0.1749 |
| S | LSTM | UL>100B | 102 | 102 | 0 | 0 | converged | 12 | 39 | 83 | 0.7772 | 0.6108 | 0.1709 | 0.0938 |
| S | TF | ALL | 119 | 108 | 9 | 2 | converged | 10 | 81 | 65 | 0.8014 | 0.7541 | 0.3954 | 0.1050 |
| S | TF | ALL>100B | 180 | 165 | 15 | 0 | converged | 5 | 142 | 99 | 0.7935 | 0.7767 | 0.1832 | 0.1210 |
| S | TF | UL | 168 | 163 | 5 | 0 | converged | 9 | 54 | 104 | 0.7973 | 0.7680 | 0.2911 | 0.0969 |
| S | TF | UL>100B | 115 | 112 | 3 | 0 | converged | 12 | 82 | 55 | 0.7428 | 0.6609 | 0.2441 | 0.1201 |
| BH | RF | ALL | 110 | 109 | 1 | 0 | converged | 10 | 79 | 69 | 0.8420 | 0.8242 | 0.0822 | 0.0094 |
| BH | RF | ALL>100B | 149 | 148 | 1 | 0 | converged | 8 | 118 | 141 | 0.8223 | 0.8121 | 0.0130 | 0.0046 |
| BH | RF | UL | 73 | 71 | 2 | 0 | converged | 8 | 23 | 68 | 0.8135 | 0.8046 | 0.0516 | 0.0105 |
| BH | RF | UL>100B | 100 | 99 | 1 | 0 | converged | 10 | 93 | 66 | 0.7291 | 0.7248 | 0.0719 | 0.0075 |
| BH | LGBM | ALL | 177 | 148 | 29 | 0 | converged | 24 | 96 | 72 | 0.8887 | 0.8547 | 0.4901 | 0.0402 |
| BH | LGBM | ALL>100B | 200 | 168 | 32 | 0 | **limit** | 21 | 179 | 180 | 0.8830 | 0.8427 | 0.5207 | 0.0251 |
| BH | LGBM | UL | 155 | 125 | 30 | 0 | converged | 20 | 119 | 140 | 0.8696 | 0.8370 | 0.8311 | 0.0281 |
| BH | LGBM | UL>100B | 186 | 159 | 27 | 0 | converged | 27 | 150 | 145 | 0.7835 | 0.7190 | 0.2592 | 0.0093 |
| BH | FFNN | ALL | 129 | 129 | 0 | 0 | converged | 13 | 75 | 122 | 0.7849 | 0.7571 | 0.1822 | 0.1666 |
| BH | FFNN | ALL>100B | 78 | 78 | 0 | 0 | converged | 6 | 25 | 25 | 0.7671 | 0.7671 | 0.1651 | 0.1651 |
| BH | FFNN | UL | 112 | 112 | 0 | 0 | converged | 10 | 41 | 81 | 0.7606 | 0.6589 | 0.1732 | 0.1591 |
| BH | FFNN | UL>100B | 116 | 116 | 0 | 0 | converged | 10 | 86 | 76 | 0.6439 | 0.6111 | 0.1710 | 0.1581 |
| BH | CNN | ALL | 149 | 129 | 20 | 0 | converged | 8 | 61 | 112 | 0.8697 | 0.8506 | 0.5007 | 0.2137 |
| BH | CNN | ALL>100B | 149 | 143 | 6 | 0 | converged | 13 | 116 | 135 | 0.8846 | 0.8775 | 0.4321 | 0.2233 |
| BH | CNN | UL | 78 | 73 | 5 | 0 | converged | 9 | 42 | 4 | 0.8662 | 0.5181 | 0.2212 | 0.1731 |
| BH | CNN | UL>100B | 72 | 71 | 1 | 0 | converged | 14 | 6 | 41 | 0.7633 | 0.5338 | 0.2443 | 0.1725 |
| BH | LSTM | ALL | 156 | 129 | 27 | 0 | converged | 13 | 117 | 101 | 0.9084 | 0.9002 | 0.4792 | 0.1951 |
| BH | LSTM | ALL>100B | 92 | 82 | 10 | 0 | converged | 13 | 36 | 31 | 0.9208 | 0.8744 | 1.0000 | 0.2082 |
| BH | LSTM | UL | 201 | 171 | 29 | 1 | **limit** | 11 | 152 | 73 | 0.9025 | 0.8744 | 0.2988 | 0.1866 |
| BH | LSTM | UL>100B | 84 | 74 | 10 | 0 | converged | 9 | 52 | 11 | 0.8500 | 0.7688 | 0.2387 | 0.1714 |
| BH | TF | ALL | 201 | 132 | 69 | 0 | **limit** | 15 | 194 | 152 | 0.8752 | 0.8052 | 0.4315 | 0.2402 |
| BH | TF | ALL>100B | 169 | 115 | 54 | 0 | converged | 13 | 39 | 114 | 0.8850 | 0.8270 | 0.4981 | 0.2346 |
| BH | TF | UL | 190 | 136 | 54 | 0 | converged | 12 | 137 | 63 | 0.8620 | 0.8088 | 0.2939 | 0.2329 |
| BH | TF | UL>100B | 165 | 114 | 51 | 0 | converged | 10 | 84 | 122 | 0.8363 | 0.7781 | 0.8986 | 0.2486 |

---

## 5. Selected configurations per family

`Trial` is the zero-based trial number within its study. `Score` is validation AP (S) or R²
(BH); `E_norm` is the normalised energy objective. The feature-set column gives the `dataset`
variant, the selected window set, and the resulting input width; `--` means the parameter does
not apply. Exponent parameters are given as the sampled exponent *x*, so the effective value
is 2^*x* (see §6.1). `seq. len` is the history length in seconds. `pos. weight` is the
positive-class weight of the loss, sampled for stalling only.

### Random forest (RF)

| KQI | View | Sel. | Trial | Score | E_norm | Feature set (variant / windows / count) | trees | max depth 2^x | max leaves 2^x | min samples/leaf | max features |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S | ALL | pf | 108 | 0.8551 | 0.0776 | multiple_windows_fs / up_to_30s / 231 | 180 | 5 | 12 | 6 | 0.190 |
| S | ALL | ea | 81 | 0.8310 | 0.0064 | multiple_windows / up_to_30s / 288 | 10 | 4 | 11 | 8 | 0.408 |
| S | ALL>100B | pf | 38 | 0.8436 | 0.0713 | multiple_windows / up_to_30s / 144 | 250 | 6 | 10 | 14 | 0.708 |
| S | ALL>100B | ea | 64 | 0.8425 | 0.0142 | multiple_windows / up_to_30s / 144 | 40 | 5 | 12 | 13 | 0.267 |
| S | UL | pf | 79 | 0.8448 | 0.0576 | multiple_windows_fs / up_to_30s / 115 | 190 | 5 | 12 | 3 | 0.446 |
| S | UL | ea | 91 | 0.8356 | 0.0101 | multiple_windows_fs / up_to_30s / 115 | 30 | 6 | 12 | 4 | 0.181 |
| S | UL>100B | pf | 46 | 0.7705 | 0.0836 | multiple_windows / up_to_30s / 72 | 330 | 5 | 12 | 10 | 0.347 |
| S | UL>100B | ea | 70 | 0.7539 | 0.0076 | multiple_windows / up_to_30s / 72 | 30 | 4 | 12 | 16 | 0.323 |
| BH | ALL | pf | 79 | 0.8420 | 0.0822 | multiple_windows / up_to_30s / 288 | 190 | 6 | 12 | 16 | 0.326 |
| BH | ALL | ea | 69 | 0.8242 | 0.0094 | multiple_windows_fs / up_to_30s / 204 | 20 | 6 | 12 | 16 | 0.488 |
| BH | ALL>100B | pf | 118 | 0.8223 | 0.0130 | multiple_windows / up_to_30s / 144 | 40 | 5 | 12 | 15 | 0.380 |
| BH | ALL>100B | ea | 141 | 0.8121 | 0.0046 | multiple_windows / up_to_30s / 144 | 10 | 5 | 12 | 15 | 0.325 |
| BH | UL | pf | 23 | 0.8135 | 0.0516 | multiple_windows_fs / up_to_30s / 98 | 220 | 4 | 12 | 11 | 0.593 |
| BH | UL | ea | 68 | 0.8046 | 0.0105 | multiple_windows_fs / up_to_30s / 98 | 40 | 5 | 12 | 7 | 0.222 |
| BH | UL>100B | pf | 93 | 0.7291 | 0.0719 | multiple_windows / up_to_30s / 72 | 340 | 4 | 12 | 16 | 0.473 |
| BH | UL>100B | ea | 66 | 0.7248 | 0.0075 | multiple_windows / up_to_30s / 72 | 30 | 4 | 12 | 15 | 0.332 |

### LightGBM (LGBM)

| KQI | View | Sel. | Trial | Score | E_norm | Feature set (variant / windows / count) | trees | max depth 2^x | leaves 2^x | learning rate |
|---|---|---|---|---|---|---|---|---|---|---|
| S | ALL | pf | 83 | 0.8688 | 0.1629 | multiple_windows_fs / up_to_30s / 231 | 1370 | 6 | 7 | 3.116e-02 |
| S | ALL | ea | 80 | 0.8375 | 0.0146 | multiple_windows_fs / up_to_30s / 231 | 90 | 8 | 7 | 2.729e-02 |
| S | ALL>100B | pf | 79 | 0.8644 | 0.1216 | multiple_windows_fs / up_to_30s / 114 | 2100 | 5 | 7 | 4.631e-02 |
| S | ALL>100B | ea | 98 | 0.8381 | 0.0101 | multiple_windows / up_to_30s / 144 | 60 | 7 | 7 | 5.750e-02 |
| S | UL | pf | 120 | 0.8534 | 0.2217 | multiple_windows / up_to_30s / 144 | 1230 | 8 | 7 | 2.539e-02 |
| S | UL | ea | 55 | 0.8468 | 0.0519 | multiple_windows_fs / up_to_30s / 115 | 1690 | 6 | 7 | 9.026e-02 |
| S | UL>100B | pf | 98 | 0.7938 | 0.1192 | multiple_windows_fs / up_to_30s / 56 | 2320 | 8 | 7 | 5.177e-02 |
| S | UL>100B | ea | 130 | 0.7685 | 0.0181 | multiple_windows_fs / up_to_30s / 56 | 110 | 6 | 7 | 4.943e-02 |
| BH | ALL | pf | 96 | 0.8887 | 0.4901 | multiple_windows / up_to_30s / 288 | 2400 | 7 | 7 | 6.284e-02 |
| BH | ALL | ea | 72 | 0.8547 | 0.0402 | multiple_windows / up_to_30s / 288 | 240 | 6 | 6 | 9.821e-02 |
| BH | ALL>100B | pf | 179 | 0.8830 | 0.5207 | multiple_windows / up_to_30s / 144 | 2480 | 5 | 7 | 6.872e-02 |
| BH | ALL>100B | ea | 180 | 0.8427 | 0.0251 | multiple_windows / up_to_30s / 144 | 110 | 4 | 7 | 8.733e-02 |
| BH | UL | pf | 119 | 0.8696 | 0.8311 | multiple_windows / up_to_30s / 144 | 2950 | 5 | 7 | 4.452e-02 |
| BH | UL | ea | 140 | 0.8370 | 0.0281 | multiple_windows / up_to_30s / 144 | 130 | 6 | 7 | 9.443e-02 |
| BH | UL>100B | pf | 150 | 0.7835 | 0.2592 | multiple_windows / up_to_30s / 72 | 1690 | 3 | 7 | 8.489e-02 |
| BH | UL>100B | ea | 145 | 0.7190 | 0.0093 | multiple_windows / up_to_30s / 72 | 50 | 4 | 7 | 9.870e-02 |

### Feed-forward network (FFNN)

| KQI | View | Sel. | Trial | Score | E_norm | Feature set (variant / windows / count) | layers | layer size 2^x | dropout | batch 2^x | learning rate | pos. weight |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S | ALL | pf | 10 | 0.6969 | 0.1028 | multiple_windows / up_to_30s / 288 | 3 | 9 | 0.0 | 12 | 1.844e-04 | 3 |
| S | ALL | ea | 23 | 0.6816 | 0.0748 | multiple_windows_fs / up_to_30s / 231 | 3 | 8 | 0.4 | 11 | 4.238e-04 | 2 |
| S | ALL>100B | pf | 45 | 0.6926 | 0.0755 | multiple_windows_fs / up_to_30s / 114 | 4 | 10 | 0.1 | 11 | 9.845e-03 | 1 |
| S | ALL>100B | ea | 36 | 0.6543 | 0.0713 | multiple_windows_fs / up_to_30s / 114 | 4 | 9 | 0.1 | 12 | 5.038e-03 | 1 |
| S | UL | pf | 41 | 0.6955 | 0.0713 | multiple_windows_fs / up_to_30s / 115 | 4 | 9 | 0.3 | 12 | 2.012e-03 | 1 |
| S | UL | ea | 70 | 0.6449 | 0.0699 | multiple_windows_fs / up_to_20s / 96 | 4 | 8 | 0.3 | 12 | 7.744e-03 | 1 |
| S | UL>100B | pf | 71 | 0.5625 | 0.0961 | multiple_windows_fs / up_to_30s / 56 | 3 | 11 | 0.0 | 12 | 1.232e-04 | 1 |
| S | UL>100B | ea | 101 | 0.4998 | 0.0678 | multiple_windows_fs / up_to_30s / 56 | 2 | 7 | 0.0 | 12 | 2.892e-02 | 1 |
| BH | ALL | pf | 75 | 0.7849 | 0.1822 | multiple_windows / up_to_30s / 288 | 4 | 10 | 0.2 | 7 | 2.694e-03 | -- |
| BH | ALL | ea | 122 | 0.7571 | 0.1666 | multiple_windows_fs / up_to_30s / 204 | 3 | 7 | 0.2 | 10 | 6.818e-03 | -- |
| BH | ALL>100B | pf | 25 | 0.7671 | 0.1651 | multiple_windows / up_to_30s / 144 | 3 | 9 | 0.1 | 8 | 7.559e-03 | -- |
| BH | ALL>100B | ea | 25 | 0.7671 | 0.1651 | multiple_windows / up_to_30s / 144 | 3 | 9 | 0.1 | 8 | 7.559e-03 | -- |
| BH | UL | pf | 41 | 0.7606 | 0.1732 | multiple_windows_fs / up_to_30s / 98 | 4 | 9 | 0.2 | 7 | 1.547e-02 | -- |
| BH | UL | ea | 81 | 0.6589 | 0.1591 | multiple_windows_fs / up_to_20s / 40 | 2 | 9 | 0.5 | 7 | 1.613e-03 | -- |
| BH | UL>100B | pf | 86 | 0.6439 | 0.1710 | multiple_windows_fs / up_to_30s / 53 | 3 | 11 | 0.5 | 7 | 3.095e-03 | -- |
| BH | UL>100B | ea | 76 | 0.6111 | 0.1581 | multiple_windows_fs / up_to_30s / 53 | 3 | 8 | 0.5 | 7 | 1.830e-03 | -- |

### 1-D CNN

| KQI | View | Sel. | Trial | Score | E_norm | Feature set (variant / windows / count) | seq. len | layers | kernels 2^x | kernel size | dropout | batch 2^x | learning rate | pos. weight |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S | ALL | pf | 77 | 0.8010 | 0.2535 | multiple_windows_fs / up_to_20s / 192 | 25 | 2 | 6 | 2 | 0.2 | 12 | 2.710e-02 | 1 |
| S | ALL | ea | 25 | 0.7599 | 0.1103 | dl_1s / -- / 48 | 20 | 2 | 5 | 9 | 0.0 | 10 | 4.549e-03 | 1 |
| S | ALL>100B | pf | 5 | 0.8037 | 0.2994 | multiple_windows / up_to_20s / 120 | 30 | 2 | 6 | 7 | 0.3 | 7 | 1.391e-03 | 2 |
| S | ALL>100B | ea | 12 | 0.7250 | 0.1043 | dl_1s / -- / 24 | 20 | 2 | 5 | 10 | 0.0 | 10 | 3.012e-04 | 3 |
| S | UL | pf | 100 | 0.7982 | 0.1813 | multiple_windows_fs / up_to_30s / 115 | 25 | 2 | 6 | 4 | 0.2 | 10 | 4.380e-03 | 1 |
| S | UL | ea | 29 | 0.7160 | 0.1011 | dl_1s / -- / 24 | 15 | 2 | 5 | 2 | 0.0 | 10 | 2.004e-02 | 1 |
| S | UL>100B | pf | 86 | 0.7233 | 0.1783 | multiple_windows_fs / up_to_30s / 56 | 25 | 3 | 5 | 6 | 0.1 | 8 | 4.174e-02 | 1 |
| S | UL>100B | ea | 42 | 0.6859 | 0.1282 | multiple_windows_fs / up_to_30s / 56 | 15 | 3 | 5 | 8 | 0.1 | 9 | 1.545e-02 | 1 |
| BH | ALL | pf | 61 | 0.8697 | 0.5007 | multiple_windows_fs / up_to_10s / 25 | 55 | 3 | 6 | 9 | 0.1 | 7 | 2.797e-04 | -- |
| BH | ALL | ea | 112 | 0.8506 | 0.2137 | multiple_windows_fs / up_to_10s / 25 | 55 | 2 | 6 | 8 | 0.0 | 7 | 2.647e-04 | -- |
| BH | ALL>100B | pf | 116 | 0.8846 | 0.4321 | multiple_windows_fs / up_to_20s / 19 | 55 | 3 | 6 | 3 | 0.0 | 7 | 2.373e-03 | -- |
| BH | ALL>100B | ea | 135 | 0.8775 | 0.2233 | multiple_windows_fs / up_to_20s / 19 | 55 | 3 | 5 | 9 | 0.0 | 7 | 3.463e-03 | -- |
| BH | UL | pf | 42 | 0.8662 | 0.2212 | multiple_windows_fs / up_to_10s / 19 | 55 | 3 | 5 | 6 | 0.0 | 7 | 9.345e-04 | -- |
| BH | UL | ea | 4 | 0.5181 | 0.1731 | dl_1s / -- / 24 | 20 | 1 | 4 | 6 | 0.2 | 9 | 3.160e-03 | -- |
| BH | UL>100B | pf | 6 | 0.7633 | 0.2443 | multiple_windows_fs / up_to_30s / 53 | 60 | 2 | 5 | 6 | 0.0 | 10 | 2.109e-02 | -- |
| BH | UL>100B | ea | 41 | 0.5338 | 0.1725 | multiple_windows_fs / up_to_20s / 42 | 5 | 1 | 5 | 3 | 0.0 | 8 | 4.990e-02 | -- |

### LSTM

| KQI | View | Sel. | Trial | Score | E_norm | Feature set (variant / windows / count) | seq. len | layers | hidden 2^x | dropout | batch 2^x | learning rate | pos. weight |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S | ALL | pf | 60 | 0.8330 | 0.4854 | multiple_windows_fs / up_to_30s / 231 | 45 | 1 | 8 | -- | 12 | 9.347e-03 | 1 |
| S | ALL | ea | 74 | 0.8028 | 0.1328 | dl_1s / -- / 48 | 45 | 3 | 5 | 0.0 | 11 | 1.354e-02 | 3 |
| S | ALL>100B | pf | 39 | 0.8467 | 0.3499 | multiple_windows / up_to_20s / 120 | 55 | 3 | 7 | 0.2 | 11 | 6.255e-03 | 1 |
| S | ALL>100B | ea | 69 | 0.7942 | 0.1099 | dl_1s / -- / 24 | 45 | 2 | 5 | 0.0 | 11 | 9.861e-03 | 3 |
| S | UL | pf | 6 | 0.8267 | 0.5570 | multiple_windows / up_to_20s / 120 | 45 | 2 | 8 | 0.2 | 7 | 5.666e-04 | 4 |
| S | UL | ea | 14 | 0.8043 | 0.1749 | multiple_windows_fs / up_to_5s / 57 | 55 | 2 | 6 | 0.4 | 10 | 2.099e-03 | 1 |
| S | UL>100B | pf | 39 | 0.7772 | 0.1709 | multiple_windows_fs / up_to_30s / 56 | 55 | 1 | 5 | -- | 9 | 9.050e-03 | 1 |
| S | UL>100B | ea | 83 | 0.6108 | 0.0938 | dl_1s / -- / 12 | 40 | 1 | 5 | -- | 10 | 8.574e-03 | 5 |
| BH | ALL | pf | 117 | 0.9084 | 0.4792 | multiple_windows_fs / up_to_10s / 25 | 55 | 3 | 7 | 0.1 | 8 | 5.552e-03 | -- |
| BH | ALL | ea | 101 | 0.9002 | 0.1951 | multiple_windows_fs / up_to_10s / 25 | 55 | 1 | 7 | -- | 7 | 8.129e-03 | -- |
| BH | ALL>100B | pf | 36 | 0.9208 | 1.0000 | multiple_windows / up_to_30s / 144 | 55 | 2 | 8 | 0.3 | 9 | 1.031e-02 | -- |
| BH | ALL>100B | ea | 31 | 0.8744 | 0.2082 | dl_1s / -- / 24 | 55 | 2 | 6 | 0.4 | 7 | 2.221e-03 | -- |
| BH | UL | pf | 152 | 0.9025 | 0.2988 | multiple_windows_fs / up_to_30s / 98 | 50 | 2 | 6 | 0.5 | 7 | 3.138e-03 | -- |
| BH | UL | ea | 73 | 0.8744 | 0.1866 | multiple_windows_fs / up_to_10s / 19 | 55 | 2 | 5 | 0.2 | 9 | 1.069e-02 | -- |
| BH | UL>100B | pf | 52 | 0.8500 | 0.2387 | multiple_windows_fs / up_to_30s / 53 | 55 | 2 | 6 | 0.4 | 9 | 2.389e-02 | -- |
| BH | UL>100B | ea | 11 | 0.7688 | 0.1714 | dl_1s / -- / 12 | 55 | 1 | 6 | -- | 7 | 3.092e-03 | -- |

### Transformer encoder (TF)

| KQI | View | Sel. | Trial | Score | E_norm | Feature set (variant / windows / count) | seq. len | layers | heads 2^x | d_model 2^x | d_ff 2^x | dropout | batch 2^x | learning rate | pos. weight |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S | ALL | pf | 81 | 0.8014 | 0.3954 | multiple_windows_fs / up_to_30s / 231 | 45 | 1 | 1 | 6 | 9 | 0.5 | 11 | 2.477e-03 | 1 |
| S | ALL | ea | 65 | 0.7541 | 0.1050 | multiple_windows_fs / up_to_30s / 231 | 5 | 1 | 1 | 6 | 8 | 0.5 | 10 | 1.687e-03 | 1 |
| S | ALL>100B | pf | 142 | 0.7935 | 0.1832 | multiple_windows / up_to_30s / 144 | 10 | 2 | 1 | 5 | 9 | 0.4 | 9 | 1.120e-03 | 1 |
| S | ALL>100B | ea | 99 | 0.7767 | 0.1210 | multiple_windows_fs / up_to_30s / 114 | 5 | 1 | 1 | 7 | 7 | 0.2 | 10 | 1.863e-03 | 1 |
| S | UL | pf | 54 | 0.7973 | 0.2911 | multiple_windows_fs / up_to_30s / 115 | 40 | 1 | 1 | 7 | 8 | 0.5 | 10 | 1.881e-04 | 8 |
| S | UL | ea | 104 | 0.7680 | 0.0969 | multiple_windows_fs / up_to_30s / 115 | 5 | 1 | 1 | 5 | 10 | 0.1 | 10 | 4.373e-03 | 2 |
| S | UL>100B | pf | 82 | 0.7428 | 0.2441 | multiple_windows / up_to_20s / 60 | 50 | 2 | 1 | 5 | 8 | 0.1 | 11 | 7.475e-03 | 1 |
| S | UL>100B | ea | 55 | 0.6609 | 0.1201 | multiple_windows_fs / up_to_30s / 56 | 15 | 1 | 2 | 5 | 7 | 0.2 | 10 | 1.576e-03 | 4 |
| BH | ALL | pf | 194 | 0.8752 | 0.4315 | multiple_windows_fs / up_to_20s / 145 | 50 | 1 | 3 | 6 | 8 | 0.1 | 9 | 2.690e-03 | -- |
| BH | ALL | ea | 152 | 0.8052 | 0.2402 | multiple_windows_fs / up_to_30s / 204 | 5 | 1 | 3 | 7 | 7 | 0.2 | 8 | 2.344e-03 | -- |
| BH | ALL>100B | pf | 39 | 0.8850 | 0.4981 | multiple_windows / up_to_30s / 144 | 55 | 1 | 1 | 6 | 10 | 0.1 | 10 | 2.648e-03 | -- |
| BH | ALL>100B | ea | 114 | 0.8270 | 0.2346 | multiple_windows / up_to_30s / 144 | 5 | 1 | 1 | 5 | 10 | 0.0 | 8 | 1.134e-02 | -- |
| BH | UL | pf | 137 | 0.8620 | 0.2939 | multiple_windows_fs / up_to_30s / 98 | 40 | 1 | 1 | 5 | 7 | 0.0 | 10 | 1.220e-02 | -- |
| BH | UL | ea | 63 | 0.8088 | 0.2329 | multiple_windows_fs / up_to_30s / 98 | 5 | 1 | 1 | 7 | 9 | 0.2 | 9 | 1.373e-03 | -- |
| BH | UL>100B | pf | 84 | 0.8363 | 0.8986 | multiple_windows_fs / up_to_30s / 53 | 40 | 2 | 2 | 7 | 10 | 0.0 | 9 | 2.462e-03 | -- |
| BH | UL>100B | ea | 122 | 0.7781 | 0.2486 | multiple_windows_fs / up_to_20s / 42 | 45 | 1 | 1 | 5 | 7 | 0.0 | 9 | 8.896e-03 | -- |

---

## 6. Hyperparameter semantics and search spaces

### 6.1 What each parameter means

From `tree_models.py:29-43`, `tree_modes_process.py:44-62` and `neural_models.py:111-183`.

| Parameter | Applies to | Effective value in the model |
|---|---|---|
| `dataset` | all | feature-set variant, see `FEATURES.md` §1.3 |
| `feature_window_size` | all | window set, `up_to_{5,10,20,30}s` |
| `sequence_window_size` | CNN, LSTM, TF | number of consecutive per-second feature vectors fed to the encoder; forced to 1 for FFNN (`model_training.py:104`) |
| `n_estimators` | RF, LGBM | number of trees, directly |
| `max_depth_exponent` | RF, LGBM | `max_depth = 2 ** x` |
| `max_leaves_exponent` | RF | `max_leaves = 2 ** x` |
| `num_leafs_exponent` | LGBM | `num_leaves = 2 ** x` |
| `min_samples_leaf` | RF | directly |
| `max_features` | RF | fraction of features considered per split, directly |
| `learning_rate` | LGBM and all neural | directly; AdamW for the neural models |
| `n_layers` | FFNN, CNN, LSTM, TF | number of layers |
| `layer_size_exponent` | FFNN | first hidden width `2 ** x`, then halved per further layer: `layers = [n_features] + [2**x // 2**i for i in 1..n_layers-1]` |
| `n_kernels_exponent` | CNN | channels of the first conv layer `2 ** x`, doubled per layer |
| `kernel_size` | CNN | conv kernel width, **used directly, not as a power of two**; sampled from `2 .. min(sequence_window_size, 10)` |
| `hidden_size_exponent` | LSTM | hidden size `2 ** x` |
| `n_head_exponent` | TF | attention heads `2 ** x` |
| `dim_model_exponent` | TF | `d_model = 2 ** x` |
| `dim_feedforward_exponent` | TF | `dim_feedforward = 2 ** x` |
| `dropout` | FFNN, CNN, LSTM, TF | directly; forced to 0 for a single-layer LSTM (`neural_models.py:165`) |
| `batch_exponent` | FFNN, CNN, LSTM, TF | training batch size `2 ** x` |
| `loss_weight` | stalling only | `pos_weight` of `BCEWithLogitsLoss` |

Two parameters are sampled but can end up unused, and this is visible in the tables of §5:

* `layer_size_exponent` is only read when `n_layers > 1` (`neural_models.py:148-151`), so for
  a single-layer FFNN the recorded value has no effect on the model.
* `dropout` is overridden to 0 for a single-layer LSTM.

### 6.2 Search spaces as stored in the databases

Decoded from `trial_params.distribution_json`, unioned over both KQIs. Where two ranges are
listed for one cell, both occur in the databases; see the notes below.

| Parameter | RF | LGBM | FFNN | CNN | LSTM | TF |
|---|---|---|---|---|---|---|
| `dataset` | {multiple_windows, multiple_windows_fs} | {multiple_windows, multiple_windows_fs} | {multiple_windows, multiple_windows_fs} | {dl_0.5s, dl_1s, multiple_windows, multiple_windows_fs} | {dl_0.5s, dl_1s, multiple_windows, multiple_windows_fs} | {dl_0.5s, dl_1s, multiple_windows, multiple_windows_fs} |
| `feature_window_size` | {up_to_30s, up_to_20s, up_to_10s, up_to_5s} | {up_to_30s, up_to_20s, up_to_10s, up_to_5s} | {up_to_5s, up_to_10s, up_to_20s, up_to_30s} | {up_to_5s, up_to_10s, up_to_20s, up_to_30s} | {up_to_5s, up_to_10s, up_to_20s, up_to_30s} | {up_to_5s, up_to_10s, up_to_20s, up_to_30s} |
| `sequence_window_size` | -- | -- | -- | 10--120, step 10 / 5--60, step 5 | 10--120, step 10 / 5--60, step 5 | 10--120, step 10 / 5--60, step 5 |
| `n_estimators` | 10--1000, step 10 | 10--5000, step 10 | -- | -- | -- | -- |
| `max_depth_exponent` | 2--6 | 2--8 | -- | -- | -- | -- |
| `max_leaves_exponent` | 6--12 | -- | -- | -- | -- | -- |
| `num_leafs_exponent` | -- | 4--7 | -- | -- | -- | -- |
| `min_samples_leaf` | 1--16 | -- | -- | -- | -- | -- |
| `max_features` | 0.1--0.8 | -- | -- | -- | -- | -- |
| `n_layers` | -- | -- | 1--4 | 1--3 | 1--3 | 1--3 |
| `layer_size_exponent` | -- | -- | 5--11 | -- | -- | -- |
| `n_kernels_exponent` | -- | -- | -- | 4--6 | -- | -- |
| `kernel_size` | -- | -- | -- | 2--10 / 2--5 | -- | -- |
| `hidden_size_exponent` | -- | -- | -- | -- | 4--8 | -- |
| `n_head_exponent` | -- | -- | -- | -- | -- | 1--3 |
| `dim_model_exponent` | -- | -- | -- | -- | -- | 5--8 |
| `dim_feedforward_exponent` | -- | -- | -- | -- | -- | 6--10 |
| `dropout` | -- | -- | 0--0.5, step 0.1 | 0--0.5, step 0.1 | 0--0.5, step 0.1 | 0--0.5, step 0.1 |
| `batch_exponent` | -- | -- | 7--12 | 7--12 | 7--12 | 7--12 |
| `learning_rate` | -- | 1e-05--0.1, log | 1e-05--0.1, log | 1e-05--0.1, log | 1e-05--0.1, log | 1e-05--0.1, log |
| `loss_weight` | -- | -- | 1--20, log | 1--20, log | 1--20, log | 1--20, log |

Notes.

* `sequence_window_size` has two distributions. The 10-120 step 10 range belongs **exactly** to
  the `dl_0.5s` trials, whose tick is 0.5 s, so it also means 5-60 s of history
  (`neural_models.py:120-122`). Of the 3,149 stored values 501 are from the wide range. No
  selected configuration exceeds 60 s.
* `kernel_size` has upper bound `min(sequence_window_size, 10)` (`neural_models.py:160`), so
  `high=5` appears for the 96 trials with a 5-step history and `high=10` for the other 792.
* `loss_weight` exists only in the six stalling databases, as the paper's footnote says.
* The `feature_window_size` choice list is ordered `up_to_30s` first in the tree databases and
  `up_to_5s` first in the neural ones. The ordering only affects the integer encoding in
  `trial_params`, which `reconstruct_configurations.py` decodes per trial from that trial's own
  `distribution_json`; it has no effect on the search.

Compared with the commented-out search-space table in `methodology.tex:24-53`, every range
agrees, with two exceptions, both in the commented table:

* The CNN column of the row "Layer / hidden size (2^x)" shows 2-10. That is the CNN's
  `kernel_size`, which is used linearly and not as an exponent, and whose upper bound is
  conditional on the history length. The CNN has no layer-size exponent; its width parameter
  is `n_kernels_exponent`, already listed in the "Kernels (2^x)" row.
* The table does not list `kernel_size` as a parameter of its own.

---

## 7. From a selected trial to a measured model

The search trained on 10 % of the data for at most 20 epochs. The 96 role entries were then
retrained from scratch (`config_controller.py:511-540` writes one config per role, consumed by
`model_training.py`):

* **Full training split**, all 9,018 runs (`model_training.py:42` and `:121`).
* Neural networks: up to **100 epochs** with early stopping on the validation split
  (`model_training.py:170`), patience 10, `min_delta=1e-4`, minimum 20 epochs
  (`machine_learning_helpers.py:158-179`). Batch size from `batch_exponent`, `drop_last=True`.
* LightGBM: early stopping with 50 rounds of patience on the **second half** of the validation
  split (`model_training.py:44-45` and `:75`). The number of trees actually kept is recorded
  as `n_estimators_es` and may be far below the sampled `n_estimators`.
* Random forests: fitted on the GPU with cuML, then converted
  (`tree_modes_process.py:66-68`), and exported to ONNX via Hummingbird.
* Stalling threshold: the threshold maximising F1 on the **first half** of the validation
  split (`model_training.py:82-84`, `machine_learning_helpers.py:278-296`), fixed before
  testing. The paper says "the classification threshold that maximizes validation F1 is fixed
  before testing" without saying it is half the split.
* Edge inference: ONNX Runtime, CPU provider on the Raspberry Pi 4 and CUDA provider on the
  Jetson Orin Nano Super; tree ensembles converted to tensor operations with Hummingbird
  (`methodology.tex:61`).

The retraining seed is a single draw, `seed = seeds[iteration]` with `iteration = 0`
(`config_controller.py:485-490`), and it is written into every generated config, so all 96
retrainings share one seed. The exact value is not recorded in the released data — it depends
on the state of the module-level `RNG` at the time `model_training()` was called — so the
retrainings are **not bit-reproducible** from the artifact. The selections in this document
are, because they depend only on the stored databases.

Naming in every downstream file is
`{kqi}_{view_code}_{family}_{performance|efficiency}`; the `config_name` column of
`configurations.csv` gives it. `performance` is the paper's performance-first and `efficiency`
is energy-aware.

---

## 8. The live (W2) shortlist

W2 evaluates a subset of these configurations, chosen per device from the W1 results:
for each view and KQI the highest-scoring, lowest-energy and knee configuration across
families (paper Table I). The per-device shortlists with the scores, energies, the Pareto
flag and the two balance criteria are in
`selected_models/{raspberry_pi,jetson_nano_orin}/{kqi}_{view_code}.csv`
(16 files, columns `energy, score, energy_scaled, score_scaled, pareto, combined_utopian,
combined_elbow`), and they are selected in `config_controller.py:556-560` by
`df["score"].idxmax()`, `df["energy"].idxmin()` and `df["combined_elbow"].idxmax()`.

Note that the W1-level selection uses a **different** knee rule from the search-level one
documented in §2.3: `evalutation.py:782-795` computes the Pareto front over the 12 W1
configurations of a view and KQI and takes the point of maximum perpendicular distance from
the chord between the front's two extremes, with a `combined_utopian` alternative that
minimises the distance to the ideal point. That selection is a property of the measurement
analysis, not of the hyperparameter search, and is out of scope here; it is listed because the
two knees are easy to confuse.

---
