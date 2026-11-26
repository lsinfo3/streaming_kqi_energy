# Reproducing the paper's numbers

Every command below is run from `code/analysis/`. Paths are relative to that directory.
Python 3.10 or newer with NumPy, pandas, SciPy and scikit-learn is enough for all of them.

```bash
cd code/analysis
```

## 1. The dataset split

```bash
python split.py
```

Prints the split sizes and the first test runs in pipeline order. Expected output:
9,018 training, 1,033 validation and 1,091 test runs; the first five test runs are
2961, 6206, 3322, 8289, 4764; the checksum over the test run IDs is 5,995,898.

The order matters beyond bookkeeping: the per-run sample blocks of the stored datasets and of
the exported prediction arrays are concatenated in exactly this order, which is what makes a
later run-level or video-level analysis possible at all.

## 2. Offline inference energy (workload W1)

```bash
python repro_w1.py ../../data/w1_offline
```

Reproduces, from the 384 measurement files of 30 repetitions each:

| Paper claim | Expected output |
|---|---|
| Table III, whole-device energy for 100,000 predictions, ALL view | all 48 cells, both devices, performance-first and energy-aware |
| Section IV-B, median energy-aware saving | −71.8 % (Raspberry Pi), −19.4 % (Jetson) |
| Section IV-B, per KQI | stalling −71.8 % and buffer health −73.7 % on the Pi; −18.4 % and −20.2 % on the Jetson |
| Section IV-B, pairs whose energy rises | 2 on the Pi, 5 on the Jetson |
| Section IV-C, batch-1 to batch-32 factor | 5.24 on the Pi over 80 configurations, 17.56 on the Jetson over 96, that is 81 % and 94 % savings |
| Section IV-C, runs under 10 seconds | 14 of 24 energy-aware tree and FFNN configurations on the Pi, 0 of 24 on the Jetson, shortest run 32.6 s |
| Table III footnote | confidence interval below 2 % of the mean for 97.9 % of batch-1 configurations, worst 3.27 % |
| Supplementary document, outlier filter | 181 of 5,760 batch-1 values removed |

Two deliberate conventions are built into the script. Means are taken after removing values
beyond 1.5 interquartile ranges, as in the paper. And the 16 Raspberry Pi batched stalling RF
and LGBM configurations are excluded from the batching factor, because their batched
predictions do not match their unbatched ones (`MEASUREMENTS.md` section 8). Including them
gives 5.64 instead of 5.24.

The baseline share of W1 energy comes out at 57 % (Pi) and 73 % (Jetson) against the 56 % and
75 % in Section IV-C, because the script uses the rounded idle powers of 2.56 W and 4.27 W
rather than the full twelve-hour idle series in `data/idle/`.

## 3. Full-test prediction scores

```bash
python repro_scores.py --predictions /path/to/general_data --out scores.csv
```

Recomputes average precision and F1 for stalling, and R² and mean absolute error for buffer
health, for all 96 exported configurations. Reproduces the Section IV-A figures: LightGBM
performance-first at 89.38 % average precision and 79.45 % F1 under the ALL view, and LSTM
performance-first at R² 93.22 % with a mean absolute error of 5.86 s.

**This one needs data that is not in the repository.** The per-sample prediction and
ground-truth arrays are about 325 MB, so `--predictions` must point at the `general_data`
directory of the measurement data release, containing `stalling_predictions/` and
`buffer_health_predictions/`.

Two things the script reports that the paper does not: the test population ranges from 249,600
to 316,964 samples across configurations, and every neural configuration's array length is an
exact multiple of 128 while the tree models' are not, because the neural pipeline drops the
final partial batch.

One convention is easy to get wrong, so the script handles it explicitly. The exported
buffer-health predictions are **raw regressor output and are not clipped** to the valid [0, 1]
range of the normalised target: individual configurations put up to 25 % of their predictions
outside it. The paper's metrics are computed after clipping, so the script clips too, and the
clipped values reproduce every stored `r2` and `mae` field to within 1e-8. Scoring the raw
arrays instead disagrees with the paper by up to 1.25 points of R². The decisive case is the
coinciding ALL>100B FFNN pair that the paper reports at R² 80.7 % and 84.2 %: clipped gives
80.68 and 84.19, raw gives 80.40 and 82.94. **Always clip buffer-health predictions before
scoring.** Stalling is unaffected, since those arrays are already probabilities in [0, 1], and
average precision and F1 agree with the stored fields to 14 decimal places.

## 4. Live single-stream monitoring (workload W2)

```bash
python repro_w2.py
```

Compares each live candidate's five-minute energy with its reference run without inference
(30 runs each, 1.5-IQR filtered means). Expected output:

| Paper claim | Expected output |
|---|---|
| Section IV-D, valid candidates | 21 of 23 on the Raspberry Pi, 22 of 24 on the Jetson |
| Section IV-D, predictions per run | 0.1–43 for the two UL>100B sequence models, 238–273 for all valid candidates |
| Section IV-D, Raspberry Pi | 10 performance-first candidates add 62.6–99.1 J (8.4–12.8 %); 11 energy-aware add at most 6.4 J (0.9 %) |
| Section IV-D, Jetson | no valid candidate adds more than 14.5 J (1.1 %) |
| Section IV-D, identical performance-first models | the Jetson uses a median 47 % more total energy |
| Section IV-D, per-prediction cost | `c_m` varies 51-fold across the Pi's performance-first candidates; `c_m` times the live predictions is at most 10.4 J |

`c_m` is computed as in Table IV, `(E_W1 - P0 T_W1) / 100,000`, with `P0` taken from the
one-stream W3 capture runs (2.54 W and 4.35 W). The two live runs that were selected by mistake
(`MEASUREMENTS.md` section 8) are excluded.

## 5. Capture and dissection under load (workload W3)

```bash
python repro_w3.py
```

| Paper claim | Expected output |
|---|---|
| Section IV-E, capture without dissection at 100 streams | 847–856 J (Raspberry Pi), 1354–1361 J (Jetson) |
| Section IV-E, capture-rate decline with dissection, Raspberry Pi | ALL loss-free up to 50, below 98 % from 60 streams; ALL>100B 70 / 80; UL 170 / 180; UL>100B no decline up to the highest load measured, 100 |
| Section IV-E, Jetson | ALL 110 / 120; ALL>100B 160 / 170; UL no decline up to 200; UL>100B loss-free at 100, the single screening run at 200 at 95 % |
| Section IV-E, full-packet capture | Raspberry Pi 40 / 50, Jetson 90 / 100 |
| Section IV-E, deferred dissection of a 100-stream ALL capture | Raspberry Pi 14.6 min and 3.47 kJ, 2.9 times the capture duration and 4.1 times its energy; Jetson 4.3 min |

A load counts as loss-free while its median captured-to-sent packet ratio reaches 98 % of the
loss-free 100-stream reference of the same view. The script prints the ratio of every load, so
each threshold can be read directly. Above 100 streams the Jetson logs have no capture counter;
the packets that reached dissection are counted instead. Loads with fewer than 8 runs are
screening runs and are marked.

## 6. Parameter count and cross-device rankings

```bash
python repro_correlation.py
```

Reproduces the Spearman correlations of Section IV-C over all 96 configurations: parameter count
against unbatched W1 energy, 0.26 across families on the Raspberry Pi and 0.39 on the Jetson,
strong within most families except the transformer (0.61 and 0.11) and the Jetson's compiled RF
(0.45); and 0.34 between the two devices' energy rankings.

## 7. The energy model

The model implementation is in `code/model/`. `audit.py` regenerates the calibration and the
scenario checks; `ENERGY_MODEL.md` section 7 gives the entry points and the directory layout
they expect.

## 8. The figures

Run from the repository root; each script writes its PDF next to itself unless an output folder
is given:

```bash
python code/figures/compact_scores.py          # Fig. 2, full-test scores
python code/figures/offline_pairs_family.py    # Fig. 3, W1 energy against score
python code/figures/compact_measurements.py    # Fig. 4, live energy differences (W2)
```

They read only `data/derived/repro_scores.csv`, `data/derived/repro_w1.csv` and `data/w2_live/`.
Fig. 1 is drawn in the paper's LaTeX source.

## What this repository does not contain

* **The per-sample prediction arrays** (about 325 MB), needed by `repro_scores.py`. Obtain them
  from the measurement data release.
* **Model artefacts.** No ONNX exports and no fitted trees. Scores are reproducible from the
  exported predictions, not by retraining.
* **Run identifiers for the predictions.** The exported arrays are flat, so scores cannot be
  attributed to individual playback runs or videos.
* **The experiment pipeline.** Feature extraction, training, ONNX export, the two device runners
  and the meter harness are not included. The W2 and W3 series here can be re-aggregated, but
  taking new measurements needs the replay setup, the meter and the two edge devices.
