# Reproducing the paper's numbers

Every command below is run from `code/analysis/`. Paths are relative to that directory.
Python 3.10 or newer with NumPy, pandas and scikit-learn is enough for all of them.

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
| Appendix, outlier filter | 181 of 5,760 batch-1 values removed |

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

## 4. The energy model

The model implementation is in `code/model/`. `audit.py` regenerates the calibration and the
scenario checks; `ENERGY_MODEL.md` section 7 gives the entry points and the directory layout
they expect.

## 5. The figures

The scripts that drew the paper's three figures are in `code/figures/`
(`compact_scores.py` for the test-score figure, `compact_measurements.py` for the offline-pair
and live-delta figures, `redraw_fonts.py` for the font pass). They read the same measurement
tree as the analysis scripts.

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
