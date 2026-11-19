# Energy-Aware Assessment and Energy Modeling for Encrypted Video KQI Prediction at the Edge

Measurement data, analysis code and documentation for the paper of the same title, by Flavian
Raithel, Tobias Hoßfeld and Frank Loh (Chair of Communication Networks, University of Würzburg).

The paper measures what it costs, in energy, to monitor encrypted video quality continuously. It
predicts two Key Quality Indicators from packet metadata at one-second granularity — playback
stalling (binary) and buffer health (regression) — with six model families in two configurations
each, and measures whole-device energy at the AC input on two edge platforms: a CPU-only
Raspberry Pi 4 Model B and a GPU-equipped Jetson Orin Nano Super.

## What is here

| Path | Contents |
|---|---|
| [`REPRODUCE.md`](REPRODUCE.md) | One command per result, with the expected output |
| [`FEATURES.md`](FEATURES.md) | The feature design: base statistics, the four traffic views, the sliding windows, the feature selection, and which feature set each model uses |
| [`CONFIGURATIONS.md`](CONFIGURATIONS.md) | Every selected configuration with its hyperparameters and objective values, reconstructed from the search databases |
| [`MEASUREMENTS.md`](MEASUREMENTS.md) | The measurement protocol: meter, device settings, the three workloads, repetitions, statistics, exclusions |
| [`ENERGY_MODEL.md`](ENERGY_MODEL.md) | The additive stage-cost model, how each parameter is calibrated, and how to re-run it |
| [`appendix/supplementary.pdf`](appendix/supplementary.pdf) | The supplementary document the paper cites: robustness checks for the energy measurements and the full hyperparameter search ranges |
| [`code/`](code/) | `analysis/` reproduction scripts, `model/` the energy model, `baseline/` the linear baselines, `features/` the feature and configuration generators, `figures/` the figure scripts |
| [`data/`](data/) | The measurement series, the 12 search databases, feature importances and derived tables. See [`data/README.md`](data/README.md) |

## Reproduced from this data

| Result | Status |
|---|---|
| Offline-inference energy, all 48 cells, both devices, both selections | reproduces exactly |
| Median energy-aware savings, 72 % (Raspberry Pi) and 19 % (Jetson), and the per-KQI split | reproduces exactly |
| Batch-1 to batch-32 energy factors, 5.2 and 17.6 | reproduces exactly |
| Full-test prediction scores, e.g. LightGBM at 89.4 % AP and LSTM at R² 93.2 % | reproduces exactly |
| Every calibrated parameter of the stage-cost energy model | reproduces exactly |
| Confidence-interval and outlier-filter statements | reproduces exactly |

`REPRODUCE.md` gives the command for each.

## Scope of the prediction scores

The dataset split assigns playback runs at random rather than videos, so test runs generally play
a video that also occurs in training, and 18 % of them have a byte-identical copy in training.
The scores therefore describe monitoring of a known video catalogue. A retraining pilot with a
video-grouped split, reported in the paper, puts the cost of unseen content at 11 to 22 points of
average precision and 18 to 23 points of R². The energy results do not depend on label quality
and are unaffected.

For concurrent load, packet capture and dissection are validated against held-out loads. The
full-pipeline energy and the device capacities at many streams are projections of the stage-cost
model; `ENERGY_MODEL.md` states which is which.

## The dataset

The traffic and player-state data are not redistributed here. They are the public YouTube
streaming dataset of Loh et al., 11,142 playback runs over 246 videos and 1,081 hours of
streaming; see the dataset references in the paper. This repository contains the measurements
taken on top of it, the derived tables and the code.

One property of that dataset is worth passing on independently of this paper: it contains 1,161
pairs of byte-identical playback runs recorded under two run numbers. `data/derived/` holds the
per-run video identities from which the pair list is regenerated.

## Requirements

Python 3.10 or newer with NumPy, pandas, SciPy, scikit-learn, Matplotlib and pyarrow. The
scripts in `code/analysis/` need nothing else. The energy model in `code/model/` and the
baselines in `code/baseline/` expect the original directory layout; see `ENERGY_MODEL.md` and
`data/README.md`.

## Licence and citation

Code is under the MIT licence, documentation and data tables under CC BY 4.0. See
[`LICENSE`](LICENSE) and [`CITATION.cff`](CITATION.cff).

## Acknowledgment

This work has been performed in the framework of the SUSTAINET-Advance project, funded by the
German BMFTR (16KIS2282, Univ. Würzburg).
