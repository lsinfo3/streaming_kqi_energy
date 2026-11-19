# Energy-Aware Assessment and Energy Modeling for Encrypted Video KQI Prediction at the Edge

Measurement data, analysis code and documentation for the paper of the same
title, by Flavian Raithel, Tobias Hoßfeld and Frank Loh (Chair of Communication Networks,
University of Würzburg).

The paper asks what it costs, in energy, to monitor encrypted video quality continuously. It
predicts two Key Quality Indicators from packet metadata at one-second granularity — playback
stalling (binary) and buffer health (regression) — with six model families in two configurations
each, and measures whole-device energy at the AC input on two edge platforms: a CPU-only
Raspberry Pi 4 Model B and a GPU-equipped Jetson Orin Nano Super.

## What is here

| Path | Contents |
|---|---|
| [`FEATURES.md`](FEATURES.md) | The feature design in full: the base statistics, the four traffic views, the sliding windows, which model uses which feature set, and the difference between the offline and the live feature path |
| [`CONFIGURATIONS.md`](CONFIGURATIONS.md) | All selected model configurations with their hyperparameters, reconstructed from the hyperparameter-search databases |
| [`MEASUREMENTS.md`](MEASUREMENTS.md) | The measurement protocol: meter, device settings, the three workloads, repetitions, statistics, and the known anomalies and exclusions |
| [`EVALUATION_CAVEATS.md`](EVALUATION_CAVEATS.md) | The dataset split, the duplicate runs, the video overlap, the test population sizes, and the transport-protocol structure of the dataset |
| [`ENERGY_MODEL.md`](ENERGY_MODEL.md) | The additive stage-cost model, how each parameter is calibrated, and which of its outputs are measured rather than projected |
| [`REPRODUCE.md`](REPRODUCE.md) | Commands that regenerate the paper's numbers from the data in this repository |
| [`appendix/supplementary.pdf`](appendix/supplementary.pdf) | **The supplementary document the paper cites.** Robustness checks for the energy measurements, and the full hyperparameter search ranges. Source in `appendix/supplementary.tex`; the same content is also in `appendix/robustness.md` and `appendix/search_spaces.md` |
| [`notes/`](notes/) | Analyses of open questions, kept separate from the paper's claims |
| [`code/`](code/) | `analysis/` reproduction scripts, `model/` the energy model, `baseline/` the linear baselines, `figures/` the figure scripts |
| [`data/`](data/) | The measurement series, the 12 hyperparameter-search databases, feature importances and derived tables. See [`data/README.md`](data/README.md) |

## Reproducibility status

The paper's energy and score results were recomputed from the data in this repository. The
following reproduce exactly:

* all 48 energy cells of the offline-inference table (Table III), both devices, both selections;
* the median energy-aware savings of 72 % (Raspberry Pi) and 19 % (Jetson), their per-KQI split,
  and the count of configuration pairs whose energy rises instead of falling;
* the batch-1 to batch-32 energy factors of 5.2 and 17.6, that is savings of 81 % and 94 %;
* the full-test scores quoted in Section IV-A, for example LightGBM at 89.4 % average precision
  and 79.4 % F1 for stalling, and LSTM at R² 93.2 % with a mean absolute error of 5.9 s for
  buffer health;
* every calibrated parameter of the stage-cost energy model (Table IV), recomputed by running
  the archived model code against the measurement series;
* the confidence-interval and outlier-filter statements in the table footnote and the appendix.

One caveat applies to absolute energies throughout. The stored `tf_energy` field sums polled
power assuming an exact 500 ms step, while the meter actually delivered about 503.3 ms per
sample, so absolute joules carry roughly 1 % systematic uncertainty on top of the reported
confidence intervals. The bias is common to every series and largely cancels in the relative
quantities the paper's conclusions rest on. `MEASUREMENTS.md` section 1.1 works this through.

`REPRODUCE.md` gives the command for each. What is **not** reproducible from this repository is
listed there too, with the reason.

## What this data does and does not show

Please read [`EVALUATION_CAVEATS.md`](EVALUATION_CAVEATS.md) before quoting a score from this
work. In short: the dataset split assigns playback runs at random rather than videos, so almost
every test run plays a video that also occurs in training, and 18 % of test runs have a
byte-identical copy in training. The reported scores therefore describe monitoring of a known
video catalogue, not generalisation to unseen content. The energy results are unaffected by
this, since they do not depend on label quality.

Equally, the concurrent-load results are a projection. Only packet capture and dissection are
validated against held-out loads; full-pipeline energy and device capacity at many streams come
from the stage-cost model, and the heavy-predictor crossover depends on an assumption the
available measurements cannot settle. [`ENERGY_MODEL.md`](ENERGY_MODEL.md) is explicit about
which is which.

## The dataset

The traffic and player-state data are **not** redistributed here. They are the public YouTube
streaming dataset of Loh et al., 11,142 playback runs over 246 videos and 1,081 hours of
streaming. See the dataset references in the paper. This repository contains only the
measurements taken on top of it, the derived tables, and the code.

One finding about that dataset is worth passing on independently of this paper: it contains
1,161 pairs of byte-identical playback runs recorded under two run numbers. The pair list is in
`data/derived/`.

## Requirements

Python 3.10 or newer with NumPy, pandas, SciPy, scikit-learn, Matplotlib and pyarrow. The
analysis scripts in `code/analysis/` need nothing else. The energy model in `code/model/` and
the baselines in `code/baseline/` were written against the original directory layout; see the
notes in `data/README.md` and `ENERGY_MODEL.md` for the paths they expect.

## Licence and citation

Code is under the MIT licence, documentation and data tables under CC BY 4.0. See
[`LICENSE`](LICENSE) and [`CITATION.cff`](CITATION.cff).

## Acknowledgment

This work has been performed in the framework of the SUSTAINET-Advance project, funded by the
German BMFTR (16KIS2282, Univ. Würzburg).
