# MEASUREMENTS.md — energy measurement protocol

Companion document to *Energy-Aware Assessment and Energy Modeling for Encrypted Video KQI
Prediction at the Edge*. It documents how every energy number in the paper was
obtained: the meter, the device settings, the three workloads, the idle baselines, the stored file
schema, the statistical treatment, and the exclusions that a re-user needs to know
about.

The paper's condensed version of this is the methodology subsections "Edge Execution and
Evaluation Workloads" and "Energy Measurement and Statistics".

Because table and figure numbers shift with the paper's layout, this document refers to them by
their LaTeX labels: `tab:all` (the ALL-view offline benchmark table), `tab:shortlist` (the live
candidates), `tab:model` (the calibrated energy-model parameters), `fig:perf` (full-test scores),
`fig:offline` (W1 energy against score) and `fig:live` (live energy differences).

File paths in this document are given relative to `measurements/` in the released measurement tree
(`data_masters_thesis/measurements/` in the source tree); this repository ships the series under
`data/` (see `data/README.md`). Measurement code paths refer to the original experiment code
(`master_thesis/src/`), which is not part of this repository.

Traffic views use the paper's notation: **ALL** (all video-flow packets), **ALL>100B**, **UL**
(uplink only), **UL>100B**. In file and directory names these appear as `all`, `gt_100`,
`uplink_only`, `uplink_only_gt_100`. The two edge devices are the **Raspberry Pi** (Raspberry Pi 4
Model B, 8 GB, CPU only) and the **Jetson** (Jetson Orin Nano Super, 8 GB shared memory, integrated
Ampere GPU). Model selection roles appear as `performance` (performance-first, *pf*) and
`efficiency` (energy-aware, *ea*) in file names.

---

## 1. The meter

All reported energy is **whole-device AC-side energy**, measured with a Tinkerforge Energy Monitor
Bricklet. The measurement chain is:

| Element | Detail |
|---|---|
| Sensing | Current clamp plus voltage transformer, both feeding the Bricklet |
| Measurement point | The AC input of the device under test, upstream of its power supply |
| Integration | The Bricklet integrates over ten mains cycles, i.e. **200 ms** per internal measurement |
| Resolution | **10 mW** |
| Polling | The host reads `get_energy_data()` every **500 ms** (`monitoring/measure_tinkerforge.py`, `interval = 0.5`) |
| Reported fields | Real power and the Bricklet's own accumulated energy register, both divided by 100 by the reader |

What this implies, and what it does not:

* The figures include **power-supply conversion losses** and the **device's idle draw**: everything
  the deployed monitor pulls from the wall, not just the SoC. For the Jetson the difference is
  directly visible in the idle series, which carries both meters: AC-side 4.273 W against 3.468 W on
  the board's own total-power rail and 0.560 W on the `VDD_CPU_GPU_CV` rail
  (`idle_baseline/jetson_nano_orin_idle_baseline_12h.json`).
* **Idle energy is not subtracted** anywhere in the paper. W1 reports the total energy for 100,000
  predictions and W2 the total energy for five minutes of monitoring, because that is the energy a
  deployment actually pays. Where the baseline share matters it is reported as a share, not removed
  (the paper's "Hardware Execution and Batching" subsection: median baseline shares of 56 % on
  the Raspberry Pi and 75 % on the Jetson).
* The meter sees the whole device, so **nothing in these series isolates a component**. Any
  statement about what a stage costs comes from differences between runs (W2 candidate minus
  reference, W3 with minus without dissection), never from a per-component reading.
* The Jetson's on-board INA sensors are recorded in parallel (`jn_*` fields, via
  `monitoring/measure_jetson_nano_orin.py` and `jtop`) and the Raspberry Pi has no equivalent. They
  are **not** used for any reported figure; they exist as a cross-check only, and they are not
  comparable to the AC-side values.

`tf_energy`, in joules, is the field the paper reports and the one the reproduction scripts
read. `tf_energy_wh` is the meter's own register at 0.01 Wh resolution, too coarse for W1 runs
of a few hundred joules.

## 2. Device settings

| Device | Setting |
|---|---|
| Jetson | **25 W power mode**, with `jetson_clocks` **not** applied, so CPU and GPU frequencies scale dynamically throughout every run |
| Raspberry Pi | **`ondemand`** CPU frequency-scaling governor |

Both devices therefore ran in their stock dynamic-frequency configuration rather than pinned to a
fixed clock. This is deliberate — it is how such a monitor would be deployed — but it has
consequences for interpretation:

* Short runs can see a different average frequency than long runs, which couples energy to runtime
  in a way a fixed-clock measurement would not.
* Frequency scaling and ONNX Runtime worker threads are the paper's stated hypothesis (in "Live
  Single-Stream Pipeline Monitoring") for the live-energy step that appears for performance-first candidates and for none of the
  energy-aware ones. That hypothesis is **not tested** by these measurements.
* ONNX Runtime ran with default thread settings on both devices, using the CPU execution provider on
  the Raspberry Pi and the CUDA execution provider on the Jetson. The exact thread settings used are
  not recorded in the released data; the only available statement is that all runs used the same
  settings.

Inference is in ONNX Runtime; tree ensembles are converted to tensor operations with Hummingbird,
except where noted in §8(a).

---

## 3. The three workloads

The pipeline is: replay workstation → switch → monitor (capture with a 96 B BPF snaplen → dissect
and filter → feature extraction → inference). Only the monitor is metered.

### W1 — offline inference (`measurements/model_execution/<device>/offline{,_fixed_batch}/`)

* 100,000 precomputed test samples, processed as fast as possible, to isolate model execution cost.
* The 100,000 samples are the **W1 subset**: drawn once from the test split with a fixed seed and
  shared by all models, so W1 scores are comparable across configurations but are not the full-test
  scores reported in `fig:perf`.
* Two batch sizes: **1** (`offline/`) and **32** (`offline_fixed_batch/`).
* 96 configurations × 2 devices × 2 batch sizes = 384 series, each of 30 repetitions
  (96 `*_stats.csv` files per device and batch size, verified by file count).

### W2 — live single-stream monitoring (`measurements/model_execution/<device>/live/`)

* The workstation replays one video stream with `tcpreplay`; the monitor captures with `tshark`,
  dissects, extracts features and predicts once per second.
* Predictions are issued only when the feature windows contain new packets, so a five-minute run
  issues fewer than 300 predictions (see §8(d)).
* Measurement interval: **five minutes** (stored `tf_duration` ≈ 299–300 s), inside replay traces of
  about six minutes that include guard time. Five minutes matches the median playback duration in
  the dataset.
* Each candidate has a paired **reference run** that follows the identical procedure but omits
  inference. Candidates live under `live/model_execution/...`, their references under
  `live/feature_extraction/...` at the same relative path. The pairing is one-to-one by path.
* The reference run **loads the ONNX model into memory but never invokes it**. This is stated in
  `data_masters_thesis/README.md` and is visible in the stored RAM columns: for the UL>100B stalling
  RF-pf the reference reports 1,172 MB of process memory against the candidate's 1,173 MB. The
  candidate-minus-reference energy difference therefore isolates the cost of the inference *calls*,
  not the cost of holding the model in memory — which is the right contrast for the paper's
  purpose, but means the reference is not a "feature extraction only" memory measurement.
* Two auxiliary live series exist and are not part of the main comparison:
  * `live/feature_extraction_ram/` — the **corrected RAM re-run**. Because the original reference
    runs also loaded the model (above), this series was re-measured without it to obtain the true
    feature-extraction memory footprint. For the same UL>100B stalling RF-pf it reports 128–129 MB
    against the 1,172 MB of the original reference, and it is the source of the paper's "130 MB for
    feature extraction alone".
  * `live/capturing_processing/` — Jetson only, two configurations, 30 repetitions each.
* Standalone feature-extraction series over stored traces, used to calibrate the model's
  feature-extraction term, are in `measurements/feature_extraction/<device>/` and cover window
  lengths of 1 s to 60 s and execution intervals of 0.5 s to 2 s, 30 repetitions each.

### W3 — capture under load (`measurements/capturing/<device>/`)

* Time-aligned streams merged with `mergecap` are replayed concurrently; the monitor captures with
  or without dissection. The load at which the capture rate declines is read from captured against
  expected packets.
* **Reliable traffic generation reaches about 200 concurrent streams.** At the 100-stream reference
  load the replay traces hold on average 3.6 million packets over about six minutes; the ALL,
  ALL>100B, UL and UL>100B views retain 100.0 %, 64.4 %, 35.4 % and 0.3 % of them.
* Directory naming: `header_only*` is the 96 B snaplen used throughout the paper, `complete_capture*`
  is full-packet capture, the `_dissection` suffix marks runs with dissection, and
  `header_only_save_to_{ram,disk}` are the two storage variants at 1 and 100 streams.
* Repetition counts vary by design: the main comparisons use 30 repetitions, intermediate loads used
  to locate the capture-rate decline or to test calibration transfer use **1–30** repetitions. The
  per-load repetition counts are visible directly in the CSVs. Examples: Raspberry Pi
  `header_only_dissection` has 30 repetitions at 1, 50–100 streams but a single run at 20, 30 and 40
  streams; Jetson `header_only_uplink_only_dissection` has 30 repetitions at 1 and 100 streams and
  1–9 runs at 150–400 streams. The single-run loads were screening runs; the energy-model validation
  uses only loads with at least 8 runs (see `ENERGY_MODEL.md` §6.1).
* Capture-only series (`header_only`) extend to 1,000 streams and some Jetson dissection series to
  400 streams, i.e. beyond the reliable replay range. Jetson loads above 200 streams are **not used**
  in the energy-model validation. Anyone reusing those high-load
  points should treat them as replay-limited rather than device-limited.

---

## 4. Idle baselines

Source files, twelve hours each, recorded with the device booted and otherwise unloaded:

* `measurements/idle_baseline/raspberry_pi_idle_baseline_12h.json`
* `measurements/idle_baseline/jetson_nano_orin_idle_baseline_12h.json`

Recomputed from the stored 500 ms power series (85,836 valid samples per device):

| | Raspberry Pi | Jetson |
|---|---|---|
| Duration | 43,201.2 s (12.0003 h) | 43,201.6 s (12.0004 h) |
| Mean of the power series | **2.5601 W** | **4.2729 W** |
| `tf_power_avg` field | 2.5601 W | 4.2729 W |
| `tf_energy` / `tf_duration` | 2.5434 W | 4.2449 W |
| Meter register (`tf_energy_wh`) | 2.5941 W (31.13 Wh) | 4.3215 W (51.86 Wh) |
| Median | 2.560 W | 4.230 W |
| 5th–95th percentile | 2.450–2.670 W | 4.020–4.580 W |
| Min / max | 2.30 / 4.20 W | 3.41 / 6.40 W |
| Global CPU utilisation (mean) | 0.09 % | 3.37 % |

**Both match the paper.** The paper states 2.56 W (Raspberry Pi) and 4.27 W (Jetson) over twelve
hours; the recomputed means are 2.5601 W and 4.2729 W, i.e. the paper's figures are the rounded
power-series means. The alternative estimators bracket them by about −0.7 % to +1.3 % (§1.1).

Two observations a re-user should carry along:

* The Jetson's idle floor is 1.67 W above the Raspberry Pi's and is 3.4 % of its own 25 W mode
  budget. The Jetson's measured idle CPU utilisation of 3.37 % against 0.09 % on the Raspberry Pi
  indicates that its idle is not a fully quiescent idle.
* The maxima (4.20 W and 6.40 W) show occasional background activity during the twelve hours. The
  medians sit at or below the means, so these excursions are short.

The energy model's baseline power `P_0` is **not** this idle figure: it is measured at one stream
with capture active (2.54 W and 4.35 W in `tab:model`) and is documented in `ENERGY_MODEL.md`.

---

## 5. File schema

Every measurement series is one CSV with a header and **30 data rows**, one per repetition. Opening
`measurements/model_execution/raspberry_pi/offline/stalling/all/lgbm/stalling_all_lgbm_performance_stats.csv`
gives 31 lines and these columns:

### Identification

| Column | Meaning |
|---|---|
| `timestamp` | Unix time at which the repetition was written |
| `config` | configuration name plus the repetition index and a short hash, e.g. `stalling_all_lgbm_performance_4-6b318f8c` — the number after the underscore is the repetition index and shows the randomised execution order |
| `tf_label`, `ul_label` | the configuration name without the repetition suffix |

### `tf_*` — Tinkerforge whole-device readings

| Column | Unit | Meaning |
|---|---|---|
| `tf_duration` | s | wall-clock duration of the metered interval (`perf_counter`) |
| `tf_energy` | **J** | `Σ power × 0.5 s` over the poll series — the field the paper reports |
| `tf_power_avg` | W | mean of the polled real power |
| `tf_power_min`, `tf_power_max` | W | extrema of the polled real power |
| `tf_energy_wh` | Wh | the Bricklet's own register, 0.01 Wh resolution (§1.1) |

### `ul_*` — process and global utilisation (`monitoring/measure_utilization.py`, 500 ms polling)

| Column group | Unit | Meaning |
|---|---|---|
| `ul_duration` | s | duration of the utilisation monitor's own interval |
| `ul_ram_process_{avg,min,max}` | MB | resident set size of the measured process |
| `ul_ram_global_{avg,min,max}` | MB | `psutil.virtual_memory().used` for the whole device |
| `ul_cpu_process_{avg,min,max}` | % | process CPU, **divided by the logical core count**, so 100 % means all cores saturated |
| `ul_cpu_global_{avg,min,max}` | % | device-wide CPU, `psutil.cpu_percent()` |

Note the two different CPU conventions: `ul_cpu_process_*` is normalised by core count and
`ul_cpu_global_*` is not. On the Jetson, additional `jn_*` columns carry the on-board rail power,
fan speed, GPU load, RAM and VRAM; they are informational only.

### Trailing columns — that configuration's prediction scores on the W1 subset

These are **not** full-test scores. They are the scores produced by the same run that was metered,
over the 100,000-sample W1 subset, and they are identical across the 30 repetitions of a
deterministic configuration.

| Workload | Columns |
|---|---|
| Stalling | `tn, fp, fn, tp, accuracy, precision, recall, f1, f1_macro, mcc, ap, pr_auc, f1_threshold, runtime, n_parameters` |
| Buffer health | `average, median, mae, medae, rmse, r2, runtime, n_parameters` |

`f1_threshold` is the classification threshold chosen on the validation split before testing, not
re-tuned here. `runtime` is the inference loop's own timing in seconds and is close to but not equal
to `tf_duration`, which brackets a slightly longer interval. For buffer health, `average` and
`median` describe the **ground truth** of the scored population, in seconds, and `mae`/`rmse` are
also in seconds.

### Live and capture series carry additional columns

W2 and W3 CSVs add replay and capture accounting: `actual, successful, failed, truncated,
retried_enobufs, retried_eagain` (the replay side) and `captured_packets, dropped_packets`. W2 adds
`processed_elements` and `predictions`. `dropped_packets` is `-1` where the counter was unavailable.
Above 100 streams the Jetson capture logs have no capture counter (`captured_packets` is `-1`);
there, `processed_packets` (the packets that reached dissection) is used instead, as in
`code/analysis/repro_w3.py`.

### The schema is device-dependent

The Jetson files carry the extra on-board `jn_*` columns, so the same workload has a wider schema
there: **60 columns against the Raspberry Pi's 38** for a W1 stalling series, and **53 against 31**
for a W1 buffer-health series. Anything that reads these CSVs must select columns by name, not by
position.

### Row counts, and why some series hold fewer than 30 rows

A row count other than 30 marks a series that is **not** one of the main
comparisons. The full picture for the live tree:

| Live directory | Files (both devices) | Rows per file |
|---|---|---|
| `live/model_execution/` | 49 | **30** |
| `live/feature_extraction/` | 49 | **30** |
| `live/capturing_processing/` (Jetson only) | 2 | 30 |
| `live/feature_extraction_ram/` | 47 | **5** |

Every file the paper's W2 results are computed from — the candidates in `live/model_execution/` and
their references in `live/feature_extraction/` — holds exactly 30 rows on both devices. The paper's
"30 repetitions" statement is therefore correct for the series it reports. The 47 five-row files are
the supplementary corrected-RAM re-run described in §3; five short repetitions are enough for a
memory footprint, and no energy figure in the paper comes from them.

Likewise, the W3 dissection series that hold 1 to 15 rows are the additional intermediate loads used
to locate the capture-rate decline and to test calibration transfer, which the paper itself states
use 1–30 repetitions (§3, W3). The main W3 comparisons at 1 and 100 streams hold 30 rows throughout.

---

## 6. Statistics

**Repetitions and ordering.** Every main comparison (W1, W2, and the main W3 loads) uses **30
repetitions**, executed in **randomised order** with different seeds and trace snapshots. The
randomised order is visible in the `config` column, whose repetition index does not follow file
order. The purpose is to keep thermal state and background activity from aligning with a particular
configuration; it does not remove those effects, it only stops them from being confounded with the
configuration under test.

**Central estimate and interval.** For each series the paper reports the **mean and a 95 %
confidence interval after excluding values beyond 1.5 interquartile ranges** from the quartiles. The
filter is applied per series, on the metric being reported.

**Relative differences.** For the two configurations selected within one family and view, the paper
reports signed relative changes
`ΔE = (E_ea − E_pf)/E_pf` and `ΔS = (S_ea − S_pf)/S_pf`, with negative values meaning reductions.

**Aggregation across families and views.** Differences aggregated over families and views are
summarised by **medians with bootstrap intervals over configuration pairs** (48 pairs per device).
Those intervals are explicitly **indicative only**: configurations from the same family share
structural properties across the four views, so the 48 pairs are not independent draws. Treat them
as a spread, not as a significance test.

**Live comparisons.** W2 candidate-minus-reference differences are assessed with **Welch's
*t*-test** on the respective 30 runs, which is the right choice here because candidate and reference
series have unequal variances.

**What the statistics do not cover.** Score comparisons across model families are made on
populations that differ in size by up to 21 % (see `FEATURES.md` §8), and run-level
bootstrap intervals exist only for the linear baselines, because the stored prediction arrays carry
no run index.

---

## 7. The outlier filter

The 1.5-IQR rule was re-applied independently to all 384 series. The reproduction confirms the
paper's appendix figures exactly.

**Batch 1 (`offline/`), 96 configurations × 30 repetitions × 2 devices = 5,760 values:**

| | Removed | Largest change in any series mean |
|---|---|---|
| Raspberry Pi | 123 / 2,880 (4.27 %) | **1.33 %** (ALL>100B buffer-health LSTM-pf) |
| Jetson | 58 / 2,880 (2.01 %) | **0.88 %** (ALL stalling LSTM-pf) |
| Total | **181 / 5,760** | |

So the filter removes about 3 % of the measurements and moves no reported batch-1 mean by more than
1.33 %. The paper's conclusions do not depend on it.

**Confidence-interval quality, batch 1:** the 95 % interval is **below 2 % of the mean for 97.9 %**
of the 192 configuration-device series (188 of 192), with a maximum of **3.27 %** (Raspberry Pi, ALL
stalling LSTM-pf). These figures back the footnote of `tab:all`.

**Batch 32 (`offline_fixed_batch/`) is noisier**, as expected from much shorter runs:

| | Removed | Largest change in any series mean |
|---|---|---|
| Raspberry Pi (excluding the 16 configurations of §8(a)) | 83 / 2,400 | **6.27 %** (UL buffer-health FFNN-ea) |
| Jetson | 200 / 2,880 (6.94 %) | **5.12 %** (UL buffer-health LSTM-ea) |

Including the 16 excluded Raspberry Pi stalling RF/LGBM configurations, the Raspberry Pi maximum
rises to 7.61 % and 111 of 2,880 values are removed. Batch-32 confidence intervals are below 2 % of
the mean for 88.0 % of series, with a maximum of 8.65 %. Batch-32 energies should therefore be read
with wider uncertainty than batch-1 energies; the batching factors the paper reports (5.2 on the
Raspberry Pi, 17.6 on the Jetson) are far larger than this noise.

---

## 8. Exclusions

### (a) Batched stalling RF and LGBM on the Raspberry Pi

At batch size 32 these 16 configurations (RF and LGBM, all four traffic views, both selection
roles) do not reproduce their unbatched predictions. They are therefore excluded from batched
accuracy comparisons, while their energy measurements are retained. No other configuration on
either device is affected: the confusion matrices of the remaining 80 Raspberry Pi and all 96
Jetson series are unchanged between batch sizes, apart from float-level differences on the
Jetson below 0.005 points.

### (b) Two live candidates were selected by mistake and are dropped

The W2 shortlist (`tab:shortlist`) should contain, per device, view and KQI, the highest-scoring, lowest-energy and
knee configurations from that device's W1 results. Two entries do not belong to any of those roles
and were included in error:

* Raspberry Pi: **UL buffer-health LGBM, energy-aware**
* Jetson: **UL stalling RF, performance-first**

They are dropped, leaving **23 candidates on the Raspberry Pi and 24 on the Jetson**. The stored
tree still contains 24 and 25 candidate directories respectively, so the two files must be filtered
out explicitly when reusing the data.

Recomputing all candidate-minus-reference differences (30 runs each, 1.5-IQR filtered means)
gives, on the Raspberry Pi:

* the 10 valid performance-first candidates: **+62.6 J to +99.1 J (8.4 % to 12.8 %)**;
* the 11 valid energy-aware candidates: **+0.6 J to +6.4 J (at most 0.86 %)**;
* the dropped UL buffer-health LGBM-ea: **+104.5 J (13.6 %)**, larger than any valid candidate.

On the Jetson no valid candidate exceeds **+14.5 J (1.13 %)**. `code/analysis/repro_w2.py` prints
these values.

Also worth knowing when reusing the shortlist: selecting candidates on validation results instead of
W1 results leaves the lowest-energy candidates unchanged, but changes three of the eight
highest-scoring candidates and four or five of the knee candidates per device.

### (c) Two UL>100B buffer-health sequence models issued almost no predictions

In W2, predictions are only issued when the feature windows contain new packets, and the UL>100B
view retains 0.3 % of packets. Two candidates are effectively silent, measured as the mean
`predictions` over their 30 runs:

| Candidate | Raspberry Pi | Jetson |
|---|---|---|
| UL>100B buffer-health **CNN-pf** | 0.2 | 0.1 |
| UL>100B buffer-health **TF-pf** | 42.6 | 38.3 |
| All other valid candidates | 246.2–272.5 | 238.0–270.8 |

Both are **excluded** from the live analysis and are drawn in grey in `fig:live`, since a
candidate that makes 0.1 predictions per five-minute run says nothing about inference energy. This
behaviour is expected for that view and is not a measurement fault. The valid range of 238–273
predictions per run is what the paper quotes.

### (d) The energy model assumes one prediction per second; live runs issue 0.8–0.9

Equation (1) uses `f = 1/s` for feature extraction and inference. The valid W2 candidates issue
238–273 predictions in a 299–300 s run, i.e. **0.80–0.91 predictions per second**. The model
therefore charges 10–20 % more inference and feature-extraction work per stream than the live runs
performed. The difference is small, and in the direction that **over**states per-stream cost, so the
projected capacities and crossovers are conservative rather than optimistic. It is not corrected
anywhere, and it should be remembered when comparing modelled against measured per-stream power.

---
