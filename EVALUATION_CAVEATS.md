# EVALUATION_CAVEATS.md — what the prediction scores do and do not establish

Companion document to *Energy-Aware Assessment and Energy Modeling for Encrypted Video KQI
Prediction at the Edge*. The paper's "Deployment Guidelines and Limitations"
section compresses the evaluation's limitations into a paragraph. This document gives the full
version: how the data was split, what that split does and does not hold out, which test runs are
duplicates of training runs, how large each configuration's test population actually is, how
transport protocol is distributed across the dataset, and one stored-metric inconsistency found
while reproducing the numbers.

Everything below is either recomputed from the released files or taken from
`journal_paper/pilots/PILOTS.md`, with the source named in each case. Where the data cannot settle a
question, the document says so instead of estimating. Because section, table and figure numbers
shift with the paper's layout, the paper is cited here by section and label names rather than by
number.

Traffic views use the paper's notation: **ALL**, **ALL>100B**, **UL**, **UL>100B** (code names
`all`, `gt_100`, `uplink_only`, `uplink_only_gt_100`). The two edge devices are the **Raspberry Pi**
and the **Jetson**. Selection roles are performance-first (*pf*, `performance`) and energy-aware
(*ea*, `efficiency`).

---

## 1. The split is a random split over runs, and it is exactly reproducible

The dataset holds **11,142 playback runs**. The split is a single permutation of the run IDs, taken
once with a fixed seed and sliced:

```python
SEED = 1234
RNG = np.random.default_rng(SEED)
NUMBER_OF_RUNS = 11143
RND_RUNS = RNG.choice(a=range(1, NUMBER_OF_RUNS), size=NUMBER_OF_RUNS-1, replace=False)
```

Those four lines are `master_thesis/src/helpers.py` **lines 11–14**. The slicing is

```python
runs = [RND_RUNS[0:9018], RND_RUNS[9018:10051], RND_RUNS[10051:11142]]
```

in `master_thesis/src/processing/preprocessing.py` **line 334** (`build_datasets_from_config`), and
identically at lines 160, 210 and 247 for the other dataset builders.

So: **train = the first 9,018 run IDs, validation = the next 1,033, test = the last 1,091** of a
`numpy` `default_rng(1234)` permutation of 1…11,142. `choice(..., replace=False)` over the full
range is a permutation, so no run appears twice and none is dropped.

Re-running those four lines reproduces the split exactly. Verified here: 9,018 / 1,033 / 1,091 runs,
11,142 distinct IDs spanning 1…11,142, first five test IDs `2961, 6206, 3322, 8289, 4764`. Anyone
can regenerate the three run-ID lists without any file from this project.

**Runs are assigned at random, not videos.** Nothing in the procedure groups runs by the video they
play, by recording date, or by campaign. That single fact is the origin of the two leakage
properties in §2 and §3.

Two further properties of the sample-level data, for completeness:

* Samples are one per second per run with a complete input history; missing values in sparse windows
  are forward-filled per run and incomplete leading rows dropped
  (`preprocessing.py:353`, `df.ffill().dropna()`). This is why per-run sample counts vary.
* Buffer-health ground truth is stored **normalised**: `bh` is masked where negative, clipped to
  [0, 120000] ms and divided by 120000 (`preprocessing.py:413`), so the target lives in [0, 1]. See
  §5 for the unit conversion this implies.

---

## 2. Video overlap: the test set contains essentially no unseen video

The 11,142 runs play **246 distinct videos** (one `videoid` per run in `application_data.csv`, for
all 11,142 runs). Recomputed against the reproduced split, using
`journal_paper/pilots/results/video_ids.csv`:

| | Value |
|---|---|
| Distinct videos in the dataset | **246** |
| Distinct videos in the training split | 245 |
| Distinct videos in the test split | **211** |
| Test runs whose video also occurs in training | **1,090 of 1,091 = 99.9 %** |
| Validation runs whose video also occurs in training | 1,033 of 1,033 = 100 % |

Exactly one test run plays the one video that training never sees.

### Consequence: a held-out-video subset of this test set does not exist

With a single run, there is no subset of the current test set on which "performance on unseen
videos" could be measured. It is not a matter of a wider confidence interval; the population is
empty. **Generalisation to unseen videos can only be estimated by retraining on a video-grouped
split**, which is what the pilot below does. No re-scoring of the paper's 96 configurations can
answer this question.

### What the retraining pilot found

`PILOTS.md` §4 (Pilot B) built two video-grouped splits of comparable size (9,003/1,039/1,099 and
9,014/1,035/1,092 runs, no video above 10 % of the validation or test runs, 0 % video overlap with
training) and refit LightGBM, XGBoost and a single decision tree with the hyperparameters tuned on
the random split. Against the random split:

* **Stalling: LightGBM loses 11 to 22 points of average precision** (92.0 % → 69.9 % on seed 0,
  80.8 % on seed 1), and 9 to 17 points of F1.
* **Buffer health: LightGBM loses 18 to 23 points of R²** (89.6 % → 66.9 % / 71.8 %), with MAE
  rising from 8.3 s to 14.5–15.3 s.
* The two *random* splits differ by only 1.6 points AP and 0.5 points R², so this is not split
  noise.
* ROC-AUC falls by only 1–1.6 points: ranking quality survives, precision at useful recall does not.

**Which conclusions survive the shift, and which do not:**

* **Stable: the large family gaps.** The ensembles stay 8–16 points ahead of the single decision tree
  on every split. Rules resting on gaps of that size are unaffected.
* **Not stable: the small ones.** LightGBM's significant lead over XGBoost on random splits
  (+1.2 and +0.8 points AP) disappears under held-out videos (+0.3 [−0.8, +1.3] and
  −0.3 [−1.0, +0.3]). Absolute score floors also do not survive: no model in the pilot reaches
  AP ≥ 85 % on unseen videos.

So the paper's scores describe **monitoring of a known video catalogue**. The energy results and the
energy–performance ordering are not affected by this — they do not depend on the absolute score
level — but any statement of the form "this configuration reaches X %" should be read as
catalogue-specific.

### Caveats on the pilot itself

It used **re-implemented** ALL-view features, not the paper's pipeline, so its random-split baseline
(AP 92.0 %) is not the paper's number (AP 89.4 %); the comparison that matters is the *difference*
between its own splits. It covers ALL only, trees only, one seed per split, 1 M training samples
rather than the full 2.57 M, and its bootstrap resamples runs rather than videos, which understates
the uncertainty of an unseen-video estimate. The video-grouped test sets hold only 32–38 videos,
which is why the two seeds differ by 11 points AP.

### A counting artefact to be aware of

`PILOTS.md` §2 records a trap. `stats_for_nerds.csv` also carries a `videoid` column, and taking the
most frequent ID per run from it yields **380** distinct IDs, 134 of which never appear in
`application_data.csv`. Those extra IDs are other videos the player showed *before* the labelled
segment; `video_traffic.csv` and the labels cover only the labelled segment, so they affect neither
features nor targets. **246 is the correct count; 380 is an artefact.** An earlier 98.9 % overlap
figure came from the same source — with the correct IDs the overlap is higher, not lower.

---

## 3. Duplicate runs: 18 % of the test set has a byte-identical copy in training

`PILOTS.md` §7.1 reports that the raw dataset contains **1,161 pairs of byte-identical runs** —
2,322 runs in total, all five CSV files sharing MD5 hashes under two different run numbers (for
example runs 576/5941 and 577/5942). 1,152 of the pairs are in the QUIC-2020 campaign.

This is reproducible without re-hashing the 50 GB dataset. Grouping
`journal_paper/pilots/results/pilot_c/protocol_per_run.csv` on
`[date, n_pkts, n_tcp, n_udp, ul_mean_size, dl_mean_size]` — the key used by `pilot_c.py:55` —
gives exactly **1,161 groups of size two and 8,820 singletons**, and nothing larger. Intersecting
those pairs with the reproduced split confirms:

| | Value |
|---|---|
| Byte-identical run pairs in the dataset | **1,161** |
| Test runs with an exact copy in training | **197 of 1,091 = 18.1 %** |
| Validation runs with an exact copy in training | **157** |

### What the duplicates do to the scores

From `PILOTS.md` §7.4, scoring the pilot's random-split LightGBM per subgroup of the 1,091 test runs:

| Subgroup | Runs | Stalling share | AP % | F1 % | R² % | MAE s |
|---|---:|---:|---|---|---|---|
| all | 1,091 | 3.75 % | 92.0 [89.9, 93.7] | 83.9 | 89.6 [88.6, 90.5] | 8.33 |
| **with a copy in training** | **197** | 8.65 % | **99.6 [99.3, 99.8]** | **97.5** | **97.5 [96.9, 97.8]** | 4.16 |
| QUIC-2020, no copy | 726 | 2.71 % | 86.3 [81.5, 89.9] | 77.3 | 84.4 [82.9, 85.8] | 9.80 |
| **all, no copy in training** | **894** | 3.26 % | **88.7 [85.9, 91.2]** | **80.2** | **88.7 [87.5, 89.7]** | 8.75 |

The 197 duplicated test runs are scored almost perfectly — AP 99.6 %, R² 97.5 % — because the model
saw those exact byte sequences during training. Removing them moves the pilot's full-test figures
from **AP 92.0 → 88.7** (F1 83.9 → 80.2) and **R² 89.6 → 88.7**.

Two things to keep straight about that 3.3-point AP effect. The two subsets are not paired, so their
confidence intervals overlap; and part of the raw difference is prevalence, since the duplicated runs
carry an 8.65 % stalling share against 3.26 % in the rest. But the effect is about the size of the
LightGBM-versus-XGBoost differences that the same pilots treat as significant, so it is not
negligible at the resolution at which model families are being compared.

### This is a property of the published dataset, and it warrants an erratum

The duplication is in the raw public dataset, not in this project's processing. Any study that
splits those runs at random inherits the leak. The pair list is cheap to regenerate (either by MD5
over the five CSVs per run, or by the grouping key above over `protocol_per_run.csv`), and
publishing it as an erratum contribution to the dataset is the appropriate response. Future splits
should keep each pair together; `pilot_c.py` already does this.

### What has **not** been done

**The paper's own per-configuration scores have not been recomputed on the deduplicated test set.**
The figures above come from a re-implemented LightGBM in the pilot, not from the paper's 96
configurations. The blocker is mechanical rather than conceptual: the stored prediction arrays in
`data_masters_thesis/general_data/{stalling,buffer_health}_predictions/` are flat, concatenated
arrays with **no run index**, so there is currently no way to say which rows belong to the 197
duplicated test runs.

The mapping does exist in principle. `preprocessing.py` writes, next to each dataset, an index array
of shape `(n_runs, 3)` holding `(run, offset, length)` — `index_data` at lines 343/360, saved at line
374 — but those index files were not copied into the released data. Until they are available, or
until the concatenated arrays are re-aligned to run boundaries, the paper's headline scores should be
read as **including** the duplicate leak, with the pilot's roughly −3 points AP and −1 point R² as
the best available estimate of its size.

---

## 4. Test population sizes differ by up to 21 % across configurations

Each of the 96 configurations stored its own test predictions, and the arrays are not the same
length. Counting rows directly in
`data_masters_thesis/general_data/{stalling,buffer_health}_predictions/*_predictions.npy`:

**Stalling** († = not a multiple of 128)

| Family | ALL pf | ALL ea | ALL>100B pf | ALL>100B ea | UL pf | UL ea | UL>100B pf | UL>100B ea |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| RF | 316,964 † | 316,964 † | 316,964 † | 316,964 † | 316,964 † | 316,964 † | 316,964 † | 316,964 † |
| LGBM | 316,964 † | 316,964 † | 316,964 † | 316,964 † | 316,964 † | 316,964 † | 316,964 † | 316,964 † |
| FFNN | 316,928 | 316,928 | 316,928 | 316,928 | 316,928 | 316,928 | 316,928 | 316,928 |
| CNN | 289,664 | 295,040 | 284,288 | 295,040 | 289,664 | 300,544 | 289,664 | 300,544 |
| LSTM | 268,416 | 268,416 | 258,304 | 268,416 | 268,416 | 258,304 | 258,304 | 273,664 |
| TF | 268,416 | 311,424 | 306,048 | 311,424 | 273,664 | 311,424 | 263,296 | 300,544 |

**Buffer health** (all 48 are multiples of 128)

| Family | ALL pf | ALL ea | ALL>100B pf | ALL>100B ea | UL pf | UL ea | UL>100B pf | UL>100B ea |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| RF | 313,216 | 313,216 | 313,216 | 313,216 | 313,216 | 313,216 | 313,216 | 313,216 |
| LGBM | 313,216 | 313,216 | 313,216 | 313,216 | 313,216 | 313,216 | 313,216 | 313,216 |
| FFNN | 313,216 | 313,216 | 313,216 | 313,216 | 313,216 | 313,216 | 313,216 | 313,216 |
| CNN | 254,592 | 254,592 | 254,592 | 254,592 | 254,592 | 291,456 | **249,600** | 307,712 |
| LSTM | 254,592 | 254,592 | 254,592 | 254,592 | 259,712 | 254,592 | 254,592 | 254,592 |
| TF | 259,712 | 307,712 | 254,592 | 307,712 | 269,952 | 307,712 | 269,952 | 264,832 |

**Range: 249,600 to 316,964 samples, i.e. the smallest population is 21.3 % below the largest** (the
largest is 27.0 % larger than the smallest). There are 20 distinct population sizes among the 96
configurations.

### Two mechanisms, not one

The paper's "Scoring Protocol & Baselines" paragraph attributes the varying counts to input histories: sequence models use 5–60
seconds of preceding traffic, so they lose the initial seconds of each run. That is correct and is
the dominant effect — it is why the LSTM, CNN and transformer counts sit between 249,600 and 311,424
and vary with the selected history length.

But it is **not the only mechanism**, and the stored arrays show the second one plainly. **All 48
buffer-health arrays and 32 of the 48 stalling arrays are exact multiples of 128**; the 16 that are
not are precisely the stalling tree arrays (RF and LGBM, all views, both roles), all at 316,964. The
neural runs dropped the final partial batch (batch size 128, `drop_last`).

The clearest single case is the **stalling FFNN**. An FFNN uses no input history, so it should score
the same population as a tree: 316,964. It stores **316,928**, which is exactly 36 fewer — and
316,964 mod 128 = 36. The missing samples are one truncated final batch, nothing else. The paper
itself records this count ("316,964 stalling samples (316,928 for FFNNs)") without naming the cause.

The buffer-health trees show the mirror image: 313,216 is divisible by 128, so the buffer-health
FFNN loses nothing and matches the trees exactly. Their being multiples of 128 is coincidence, not
truncation.

### Consequence

Cross-family score comparisons in the full-test score figure (`fig:perf`) and in the text are made on **populations that differ by
up to 21 %**. Within a family and view the comparison is clean, and the pf-versus-ea pairs that the
energy analysis rests on are mostly like-for-like. Across families they are not: the tree-model
scores and the sequence-model scores describe different sets of seconds, with the sequence models
systematically missing the early, lower-buffer part of each run. For stalling this matters more than
for buffer health, because stalling prevalence is not uniform over a playback run.

What would fix it is a re-score on the common intersection of all 96 populations, which needs the
run-index mapping discussed in §3. It is not available, so the differences cannot be corrected or
even bounded here. Treat small cross-family score differences — a point or two — as not resolved by
this evaluation.

---

## 5. One stored-metric inconsistency, and which source to trust

While reproducing the buffer-health scores, the `r2` and `mae` fields inside
`general_data/buffer_health_predictions/*_results.json` were found to disagree with a direct
recomputation from the stored `*_predictions.npy` and `*_ground_truth.npy` arrays. Checked across
all 48 buffer-health configurations:

* **33 of 48** disagree by more than 0.01 points of R².
* The largest disagreement is **1.251 points** of R²: `buffer_health_gt_100_ffnn_efficiency` stores
  `r2 = 84.193 %`, while a plain recomputation gives 82.942 %.
* MAE disagrees by up to **1.65 s** (`buffer_health_all_transformer_efficiency`: 12.72 s stored
  against 14.37 s recomputed).

**The cause is clipping, and the stored fields are the correct ones.** The `*_predictions.npy`
arrays hold the models' **raw, unclipped** outputs — across all 48 configurations they span −0.64 to
+2.67 in normalised units, while the target is confined to [0, 1] by construction (§1). The stored
metrics were computed after clipping predictions to the valid range. Clipping the arrays to [0, 1]
before scoring reproduces **all 48** `r2` fields to within 1 × 10⁻⁵ points and all 48 `mae` fields to
within 1 × 10⁻⁵ s — zero mismatches. Verified on, among others:

| Configuration | stored `r2` | raw recomputation | clipped to [0, 1] |
|---|---|---|---|
| `buffer_health_gt_100_ffnn_efficiency` | 84.193 % | 82.942 % | **84.193 %** |
| `buffer_health_all_transformer_efficiency` | 77.238 % | 76.278 % | **77.238 %** |
| `buffer_health_gt_100_ffnn_performance` | 80.682 % | 80.404 % | **80.682 %** |
| `buffer_health_all_lgbm_performance` | 90.726 % | 90.680 % | **90.726 %** |
| `buffer_health_all_lstm_performance` | 93.219 % | 93.219 % | 93.219 % |

**The paper's values match the stored fields.** The clearest check is the coinciding ALL>100B FFNN
configuration retrained for the two selection roles, which the paper's configuration-selection paragraph reports as
R² = 80.7 % and 84.2 %: those are the stored 80.682 % and 84.193 %, not the raw-array 80.404 % and 82.942 %. The
headline LSTM figure (93.2 %, MAE 5.9 s) is identical under both routes, because that model's
predictions barely leave the valid range.

**Guidance for re-users:**

* **Trust the `*_results.json` fields**, or recompute with predictions clipped to [0, 1]. Scoring
  the raw arrays without clipping understates R² by up to 1.25 points and overstates MAE by up to
  1.65 s, and it is not what the paper reports.
* Clipping is also the defensible choice on its own terms: a negative or above-120 s buffer-health
  prediction is out of range by definition, and clipping is what a deployment would do.
* Stalling is unaffected. The 48 stalling prediction arrays lie within [0, 1] already, and
  recomputed average precision matches every stored `ap` field to within 1 × 10⁻⁵ points.

### Units: buffer health is stored normalised

The target is `bh` masked where negative, clipped to [0, 120000] ms and divided by 120000, so both
ground truth and predictions are in [0, 1] where 1 = 120 s (`preprocessing.py:413`). Therefore:

* **A mean absolute error computed in stored units must be multiplied by 120 to read as seconds.**
  The paper's "MAE 5.9 s" for the ALL buffer-health LSTM-pf corresponds to 0.0488 in stored units.
* R² is scale-invariant, so it needs no conversion.
* The `average` and `median` fields in the JSON files are the **ground truth** mean and median of
  that configuration's scored population, already in seconds (verified: 70.7853 s and 72.1005 s for
  `buffer_health_all_lstm_performance`, matching 120 × the array mean and median).
* Because the target is clipped at 120 s, every buffer health above two minutes is stored as exactly
  1.0. Errors in the well-buffered regime are therefore not measurable, and MAE is dominated by the
  part of the range the clip leaves intact.

---

## 6. Transport protocol: the dataset is two campaigns, and protocol is confounded with both

From `PILOTS.md` §7 and `results/pilot_c/protocol_summary.json`, labelling each run by the majority
transport protocol of the packets its features are computed from
(`payloadProtocolNumber` 6 = TCP, 17 = UDP/QUIC):

| | TCP | QUIC | mixed | total |
|---|---:|---:|---:|---:|
| All raw runs | **1,307** | **9,835** | (22 within) | 11,142 |
| 2018 campaign (runs with samples) | 1,287 | 336 (Jan–Mar 2018) | — | 1,624 |
| 2020 campaign (runs with samples) | 16 | 9,480 | — | 9,518 |

The labels are clean: 9,632 runs are above 99 % UDP, 1,292 below 1 % UDP, and only 22 fall between
20 % and 80 % (excluded from the protocol experiments). So the common shorthand "2018 = TCP, 2020 =
QUIC" is only approximately right: **336 of the 2018 runs are QUIC** (gQUIC, January to March 2018),
and the 2020 campaign is QUIC apart from a handful of runs. One internal inconsistency to note:
`PILOTS.md` prose dates the TCP runs to April–October 2018, while `protocol_summary.json` reports a
TCP date range starting 2018-02-21; the per-run file `protocol_per_run.csv` is the authority for any
exact boundary.

### Protocol is confounded with video catalogue and recording period

* The TCP-2018 runs play **13 videos**, three of which account for 1,254 of the 1,287 runs.
* The QUIC-2018 runs play exactly those three videos.
* The QUIC-2020 runs play 233 videos, **none** shared with any 2018 run.

So a TCP-versus-QUIC-2020 comparison is simultaneously an unseen-video comparison and a
different-year comparison. The only video-controlled protocol contrast available is TCP-2018 against
QUIC-2018, on three videos.

### A model trained on one protocol collapses on the other

In the controlled 2018 comparison (same three videos, same year, matched training size), cross-protocol
testing drives LightGBM's stalling AP to **4.4–7.4 %** — at or barely above prevalence — F1 to 2–5 %,
and buffer-health R² to between **−125 % and +28 %**, with MAE of 30–50 s against 3–9 s
within-protocol. Paired on identical test runs, the loss is **−67.5 points AP** on QUIC-2018 and
**−77.5 points AP** on TCP-2018.

This is not a prevalence artefact: at most about one fifth of any cross-protocol AP drop is
attributable to prevalence, and ROC-AUC — which does not depend on prevalence — falls to 62–93 %.
Re-tuning the threshold on the target protocol recovers at most 17 points of F1, still below 50 %, so
it is not miscalibration either. The mechanism is in the features: they use transport payload length,
so pure TCP ACKs are 0 B while QUIC ACK-only packets are 30–50 B, and even the full-size downlink
payload differs by a constant offset (1,368 B TCP against 1,358 B QUIC). Trees split on those exact
values. Dropping the ≤ 100 B features does not help, because the > 100 B sizes shift too.

It is also not a pure TCP-versus-UDP effect: a QUIC-2020 model fails on QUIC-2018 runs about as badly
as the TCP model does (AP 5.8 % against 4.4 %). The shift is a **capture-campaign and transport-stack
shift** that includes, but is not limited to, the protocol change.

### A pooled model costs nothing — which is what the paper's models are

Training on both protocols matches the protocol-specific model on each protocol: all paired AP
differences lie within ±0.4 points (except +3.1 [0.0, 6.5] for QUIC-2018) and all R² differences
within ±0.5 points. The paper's split draws runs from both campaigns at random, so **the paper's
scores should be read as those of a pooled, both-protocol model**, and the per-subgroup scores in
`PILOTS.md` §7.4 confirm it has no protocol gap: after prevalence adjustment, TCP and QUIC-2020 are
equal (AP_adj 90.2 % against 92.8 %; ROC-AUC 99.35 % against 99.55 %).

That is the realistic deployment case, and it is the right way to read the numbers. What does not
follow from them is any claim about a monitor deployed on a transport stack absent from its training
data — there, by this evidence, scores collapse to near chance. Protocol coverage of the training
data, not protocol-specific models, is what the result argues for; and a deployment can at least
detect the condition cheaply, since the UDP share per flow is trivial to compute.

### Caveats on the protocol pilot

Re-implemented ALL-view features, one seed per split, hyperparameters tuned on the random split,
training capped at 0.12–0.97 M samples. The 2018 subsets are small and video-poor (three videos;
QUIC-2018 has 336 runs, 85 test runs and about 500 stalling test samples), so those intervals are
wide. Bootstrap intervals resample runs, not campaigns — with effectively two campaigns the
cross-protocol result is n = 2 at the campaign level, however tight the run-level intervals look.

---

## 7. Summary: how to read the paper's scores

| Reported as | Should be read as |
|---|---|
| Full-test AP / F1 / R² / MAE | Scores of a pooled, both-protocol model on a **known video catalogue**, on a test set in which **18 % of runs have a byte-identical copy in training** |
| Comparisons within a family and view (pf against ea) | Sound; same population, same pipeline |
| Comparisons across families | Made on populations differing by up to **21 %**; differences of 1–2 points are not resolved |
| Differences of 1–2 points between similar families | Fragile — they do not survive a video-grouped split, and are about the size of the duplicate-leak effect |
| Large gaps (ensembles against a single tree; nonlinear against linear) | Stable across both the video-grouped and the duplicate-free checks |
| Absolute score floors ("reaches AP ≥ 85 %") | Catalogue-specific; not met by any model on unseen videos in the pilot |
| Buffer-health MAE in stored units | Multiply by 120 s; and recall the 120 s clip on the target |
| All energy results | Unaffected by everything in this document — they do not depend on the absolute score level. See `MEASUREMENTS.md` for their own caveats |

Open items that would close these gaps, none of which needs new hardware measurements: the run-index
arrays (`index_*.npy`) that `preprocessing.py` already produces, which would allow the paper's own 96
configurations to be re-scored duplicate-free, with run-level bootstrap intervals and per-campaign
subgroups; and a re-score on the common intersection of the 96 test populations. A video-grouped
evaluation of the paper's own models remains out of reach without retraining.
