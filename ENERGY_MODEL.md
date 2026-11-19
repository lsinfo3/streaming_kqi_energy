# The stage-cost energy model

This document describes the additive energy model of the paper *Energy-Aware Assessment and
Energy Modeling for Encrypted Video KQI Prediction at the Edge*, Equation (1)
and Table IV. It covers what each parameter means, how it is estimated, which measurement
series it comes from, what its calibrated value is on each device, and what the model does and
does not support.

Two sources are used throughout and are kept apart deliberately:

* **Paper** — the value as printed in the paper (Table IV, Section III-F for the definitions or
  Section IV-F for the results).
* **Recomputed** — the value this document obtained by running the archived model code
  (`code/model/`) against the released measurement logs (`data_masters_thesis/measurements/`)
  while writing this file. Every recomputed value below was produced in a single run of
  `energy_model.py`, `uncertainty.py` and `audit.py`; the sections that state "verified" mean
  the recomputed value rounds to the printed one.

The model is a *projection* tool. Only its capture and dissection terms are validated against
measurements that were not used for calibration. Everything it says about concurrent
full-pipeline energy, about device crossovers, and about stream capacity is an extrapolation
from single-stream and stage-level measurements.

---

## 1. The equation

For a given device, traffic view and predictor *m*, the whole-device energy over a duration
*T* with *S* concurrent streams is

```
E(T, S) = T [ P0 + dP_m + S * p_S ]

p_S     = p_rx + r_v * c_v + f * (c_x + c_m)
```

| Symbol | Meaning | Unit |
|---|---|---|
| `E(T,S)` | whole-device energy at the AC input over the interval | J |
| `T` | duration of the monitoring interval | s |
| `S` | number of concurrent monitored streams | 1 (count) |
| `P0` | baseline power with capture active, independent of stream count | W |
| `dP_m` | live-monitoring offset of predictor *m* (an effective residual, see §4) | W |
| `p_S` | per-stream power contribution | W per stream |
| `p_rx` | packet reception cost per stream | W per stream |
| `r_v` | retained packet rate of traffic view *v*, per stream | packets/s per stream |
| `c_v` | dissection energy per retained packet | J per packet |
| `c_x` | feature-extraction energy per update (see §3.5 for the exact normalisation) | J per update |
| `c_m` | inference energy per prediction for predictor *m* | J per prediction |
| `f` | update and prediction frequency, fixed at 1 Hz | Hz |

The model is per device. `P0`, `p_rx`, `c_v`, `c_x`, `c_m` and `dP_m` are all device-specific;
`r_v` is a property of the replayed traffic and is shared by both devices.

### 1.1 What the code adds to the printed equation

The implementation (`code/model/energy_model.py`) is slightly more general than Equation (1),
and the extra machinery matters for reading the results:

* **Inference is a separate function.** `inference_power()` returns the inference share of power
  as `dP_m + S * f * c_m` in the default scenario A, which is exactly the `dP_m + S * f * c_m`
  implied by Equation (1). Scenarios B and C (§5) replace it with expressions that Equation (1)
  cannot express as written:
  * **B**: `max( A, min( ceil(S / batch) * dP_m + S * f * c_m , P_W1 - P0 ) )`, i.e. the offset
    recurs with each inference call and is capped by the measured W1 active power above
    baseline, and never falls below scenario A.
  * **C**: `S * f * c_m` for `S > 1`, i.e. the offset is dropped beyond one stream.
* **A residual scaling factor `k`** multiplies the capture term `S * (p_rx + r_v * c_v)`. It is
  1 in the point model; the uncertainty analysis draws it from the validation residuals (§6).
* **An optional concave capture term** `100 * k * (p_rx + r_v*c_v) * (S/100)^b` exists for a
  sensitivity check only (`b` fitted to 0.756). The paper keeps the linear term; see §9.
* **The reported quantity** in the scaling results is energy per stream-hour,
  `e(S) = 3600 * P(S) / S`, and a *crossover* is the smallest `S` at which the Jetson's `e(S)`
  falls below the Raspberry Pi's.

---

## 2. Calibrated parameters at a glance

| Parameter | Raspberry Pi 4 (paper) | Jetson Orin Nano Super (paper) | Recomputed (Pi / Jetson) | Status |
|---|---|---|---|---|
| `P0` baseline power | 2.54 W | 4.35 W | 2.536 W / 4.351 W | verified |
| `p_rx` receive, per stream | 2.9 mW | 1.4 mW | 2.895 mW / 1.378 mW | verified |
| `c_v` dissection, per packet | 209 µJ | 93 µJ | 208.7 µJ / 92.5 µJ | verified |
| `c_x` per update, ALL | 24–38 (31) mJ | 12–18 (15) mJ | 23.6–37.6 (31.4) / 11.7–17.9 (15.2) mJ | verified |
| `c_x` per update, UL | 11–17 (15) mJ | 6–9 (8) mJ | 11.4–17.3 (15.4) / 6.0–8.6 (7.7) mJ | verified |
| `dP_m` live offset, light | 0 W | 0 W | 0.0018 W / 0.0009 W, both unresolved → 0 | verified |
| `dP_m` live offset, heavy | 0.279 W | 0 W | 0.2787 W / 0.0166 W (interval includes 0) → 0 | verified |
| `c_m` per prediction, batch 1 | 0.08–42 mJ | 0.3–25 mJ | 0.076–42.25 mJ / 0.313–24.98 mJ | verified |
| `c_m` per prediction, batch 32 | ≤ 15.3 mJ | ≤ 1.1 mJ | max 15.287 mJ / max 1.112 mJ | verified |
| `r_v` packets/s per stream | 99 / 64 / 35 / 0.27 for ALL / ALL>100B / UL / UL>100B | | 98.93 / 63.70 / 35.03 / 0.27 | verified |

Two views that Table IV omits for space are also calibrated in the code: `c_x` for ALL>100B is
13.4–21.2 (17.8) mJ on the Pi and 7.3–10.7 (9.3) mJ on the Jetson; for UL>100B it is
2.3–9.4 (8.3) mJ and 1.4–5.1 (4.4) mJ.

All series are IQR-filtered before averaging: runs whose W1/W2/W3 energy lies beyond 1.5
interquartile ranges are dropped, as in Section III-E of the paper. Parameters are computed
from per-run mean power (`tf_energy / tf_duration`), not from energy divided by the nominal
300 s run length.

---

## 3. How each parameter is estimated

### 3.1 `P0` — baseline power with capture active

**Series.** W3 capture *without* dissection, 1 stream and 100 streams, all four traffic views
(`measurements/capturing/<device>/header_only*/`).

**Procedure.** The code averages the per-run mean power over the four views at 1 stream
(`nd1`) and at 100 streams (`nd100`), derives `p_rx = (nd100 - nd1) / 99`, and then sets
`P0 = nd1 - p_rx`.

**Note on the definition.** `P0` is therefore the *zero-stream intercept* of the
capture-without-dissection line, not the measured single-stream power. Table IV's source
column ("W3, 1 stream") is accurate about the series but the single `p_rx` subtraction is
easy to miss. The difference is small (one `p_rx`, i.e. 2.9 mW / 1.4 mW).

**Sanity check.** The recomputed `P0` is within 0.08 W of the twelve-hour idle baselines
(2.560 W / 4.273 W): below idle on the Pi, above it on the Jetson, consistent with the paper's
statement that single-stream capture adds no measurable power on the Pi and 0.08 W on the
Jetson.

### 3.2 `p_rx` — packet reception per stream

**Series and procedure.** As above: the 1-to-100-stream power difference of capture without
dissection, divided by 99, averaged over the four views. Averaging over views is deliberate:
at 100 streams, capture without dissection costs nearly the same in every view (847–856 J on
the Pi, 1354–1361 J on the Jetson over five minutes), so kernel filtering is treated as free
and `p_rx` carries no view dependence.

**Values.** 2.895 mW (Pi) and 1.378 mW (Jetson) per stream, bootstrap 5th–95th percentiles
2.86–2.93 mW and 1.33–1.43 mW.

### 3.3 `c_v` — dissection energy per retained packet

**Series.** W3 capture *with* dissection at 100 streams
(`measurements/capturing/<device>/header_only*_dissection/`), together with the measured
packet rates of the replay traces (`general_data/pcap_stats/stats_s*_r100.csv`).

**Procedure.** A least-squares slope of per-run mean power against the retained packet rate
`100 * r_v` across traffic views at 100 streams. On the Jetson all four views are used. On the
Raspberry Pi only UL and UL>100B are used, because the Pi already drops packets under ALL and
ALL>100B at 100 streams, so those two points do not represent the same processed workload.

**Caveat worth knowing.** With two views the Pi "slope" is a two-point slope through UL and
UL>100B, exactly determined and with no residual to inspect. The intercept of that fit is a
consistency check only and is not used by the model: 2.840 W against `P0 + 100*p_rx` = 2.825 W
on the Pi, and 4.481 W against 4.489 W on the Jetson.

**Values.** 208.7 µJ (Pi) and 92.5 µJ (Jetson) per retained packet, bootstrap
201.8–215.6 µJ and 90.0–95.2 µJ.

**Scale of the term.** At the ALL packet rate this is 20.6 mW per stream on the Pi and 9.2 mW
on the Jetson, i.e. the 21 mJ / 9 mJ per stream-second quoted in Section IV-F. Verified.

### 3.4 `r_v` — retained packet rate per stream

**Series.** The 30 replay-trace statistics files for the 100-stream overlays
(`general_data/pcap_stats/stats_s*_r100.csv`), field `pkts/s`, divided by the overlay factor
100.

**Values.** 98.93 / 63.70 / 35.03 / 0.27 packets per second per stream for ALL / ALL>100B /
UL / UL>100B. Verified against the paper's 99 / 64 / 35 / 0.27.

**One pitfall that was already fixed in the released code.** An earlier revision derived `r_v`
from total packets divided by `100 * 300 s`, which gives 120 / 77 / 43 / 0.33 packets/s. The
replay traces run for about 364 s, not 300 s, so that estimate was about 21 % too high and the
corresponding `c_v` about 17 % too low. The product `r_v * c_v`, which is the only thing the
model uses, is unchanged. If you compare against older notes, check which `r_v` basis they use.

### 3.5 `c_x` — feature extraction per update

**Series.** The stored-trace (offline) feature-extraction campaign,
`measurements/feature_extraction/<device>/offline/<view>/<view>_{1,10,60}s_feature_extraction_stats.csv`:
100 single-stream capture files processed back to back on one core, once per window length.

**Procedure.** `c_x = (E - P0 * T) / N`, i.e. the energy above baseline divided by the number
of stream-seconds processed. The nominal value is the 10 s window; the 1 s and 60 s windows
give the range used for uncertainty, and the uncertainty analysis draws the window position
uniformly between the 1 s and 60 s values.

**The normalisation is per stream-second, not per executed update.** `N` is taken from the
*60 s* window run's `total_output_elements` (27,985 for ALL) for all three window lengths,
because a 60 s window is almost never empty, so its update count equals the number of clock
ticks. Shorter windows skip empty windows and therefore emit fewer updates (21,449 with a 1 s
window for ALL, and only 4,141 for UL>100B at 1 s), but the per-second loop still runs. Table
IV labels `c_x` "per update"; the code computes energy per stream-second. The two coincide for
long windows and differ for short ones, most strongly for UL>100B. This is a labelling
mismatch between the paper and the code, not a numerical disagreement with the released
values.

**Values (mJ per stream-second).**

| View | Pi 1 s / 10 s / 60 s | Jetson 1 s / 10 s / 60 s |
|---|---|---|
| ALL | 23.6 / 31.4 / 37.6 | 11.7 / 15.2 / 17.9 |
| ALL>100B | 13.4 / 17.8 / 21.2 | 7.3 / 9.3 / 10.7 |
| UL | 11.4 / 15.4 / 17.3 | 6.0 / 7.7 / 8.6 |
| UL>100B | 2.3 / 8.3 / 9.4 | 1.4 / 4.4 / 5.1 |

At one update per second and the ALL view, `c_x` is the largest projected per-stream
contribution on both devices — larger than dissection (21 / 9 mJ) and larger than the heavy
predictor's inference (18 / 9 mJ). This is the paper's central scaling claim and it rests
entirely on this single campaign.

### 3.6 `c_m` — inference energy per prediction

**Series.** W1 offline inference, `measurements/model_execution/<device>/offline/...` for batch
size 1 and `.../offline_fixed_batch/...` for batch size 32; 100,000 predictions per run.

**Procedure.** `c_m = (E_W1 - P0 * T_W1) / 100,000`, per device and batch size. The baseline is
removed with the same `P0` as everywhere else, so `c_m` is the *incremental* energy of a
prediction, not its whole-device share.

**Values.** Over the valid live candidates (those that actually issued predictions in W2),
recomputed ranges are 0.076–42.25 mJ on the Pi and 0.313–24.98 mJ on the Jetson at batch size 1,
and at most 15.287 mJ and 1.112 mJ at batch size 32. These round to the paper's 0.08–42 mJ,
0.3–25 mJ, ≤ 15.3 mJ and ≤ 1.1 mJ. Verified.

Two details about the batch-32 range: the Pi's batched stalling RF and LGBM are excluded
because their batched outputs differ from their unbatched ones (Section IV-C), and the smallest
batch-32 value on the Pi comes out marginally negative (−0.003 mJ for the energy-aware
UL>100B buffer-health LGBM), i.e. indistinguishable from zero once the baseline is removed.

**The three model classes used for projections.** The model does not project all candidates. It
uses one *light* and one *heavy* representative per device, plus a batched variant of the heavy
one:

| Class | Raspberry Pi | Jetson | `c_m` Pi / Jetson |
|---|---|---|---|
| light | energy-aware buffer-health RF, ALL | energy-aware buffer-health FFNN, ALL | 0.096 mJ / 0.405 mJ |
| heavy | performance-first buffer-health LSTM, ALL | performance-first buffer-health LSTM, ALL | 18.000 mJ / 8.910 mJ |
| heavy, batch 32 | same model, batch size 32 | same model, batch size 32 | 5.701 mJ / 0.766 mJ |

The W1 active power of these runs, which caps scenario B, is 4.532 W (Pi heavy), 4.557 W (Pi
heavy batched), 3.789 W (Pi light), 6.769 W (Jetson heavy), 9.056 W (Jetson heavy batched) and
5.327 W (Jetson light).

---

## 4. The live-monitoring offset `dP_m`

This is the most delicate part of the model and the part most easily misread.

### 4.1 What is measured

W2 measures a complete single-stream pipeline for five minutes, once with inference
(`live/model_execution/...`) and once with an otherwise identical reference run without
inference (`live/feature_extraction/...`), 30 runs each. The raw observable is the power
difference between the two, which the paper's Section IV-D calls the live increment:

| | Pi heavy | Jetson heavy | Pi light | Jetson light |
|---|---|---|---|---|
| observed W2 increment | 0.2947 W | 0.0246 W | 0.0019 W | 0.0013 W |

### 4.2 What is subtracted

The predictions the live run actually issued are already accounted for by `c_m`, so counting
the whole increment as an offset would double-count them. The retained offset is the
*residual*

```
dP_m = max(0, (P_W2,inference - P_W2,reference) - rate * c_m)
```

where `rate` is the prediction rate the live run actually achieved (predictions divided by run
duration) and `c_m` is the **unbatched** W1 incremental energy per prediction. Power and
prediction rate are taken from the same runs and are bootstrapped **together**, so the
subtraction keeps the pairing between a run's power and the number of predictions it made.
10,000 bootstrap draws (screening seed 2026 in the released code) produce an interval, and an
offset is retained **only when the bootstrap 95 % interval lies entirely above zero**;
otherwise it is set to zero and reported as unresolved.

### 4.3 The resulting values

Recomputed with the released code (`audit.py`, which reproduces the stored
`audit_results.json` exactly):

| Device / class | observed increment | actual prediction rate | residual | bootstrap 95 % interval | used in the model |
|---|---|---|---|---|---|
| Pi, heavy | 0.2947 W | 0.891 Hz | **0.2787 W** | 0.2317 – 0.3225 W | 0.2787 W |
| Jetson, heavy | 0.0246 W | 0.890 Hz | 0.0166 W | −0.0003 – 0.0335 W | 0 (interval includes zero) |
| Pi, light | 0.0019 W | 0.848 Hz | 0.0018 W | −0.0306 – 0.0346 W | 0 (unresolved) |
| Jetson, light | 0.0013 W | 0.881 Hz | 0.0009 W | −0.0156 – 0.0168 W | 0 (unresolved) |

This matches the paper exactly: only the heavy predictor on the Raspberry Pi retains an
offset, 0.279 W with interval 0.2317–0.3225 W; the Jetson's 0.0166 W is retained as zero
because its interval only just includes zero; and both light offsets are unresolved. Verified.

The batched heavy class deliberately reuses the *unbatched* residual and substitutes the
batch-32 `c_m`; `audit.py` asserts this.

### 4.4 This is an effective decomposition, not a causal attribution

`dP_m` is a residual: whatever part of the measured live increment is not explained by the
per-prediction energy of the W1 benchmark. It is not a measurement of any identified hardware
mechanism. Section IV-D of the paper points at plausible contributors — runtime threads,
frequency scaling, the fact that the ten performance-first Pi candidates add similar energy
despite a 50-fold spread in per-prediction cost — but nothing in the data isolates them. Three
consequences follow:

1. The split of the live increment between `dP_m` and `S * f * c_m` is a modelling choice, not
   an observation. Only their sum at one stream is measured.
2. Because it is a residual, it absorbs any systematic difference between the W1 and W2
   campaigns, including thermal and campaign offsets.
3. Nothing in the data says how `dP_m` behaves with more than one stream. That is the subject
   of §5, and it is the model's largest uncertainty.

---

## 5. The three scaling assumptions, and why they cannot be separated

`dP_m` was measured at one stream. The model therefore enumerates three scenarios rather than
picking one:

| Scenario | Assumption | Inference power at `S` streams |
|---|---|---|
| **A** (default) | the offset is paid once per device | `dP_m + S * f * c_m` |
| **B** | the offset recurs with each inference call, bounded by the measured W1 active power | `max(A, min(ceil(S/batch) * dP_m + S*f*c_m, P_W1 - P0))` |
| **C** | the offset disappears beyond one stream | `S * f * c_m` for `S > 1` |

All three coincide at `S = 1` for unbatched inference, which `audit.py` asserts; they are
indistinguishable by construction at the only load where the offset was measured.

### Effect on the heavy-predictor crossover (ALL view, nominal 10 s feature window)

Recomputed from `audit_results.json`, 10,000 draws, as median [5th, 95th percentile]:

| Scenario | ALL view | UL view |
|---|---|---|
| A — paid once per device | **42 [31, 57]** streams | 72 [57, 90] |
| B — recurring per inference call, capped at W1 active power | **6 [6, 7]** streams | 7 [6, 7] |
| C — absent beyond one stream | **49 [37, 67]** streams | 85 [68, 106] |

These are the paper's numbers: 42 streams (31–57) under A, 49 under C, and 6 to 7 under B.
Verified. (For reference, with `c_x` forced to zero the same three scenarios give 75, 7 and 88
streams for ALL.)

**This spread — 6 to 49 streams for the same predictor on the same hardware — is the single
largest uncertainty in the paper's scaling claims.** It is a factor of roughly eight, it is
wider than every other uncertainty in the model combined, and it is structural rather than
statistical: no amount of resampling the existing data narrows it. Under A and C the Jetson
becomes the cheaper monitor only at loads near or beyond the Pi's own capture capacity, which
supports "use the Pi for small deployments". Under B the Jetson is already cheaper at about
seven streams, which reverses the recommendation for heavy predictors at almost any realistic
load.

Scenario B is intended as a pessimistic bound rather than a likely physical model: the Pi's
0.279 W offset would recur with every call and the Pi reaches its measured W1 active power from
about seven streams, at which point the cap takes over. Note also that B's intervals are tight
here only because the Jetson's offset is zero after the §4 correction; in the earlier,
uncorrected accounting the Jetson also paid a non-zero offset per call and the B distribution
was bimodal (5th–95th percentiles as wide as 6–154 streams).

**What would settle it:** a multi-stream run of the complete pipeline with inference, which
does not exist in this dataset. Until then the three scenarios should be quoted together, and
any single crossover number taken from this model should name its scenario.

For completeness, the light predictors have no retained offset, so A, B and C are identical for
them, and their crossover is 65 [45, 101] streams for ALL and 150.5 [104, 233] for UL.

---

## 6. Validation

### 6.1 What is validated

W3 capture with dissection at intermediate loads that were **not** used for calibration. A
load counts as loss-free when its captured-to-sent packet ratio reaches 98 % of the loss-free
reference ratio for the same view. Only the capture terms (`P0`, `p_rx`, `r_v * c_v`) enter
this comparison; feature extraction and inference are not active in W3.

Recomputed error ranges, `(model − measured) / measured` on power:

| Device | loss-free intermediate loads | points |
|---|---|---|
| Raspberry Pi | **−9.3 % … +12.7 %**, median absolute error 5.9 % | 18 |
| Jetson | **+0.3 % … +4.3 %**, median absolute error 1.5 % | 11 |

These are the paper's figures. Verified. Three further facts belong with them:

* The extremes on the Pi are −9.3 % at 10 streams of full-packet capture and +12.7 % at 170
  streams under UL; both are 30-run means. The Pi's loss-free set *also* contains three
  single-run points (ALL at 20, 30 and 40 streams, errors −6.6 %, −3.4 %, −2.2 %), and the
  Jetson's contains two (UL at 150 and 200 streams, +3.5 % and +4.3 %). The paper's
  parenthetical "including two single-run measurements" matches the Jetson set; the Pi set has
  three such points, and the +4.3 % end of the Jetson range *is* one of them.
* The validation set includes the `FULL` (complete-capture) view, which is not one of the four
  traffic views of the paper; it is modelled with the ALL packet rate and the same `c_v`.
* Saturated and onset loads are excluded from the quoted range. There the model is too high by
  up to +22 % on the Pi, which is expected: the device is dropping packets and therefore doing
  less work than the model assumes. The code classifies loads as loss-free (≥ 0.98), onset
  (0.95–0.98) or saturated (< 0.95) and the model should not be used in the latter two regimes.

### 6.2 How the residuals re-enter the model

The uncertainty analysis propagates 10,000 draws that (a) bootstrap every primitive
measurement series independently, (b) draw the feature-extraction window uniformly between the
1 s and 60 s values, and (c) multiply the per-stream capture term by a factor `k` drawn from
the observed validation residuals of loss-free intermediate loads with at least 8 runs and
`S ≥ 20`. Recomputed, those factors are 14 values in 0.72–1.51 on the Pi and 9 values in
0.82–0.98 on the Jetson.

The one-at-a-time ranking confirms where the uncertainty lives: for the heavy ALL crossover
(base 41 streams at the window midpoint), the Pi residual factor `k` moves it to 48/36 and the
Pi feature window to 46/37, while every bootstrapped parameter (`c_v`, `p_rx`, `P0`, `dP_m`,
`c_m`, `c_x` run noise, `r_v`) moves it by at most one stream. The crossover is dominated by
the transfer of stage measurements to concurrent load, not by measurement noise.

### 6.3 What is *not* validated

* **Concurrent full-pipeline energy.** No measurement in this dataset runs capture, dissection,
  feature extraction and inference together for more than one stream. Every per-stream number
  that includes `c_x` or `c_m` beyond `S = 1` is a projection.
* **The capacity estimates.** 33–50 ALL streams on the Pi and 54–82 on the Jetson are derived
  from per-stage CPU time assuming one core per pipeline stage, not from a load test. The same
  budget gives 12–27 (Pi) and 33–48 (Jetson) streams if all stages share a single thread, and
  58 (Pi) and 120 (Jetson) if all core time is spread perfectly over all cores with dissection
  still single-threaded. The 33–50 figure is the one-core-per-stage range for the light and
  batched classes; the heavy class gives 28–50 under the same assumption, which explains the
  28–50 that appears in some working notes.
* **Device choice for heavy predictors**, for the reason given in §5.

### 6.4 The robust scaling claim

The paper's positive scaling claim is deliberately narrow: with light predictors the Pi stays
cheaper per stream through 44 ALL streams or 103 UL streams in at least 95 % of draws under all
considered offset assumptions. Recomputed, the claim holds in 95.5 % of draws at 44 ALL
streams and 95.1 % at 103 UL streams, in every structural scenario. Verified. The wider
statements that earlier drafts carried (Pi cheaper up to 70 ALL or 107 UL streams with the
heaviest model, or up to its capture capacity with light ones) fail that test, the first two in
0 % of draws under scenario A with feature extraction counted.

---

## 7. Re-running the model

### 7.1 Files and entry points

The model is three modules plus an audit driver:

| File | What it does |
|---|---|
| `code/model/logdata.py` | loaders for the per-run measurement logs; IQR filtering; no model logic |
| `code/model/energy_model.py` | `load_inputs()`, `calibrate()`, the model itself (`power`, `e_stream_hour`, `crossover`), the W3 validation table, the CPU-time budget and capacity estimates. Running it as a script prints the full point-estimate report |
| `code/model/uncertainty.py` | the 10,000-draw bootstrap and scenario analysis, the robustness labels, the one-at-a-time ranking, and the scaling figure |
| `code/model/audit.py` | re-derives the offset accounting with assertions and writes `audit_results.json` (offsets with intervals, crossover percentiles, nominal energies, batching savings, the one-at-a-time ranking) |

Commands, run from inside the model directory:

```
python3 energy_model.py     # point report
python3 uncertainty.py      # bootstrap, scenarios, figure; ~10 s after the inputs load
python3 audit.py            # writes audit_results.json beside the script
```

The v5 README states the audit command as `python model/audit.py`; the script itself imports
`uncertainty` as a top-level module, so it must be started with the model directory on
`sys.path` — run it from inside that directory, or set `PYTHONPATH` to it. Reading the
percentile arrays in `audit_results.json`: they are ordered 5th / 50th / 95th unless the key
says `CI95`, in which case they are the 2.5th and 97.5th percentiles of a bootstrap.

Requirements: Python 3, NumPy, pandas, SciPy (the concave sensitivity fit) and Matplotlib (the
figure). The code was exercised here with NumPy 1.26, pandas 2.2, SciPy 1.13 and Matplotlib
3.10.

### 7.2 Inputs, and what breaks outside the original machine

`logdata.py` resolves the data root as `<model directory>/../../data_masters_thesis`. That is
the only path configuration in the code and it is **relative to the model file's own
location**, so the model directory must sit exactly two levels below a directory that contains
`data_masters_thesis/`. In the artifact layout (`artifact/code/model/`) that resolves to
`artifact/data_masters_thesis`, which does not exist; create a symlink there to the measurement
tree, or run the code from a directory laid out like the original
`<root>/noms_paper/model/` beside `<root>/data_masters_thesis/`. Confirmed by running: with
that symlink in place, all three scripts complete against the released logs.

Within the data root, the model reads:

* `measurements/capturing/<device>/<view>[_dissection]/...` — W3, all loads
* `measurements/model_execution/<device>/offline/...` and `.../offline_fixed_batch/...` — W1
* `measurements/model_execution/<device>/live/{model_execution,feature_extraction}/...` — W2
* `measurements/feature_extraction/<device>/{offline,live}/...` — the `c_x` campaigns
* `measurements/processing/<device>/...` — deferred-processing cost (reported, not used in the model)
* `measurements/idle_baseline/<device>_idle_baseline_12h.json` — the idle cross-check
* `general_data/pcap_stats/stats_s*_r100.csv` — `r_v`

The measurement dataset is not redistributed with this repository; see the repository README for
where it comes from.

Two further portability notes, both verified by running the code:

* `uncertainty.py` writes its figure to `../figures/energy_model.pdf`. Matplotlib does not
  create missing directories, so `../figures/` must exist or the script ends with a
  `FileNotFoundError` after printing all of its results.
* The same function then writes a second copy as PNG to a **hardcoded absolute path inside a
  session scratchpad directory** from the original machine. That directory still happened to
  exist on the machine used for this write-up, so the script completed; on any other machine
  the line raises `FileNotFoundError`. It is the last statement of the figure routine, so all
  printed results and the PDF are already written when it fails, but the script exits non-zero.
  The line should be deleted before the code is published.
* `python figures_src/redraw.py --model-only`, the other command in the v5 README, needs the
  paper's figure sources, which are not part of the model directory.

### 7.3 What `audit.py` guarantees

Running it here printed "Offset accounting, shared batch residual and single-stream scenario
consistency passed." and regenerated a file identical to the stored `audit_results.json`. Its
assertions check that the residual plus the prediction work recovers the observed W2 increment
exactly, that the batched heavy class reuses the unbatched residual, and that scenarios A, B
and C agree at one stream. It does not check the model against any measurement; that is
`energy_model.validation()`.

### 7.4 A note on which copy of the code to use

The archive contains two versions of `energy_model.py`. They differ in exactly one respect,
the offset accounting of §4:

* the **v0** version sets `dP_m` to the raw W2 inference-minus-reference power difference and
  keeps it when a two-sided Welch test on the absolute difference clears zero. Run against the
  released data, it yields `dP_m` = 294.7 mW (Pi heavy) and 24.6 mW (Jetson heavy) — the
  Jetson offset is *retained* as significant.
* the **v5** version subtracts the actual prediction rate times the unbatched `c_m`, clips the
  residual at zero and retains it only when a paired bootstrap 95 % interval is strictly
  positive. This is the version the paper describes and the one that produces Table IV's
  0.279 W / 0 W.

`logdata.py` and `uncertainty.py` are byte-identical between the two. Only the v5
`energy_model.py` reproduces the paper, so that is the copy the artifact should ship. See §9
for why this matters.

---

## 8. Reading the projections correctly

A short list of conventions that are easy to get wrong when quoting this model:

* Numbers "per stream-hour" are `3600 * P(S) / S`, so they fall steeply with `S` simply because
  the shared baseline is divided by more streams. At one stream the baseline is 87–98 % of
  total power; comparing `e(1)` across devices says almost nothing about the pipeline.
* A crossover is defined on energy *per stream*, not on total energy. The Jetson never draws
  less total power than the Pi in this model.
* `c_x` and `c_m` are applied at `f = 1 Hz` per stream, so the per-stream cost of feature
  extraction and inference scales with the number of streams and the baseline does not. That
  asymmetry is what produces every crossover in the paper.
* Projections beyond a device's capture capacity (60 ALL streams on the Pi, 120 on the Jetson,
  by the loss-onset measurements) describe a monitor that is already dropping packets. The
  scaling figure in the paper shades that region for the Pi.

---

## 9. Known limitations of the model

1. **Feature-extraction cost comes from single-window offline runs.** `c_x` is measured by
   processing stored single-stream captures back to back on one core, one window length per run,
   in a pandas loop. The live W2 pipeline computes several windows per update with NumPy, and
   no measurement exists of simultaneous multi-window extraction under concurrent load. The
   CPU time per stream-second of the two paths agrees (Pi 30 / 23 / 16 / 2 ms and Jetson
   18 / 12 / 8 / 1 ms per stream-second for ALL / ALL>100B / UL / UL>100B in the W2 reference
   runs, against 20–31 ms and 12–19 ms for ALL offline), but energy per update under concurrent
   load is not measured. Since `c_x` is the largest projected per-stream term, this is the
   weakest link in the scaling argument after the offset scenarios.
2. **Campaign differences affect the capture calibration and cannot be separated from a
   load-dependent `c_v`.** Calibration points were measured in May/June 2026, the intermediate
   validation loads in October/November 2026. Within the later Pi campaign the marginal power
   per stream is 5.3 mW (UL, 110–170 streams) against the model's 10.2 mW, and 110 UL streams
   in October drew less power (3.480 W) than 100 UL streams in the calibration campaign
   (3.571 W). On the Jetson, the same capture-without-dissection configuration drew 4.239 W in
   October against 4.339 W in June at one stream, at a lower fan level (22 % against 25 %).
   A concave per-stream term (`S^0.756`) removes about 40 % of the Pi's residual RMS, but it
   cannot be distinguished from these campaign offsets, so the model stays linear and carries
   the residual as the factor `k`.
3. **The model assumes one prediction and one feature update per second, while the live runs
   issued 0.85–0.89 predictions per second** (predictions are emitted only when a feature
   window contains new packets). At `f = 1 Hz` the model therefore slightly overstates the
   inference and feature-extraction terms relative to the W2 behaviour. The offset accounting
   of §4 uses the *actual* rate, so the two are consistent at one stream, but the projection
   does not reduce `f`.
4. **Single-stream `c_x` could not be confirmed independently.** Comparing the W2 reference
   (features, no inference) against W3 single-stream capture with dissection gives a difference
   consistent with `c_x` only for Pi ALL (+3.2 J per 300 s, expected 7.1–11.3 J). For the other
   views and for the whole Jetson the difference is negative (down to −35 J) because campaign
   offsets of up to 35 J exceed the effect being measured.
5. **`p_rx` carries no view dependence and `c_v` on the Pi rests on two views**, as described in
   §3.2 and §3.3.
6. **`dP_m` is a residual, not a mechanism** (§4.4), and its scaling is unmeasured (§5).
7. **The capacity numbers assume one CPU core per stage and inference time somewhere between
   the W1 wall time and the W1 core time** (ONNX Runtime used four threads on the Pi, so
   single-thread inference time is unknown). They are not a measured throughput.
8. **The Pi's batched stalling RF and LGBM are excluded** from the batch-32 parameters because
   their batched outputs differ from their unbatched ones.
9. **Validation covers capture and dissection only.** Nothing in §6 tests the terms that the
   scaling conclusions actually depend on.

---

## 10. Verification summary

Verified by recomputation from the released measurement logs while writing this document:
`P0`, `p_rx`, `c_v`, `r_v`, all `c_x` values and ranges, `c_m` ranges at both batch sizes, all
four live offsets with their bootstrap intervals and the actual prediction rates, the three
crossover scenarios with their percentiles, the validation error ranges and their constituent
points, the residual factors `k`, the one-at-a-time ranking, the robustness shares behind the
44-ALL/103-UL claim, and the capacity ranges. `audit.py` reproduced the archived
`audit_results.json` without differences.

Taken from the paper without independent recomputation: the W1/W2/W3 protocol details
(repetitions, randomised order, meter configuration) and the per-view retained-packet
percentages of the replay traces quoted in Section III-D, both of which are measurement
documentation rather than model parameters.
