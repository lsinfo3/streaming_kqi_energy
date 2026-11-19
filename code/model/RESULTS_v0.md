# Energy model: results of N1–N3 (13 Nov 2026)

All numbers below come from `python3 energy_model.py` and `python3 uncertainty.py` (run in `noms_paper/model/`,
seed 2027, 10,000 draws, runtime about 20 s). Section VI (`sections/model.tex`) already uses them.

## 1. What changed in the model

1. **Feature extraction added (N1.1).** `c_x` per stream-second, calibrated per device and view from the offline
   feature-extraction logs (100 single-stream traces back to back on one core): `c_x = (E − P0·T) / stream-seconds`,
   1 s / 10 s / 60 s windows. Stream-seconds = feature updates of the 60 s window run (27,985 for ALL; shorter windows
   give fewer updates because empty windows are skipped: 21,449 with 1 s, so `total_output_elements` is *not* 30,000;
   the per-second loop runs anyway, so normalizing per stream-second is right, also for UL>100B with 4,141 updates
   at 1 s). Point value = 10 s window; range = 1–60 s; the uncertainty analysis draws the window uniformly.
2. **Power units and measured packet rate.** Calibration now uses per-run mean power (E/T with the logged duration)
   instead of E/300 s. `r_v` = packets per second and stream of the 30 replay traces (pcap_stats, 99/64/35/0.27 for
   ALL…UL>100B) instead of PK/(100·300 s) = 120/77/43/0.33. The replay traces last about 364 s, not 300 s, so the old
   `r_v` was 21 % too high and `c_v` 17 % too low; the product `r_v·c_v` (what the model uses) is unchanged
   (Pi 20.6 mW/stream, Jetson 9.2 mW/stream for ALL).
3. **Non-significant live steps set to zero.** `ΔP_m` of the light candidates (Pi RF-ea 1.9 mW, Jetson FFNN-ea
   1.3 mW, 95 % CI includes 0) is 0 in the model; heavy LSTM-pf: Pi 295 mW, Jetson 25 mW (both significant).
4. **Structural scenarios** in `energy_model.inference_power`: A (step paid once, default), B (step recurs with every
   inference call, capped at the W1 active power, never below A), C (no step beyond one stream).
5. **Validation at intermediate loads** (`energy_model.validation`), **CPU-time budget** (`cpu_budget`, `capacity`),
   **deferred processing** recomputed from the processing logs (`logdata.processing_energy`).
6. Fig. 6 is now drawn by `uncertainty.py` (median + 90 % band, scenario A, c_x from logs; dotted line = heavy, c_x = 0).
   `energy_model.py` no longer draws a figure.

## 2. Key numbers, old vs new

| Quantity | Old (paper) | New |
|---|---|---|
| P0 Pi / Jetson | 2.546 / 4.380 W | 2.536 / 4.351 W (power units) |
| p_rx | 2.94 / 1.43 mW | 2.89 / 1.38 mW |
| c_v | 173 / 78 µJ (per PK packet) | 209 / 93 µJ (per replayed packet; r_v·c_v unchanged) |
| c_v^off deferred | 1049 / 411 µJ | 1274 / 499 µJ (same basis as new c_v; ratio 6.1× / 5.4× unchanged) |
| r_v (ALL…UL>100B) | 120 / 77 / 43 / 0.33 | 99 / 64 / 35 / 0.27 packets/s |
| c_x ALL, Pi | 0 | 23.6 / 31.4 / 37.6 mJ (1/10/60 s) per stream-second |
| c_x ALL, Jetson | 0 | 11.7 / 15.2 / 17.9 mJ |
| c_x UL, Pi / Jetson | 0 | 11.4–17.3 (10 s: 15.4) / 6.0–8.6 (10 s: 7.7) mJ |
| c_x ALL>100B Pi / Jetson | 0 | 13.4–21.2 / 7.3–10.7 mJ |
| c_x UL>100B Pi / Jetson | 0 | 2.3–9.4 / 1.4–5.1 mJ |
| Crossover ALL light / heavy / batch | 146 / 72 / 88 | **63 / 41 / 46** (c_x = 0: 143 / 70 / 87) |
| Crossover UL light / heavy / batch | – / 107 / – | **141 / 70 / 85** (c_x = 0: 345 / 106 / 147) |
| e(1) Pi / Jetson [J per stream-hour], light | 9259 / 15811 | 9327 / 15756 |
| e(1) heavy | 10378 / 15925 | 10452 / 15875 |
| e(20) light / heavy / batch | – | 655/877, 772/912, 728/883 |
| e(60) light | 238 / 303 | **350 / 355** (c_x = 0: 237 / 300) |
| e(60) heavy | 320 / 335 | **432 / 387** (c_x = 0: 319 / 333) |
| e(60) heavy batch 32 | 276 / 306 | **388 / 358** (c_x = 0: 275 / 303) |
| S* Pi light / heavy / batch | 107 / 68 / 97 | **46 / 39 / 47** |
| S* Jetson light / heavy / batch | 393 / 225 / 382 | **167 / 126 / 165** |
| Baseline share at S = 1 | ≥ 88 % Pi, 99 % Jetson | ≥ 87.3 % Pi, ≥ 98.7 % Jetson |
| Inference share at 60 ALL streams, heavy | 26 % Pi, 10 % Jetson | 19.1 % Pi, 8.7 % Jetson |
| Feature share at 60 ALL streams, heavy | – | 26.1 % (19.6–31.3 %) Pi, 14.1 % (10.9–16.6 %) Jetson |
| Receive + dissection share at 60, heavy | – | 19.6 % Pi, 9.8 % Jetson |
| Per stream-hour, ALL, Pi [J] | receive 11, dissection 75 | receive 10, dissection 74, features 85–135 (113), heavy inference 65, batched 21 |
| Per stream-hour, ALL, Jetson [J] | 5, 34 | 5, 33, features 42–64 (55), heavy inference 32, batched 3 |
| Batching saving per stream, Pi, 60 ALL | 14 % | **10.2 %** (c_x = 0: 13.9 %) |
| Single-stream check (W3 with dissection) | within 4.3 % Pi, 0.3 % Jetson | Pi −4.9 … −0.6 % (incl. full packets), Jetson within 0.2 % |
| P0 vs idle | within 0.11 W | within 0.08 W (idle 2.560 / 4.273 W) |

(Old crossovers were "first S with Jetson cheaper"; light ALL was 146 in the script.)

## 3. N1.2: c_x at one stream

W2 reference (features, no inference, mean over candidates) minus W3 single stream with dissection, J per 300 s,
95 % CI (Welch), vs. expected `c_x·300 s` (1–60 s windows):

| Device | View | Difference | Expected |
|---|---|---|---|
| Pi | ALL | +3.2 [−4.9, +11.3] | 7.1–11.3 |
| Pi | ALL>100B | −5.3 [−9.5, −1.0] | 4.0–6.3 |
| Pi | UL | −9.4 [−15.3, −3.5] | 3.4–5.2 |
| Pi | UL>100B | −17.5 [−20.0, −15.0] | 0.7–2.8 |
| Jetson | ALL … UL>100B | −26 to −35 (CIs exclude 0) | 0.4–5.4 |

Verdict: **consistent for Pi ALL, not resolvable otherwise** — the W2 references (Oct 2026) and W3 single-stream runs
(May/June 2026) differ by campaign offsets of up to 35 J that exceed the expected effect.
Supporting live evidence: the process CPU of the W2 reference pipeline is, per stream-second (median over candidates),
Pi 30 / 23 / 16 / 2 ms and Jetson 18 / 12 / 8 / 1 ms for ALL / ALL>100B / UL / UL>100B, matching the offline CPU time
(Pi ALL 20–31 ms, UL 10–15 ms; Jetson ALL 12–19 ms, UL 6–9 ms). The separate live feature-extraction campaign (Jul 2026,
one window per run) shows Pi ALL 875 / 894 / 902 J per 300 s for 1 / 10 / 60 s windows and process CPU 43–57 ms/s.

## 4. N1.3: CPU-time budget and capacity

Core time per stream-second (dissection: slope of busy cores over S at loss-free loads in the W3 CPU logs; features:
offline runs at one core; inference: W1 wall time per prediction × busy cores during W1):

| | Pi | Jetson |
|---|---|---|
| Dissection ALL / UL / full packets | 17.3 / 6.2 / 21.5 ms (one core: 58 ALL, 160 UL streams; loss onset 50–60 / 170) | 8.3 / 3.6 / 9.7 ms (one core: 120 ALL, 281 UL) |
| Features ALL / UL | 19.8–30.8 / 9.6–14.5 ms | 12.3–18.5 / 6.1–9.2 ms |
| Inference light | 0.08 ms (1 core) | 0.41 ms (GPU, 1.1 cores busy) |
| Inference heavy | 9.0 ms wall on 4 cores = 36 ms core time | 3.7 ms (GPU) |
| Inference heavy, batch 32 | 2.8 ms wall on 4 cores = 11.3 ms core | 0.16 ms (GPU) |

Capacity (streams), ranges over window 1–60 s and inference time between W1 wall and W1 core time:

| | Pi ALL | Pi UL | Jetson ALL | Jetson UL |
|---|---|---|---|---|
| (a) all stages in one thread | light 21–27, heavy 12–22, batch 17–25 | 48–63, 18–40, 31–54 | 37–48, 33–41, 37–48 | 76–99, 61–75, 77–101 |
| (b) one core per stage | 33–50, 28–50, 33–50 | 69–104, 28–104, 69–104 | 54–82 (all classes) | 108–163 |
| (c) ideal use of all cores (dissection single-threaded) | 58, 48–55, 58 | 160, 71–77, 125–148 | 120 | 281 |

Bottom line: **usable Pi capacity with the complete pipeline is about 12–27 ALL streams single-threaded and 28–50 with
one core per stage, not 60**; Jetson 33–48 / 54–82, dissection-bound at 120.

## 5. N2: residuals at intermediate loads

State from captured (Jetson Nov campaign: processed) packets / sent relative to the loss-free ratio of the same view
(100 streams without dissection; UL>100B: 100 streams with dissection): loss-free ≥ 0.98, onset 0.95–0.98,
saturated < 0.95. Error = (model − measured) / measured, power. Calibration points (Pi UL, UL>100B at 100; Jetson all
views at 100) omitted.

| Device | View | S | runs | campaign | captured (rel.) | state | measured [W] | model [W] | error |
|---|---|---:|---:|---|---:|---|---:|---:|---:|
| pi | ALL | 1 | 30 | 2026-05 | 1.000 | loss-free | 2.673 | 2.559 | -4.2 % |
| pi | ALL | 20 | 1 | 2026-10 | 1.001 | loss-free | 3.218 | 3.007 | -6.6 % |
| pi | ALL | 30 | 1 | 2026-10 | 0.997 | loss-free | 3.357 | 3.242 | -3.4 % |
| pi | ALL | 40 | 1 | 2026-10 | 0.995 | loss-free | 3.557 | 3.477 | -2.2 % |
| pi | ALL | 50 | 30 | 2026-10 | 0.995 | loss-free | 3.718 | 3.713 | -0.1 % |
| pi | ALL | 60 | 29 | 2026-10 | 0.948 | saturated | 3.829 | 3.948 | +3.1 % |
| pi | ALL | 70 | 29 | 2026-10 | 0.811 | saturated | 3.864 | 4.183 | +8.3 % |
| pi | ALL | 80 | 27 | 2026-10 | 0.705 | saturated | 3.871 | 4.419 | +14.2 % |
| pi | ALL | 90 | 28 | 2026-10 | 0.623 | saturated | 3.876 | 4.654 | +20.1 % |
| pi | ALL | 100 | 30 | 2026-06 | 0.556 | saturated | 4.000 | 4.890 | +22.2 % |
| pi | ALL>100B | 1 | 27 | 2026-05 | 1.000 | loss-free | 2.637 | 2.552 | -3.2 % |
| pi | ALL>100B | 50 | 30 | 2026-10 | 0.994 | loss-free | 3.454 | 3.345 | -3.2 % |
| pi | ALL>100B | 60 | 30 | 2026-10 | 0.995 | loss-free | 3.565 | 3.507 | -1.6 % |
| pi | ALL>100B | 70 | 30 | 2026-10 | 0.989 | loss-free | 3.660 | 3.669 | +0.2 % |
| pi | ALL>100B | 80 | 30 | 2026-10 | 0.897 | saturated | 3.709 | 3.831 | +3.3 % |
| pi | ALL>100B | 90 | 30 | 2026-10 | 0.790 | saturated | 3.733 | 3.993 | +6.9 % |
| pi | ALL>100B | 100 | 30 | 2026-06 | 0.691 | saturated | 3.974 | 4.154 | +4.5 % |
| pi | UL | 1 | 30 | 2026-05 | 1.000 | loss-free | 2.610 | 2.546 | -2.4 % |
| pi | UL | 110 | 12 | 2026-10 | 0.989 | loss-free | 3.480 | 3.658 | +5.1 % |
| pi | UL | 120 | 15 | 2026-10 | 0.989 | loss-free | 3.527 | 3.760 | +6.6 % |
| pi | UL | 130 | 8 | 2026-10 | 0.992 | loss-free | 3.580 | 3.862 | +7.9 % |
| pi | UL | 140 | 14 | 2026-10 | 0.991 | loss-free | 3.661 | 3.964 | +8.3 % |
| pi | UL | 150 | 30 | 2026-10 | 0.991 | loss-free | 3.699 | 4.066 | +9.9 % |
| pi | UL | 160 | 30 | 2026-11 | 0.990 | loss-free | 3.742 | 4.168 | +11.4 % |
| pi | UL | 170 | 30 | 2026-11 | 0.986 | loss-free | 3.790 | 4.270 | +12.7 % |
| pi | UL | 180 | 29 | 2026-11 | 0.929 | saturated | 3.781 | 4.372 | +15.7 % |
| pi | UL | 190 | 26 | 2026-11 | 0.886 | saturated | 3.813 | 4.474 | +17.3 % |
| pi | UL | 200 | 29 | 2026-11 | 0.834 | saturated | 3.805 | 4.576 | +20.3 % |
| pi | UL>100B | 1 | 26 | 2026-05 | 1.000 | loss-free | 2.553 | 2.539 | -0.6 % |
| pi | FULL | 1 | 30 | 2026-05 | 1.000 | loss-free | 2.692 | 2.559 | -4.9 % |
| pi | FULL | 10 | 30 | 2026-11 | 0.998 | loss-free | 3.055 | 2.771 | -9.3 % |
| pi | FULL | 20 | 30 | 2026-11 | 0.997 | loss-free | 3.245 | 3.007 | -7.4 % |
| pi | FULL | 30 | 30 | 2026-11 | 0.995 | loss-free | 3.422 | 3.242 | -5.3 % |
| pi | FULL | 40 | 30 | 2026-11 | 0.994 | loss-free | 3.623 | 3.477 | -4.0 % |
| pi | FULL | 50 | 29 | 2026-10 | 0.944 | saturated | 3.767 | 3.713 | -1.4 % |
| pi | FULL | 60–100 | 30 | 2026-10/06 | 0.43–0.77 | saturated | 3.78–4.00 | 3.95–4.89 | +4.5 … +22.4 % |
| jn | ALL | 1 | 30 | 2026-06 | 1.000 | loss-free | 4.359 | 4.361 | +0.0 % |
| jn | ALL | 110 | 29 | 2026-11 | 0.995 | loss-free | 5.305 | 5.510 | +3.9 % |
| jn | ALL | 120 | 30 | 2026-11 | 0.977 | onset | 5.366 | 5.615 | +4.6 % |
| jn | ALL | 130–150 | 28–30 | 2026-11 | 0.85–0.95 | saturated | 5.40–5.44 | 5.72–5.93 | +5.9 … +9.1 % |
| jn | ALL>100B | 1 | 30 | 2026-06 | 1.000 | loss-free | 4.368 | 4.358 | -0.2 % |
| jn | ALL>100B | 110 | 10 | 2026-11 | 1.007 | loss-free | 5.133 | 5.151 | +0.3 % |
| jn | ALL>100B | 120 | 10 | 2026-11 | 1.006 | loss-free | 5.189 | 5.224 | +0.7 % |
| jn | ALL>100B | 130 | 28 | 2026-11 | 1.007 | loss-free | 5.242 | 5.296 | +1.0 % |
| jn | ALL>100B | 140 | 30 | 2026-11 | 1.005 | loss-free | 5.285 | 5.369 | +1.6 % |
| jn | ALL>100B | 150 | 29 | 2026-11 | 1.002 | loss-free | 5.361 | 5.442 | +1.5 % |
| jn | ALL>100B | 160 | 30 | 2026-11 | 0.993 | loss-free | 5.344 | 5.515 | +3.2 % |
| jn | ALL>100B | 170 | 30 | 2026-11 | 0.977 | onset | 5.394 | 5.587 | +3.6 % |
| jn | ALL>100B | 180 | 10 | 2026-11 | 0.963 | onset | 5.318 | 5.660 | +6.4 % |
| jn | UL | 1 | 30 | 2026-06 | 1.000 | loss-free | 4.361 | 4.355 | -0.1 % |
| jn | UL | 150 | 1 | 2026-10 | 0.987 | loss-free | 4.873 | 5.044 | +3.5 % |
| jn | UL | 200 | 1 | 2026-10 | 0.990 | loss-free | 5.056 | 5.275 | +4.3 % |
| jn | UL | 240–300 | 6–9 | 2026-10 | 0.26–0.90 | saturated | 4.79–4.91 | 5.46–5.74 | +12.7 … +17.6 % |
| jn | UL | 350 | 1 | 2026-10 | 0.979 | onset | 5.375 | 5.968 | +11.0 % |
| jn | UL | 400 | 1 | 2026-10 | 0.985 | loss-free | 5.463 | 6.199 | +13.5 % |
| jn | UL>100B | 1 | 29 | 2026-06 | 1.000 | loss-free | 4.362 | 4.352 | -0.2 % |
| jn | UL>100B | 200–400 | 1–3 | 2026-10 | 0.95–1.03 (few packets, noisy) | mixed | 4.48–4.56 | 4.63–4.91 | +3.4 … +9.2 % |
| jn | FULL | 1 | 30 | 2026-06 | 1.000 | loss-free | 4.370 | 4.361 | -0.2 % |
| jn | FULL | 80 | 29 | 2026-11 | 0.994 | loss-free | 5.167 | 5.194 | +0.5 % |
| jn | FULL | 90 | 30 | 2026-11 | 0.984 | loss-free | 5.242 | 5.299 | +1.1 % |
| jn | FULL | 100–120 | 26–30 | 2026-11 | 0.91–0.96 | onset/saturated | 5.29–5.37 | 5.40–5.62 | +2.2 … +4.5 % |

(FULL = complete capture, modeled with the ALL packet rate and the same c_v. Full table: `energy_model.py` output.)

**Honest error range (replaces "within 4.3 %"):**
- Single stream: Pi −4.9 … −0.6 %, Jetson within 0.2 %.
- Loss-free intermediate loads, repeated runs: **Pi −9.3 … +12.7 %** (low loads too low, UL near capacity too high;
  median |error| of all Pi loss-free points 5.9 %), **Jetson +0.3 … +3.9 %**.
- Jetson single runs (Oct campaign, lower fan level): +3.4 … +13.5 % at 150–400 UL/UL>100B streams.
- Saturated loads: model too high by up to +22 % (expected: less work done).
- Campaign effects: Pi intermediate loads logged Oct/Nov, calibration May/June; 110 UL streams (Oct) drew 3.480 W,
  less than 100 UL streams (May) at 3.571 W. Marginal power per stream within the later campaign: UL 110–170 5.3 mW vs
  model 10.2 mW; full packets 10–40 18.8 mW vs 23.5 mW. Jetson Oct campaign (fan 22 % vs 25 %): capture without
  dissection 4.239 vs 4.339 W (1 stream) and 4.358 vs 4.480 W (100 streams).

**N2.4 decision: keep linear.** A concave capture term `100 (p_rx + r_v c_v)(S/100)^b`, b = 0.756, lowers the rms
error of the Pi loss-free points from 6.9 to 4.3 pp and the range to −3.5 … +7.1 %, i.e., it removes about 40 % of the
rms, not most, and it cannot separate load dependence from campaign offsets. Sensitivity: point crossover heavy ALL
41 → 35, light ALL 63 → 56. The residual is carried into N3 as factor k on the per-stream capture term, drawn from the
loss-free repeated-run points with S ≥ 20: Pi k ∈ {0.72 … 1.51} (14 values), Jetson k ∈ {0.82 … 0.98} (9 values).

## 6. N3: uncertainty

Parameter spread (median [5 %, 95 %]): Pi P0 2.536 [2.533, 2.538] W, p_rx 2.90 [2.86, 2.93] mW, c_v 209 [202, 216] µJ,
c_x ALL 30.6 [24.2, 36.9] mJ, ΔP heavy 295 [255, 333] mW, c_m heavy 18.0 [17.9, 18.1] mJ, batch 5.70 mJ;
Jetson P0 4.351 [4.347, 4.354] W, c_v 92.5 [90.0, 95.2] µJ, c_x ALL 14.8 [12.0, 17.7] mJ, ΔP heavy 24.5 [9.9, 39.2] mW,
c_m heavy 8.91 mJ, batch 0.766 mJ; r_v ALL 98.95 [97.4, 100.5].

### Crossover (Jetson cheaper per stream from S), median [5 %, 95 %]

| View | Class | A / logs | B / logs | C / logs | A / c_x=0 | B / c_x=0 | C / c_x=0 |
|---|---|---|---|---|---|---|---|
| ALL | light | 65 [45, 101] | 65 [45, 101] | 65 [45, 101] | 162 [73, 266] | 162 [73, 266] | 162 [73, 266] |
| ALL | heavy | 42 [31, 57] | 6 [6, 8] | 49 [37, 67] | 75 [46, 96] | 7 [6, 154] | 88 [53, 112] |
| ALL | heavy, batch 32 | 47 [34, 67] | 39 [33, 56] | 55 [40, 79] | 94 [52, 129] | 65 [43, 86] | 110 [61, 151] |
| UL | light | 150 [104, 233] | same | same | 394 [171, 662] | same | same |
| UL | heavy | 72 [58, 90] | 7 [6, 146] | 85 [68, 106] | 110 [78, 128] | 152 [6, 187] | 130 [91, 150] |
| UL | heavy, batch 32 | 90 [68, 119] | 65 [56, 79] | 105 [80, 140] | 156 [98, 194] | 97 [65, 102] | 185 [115, 228] |

(Light: no live step, so A = B = C. Scenario B with the heavy model: the 0.3 W Pi step recurs per call and the Pi
reaches its W1 power from about 7 streams; under B the Jetson also pays its step per call, so the B ranges depend
strongly on the drawn Jetson step (10–39 mW) and are bimodal without c_x.)

### P(Pi cheaper per stream), ALL, at S = 1 / 20 / 60

| Class | A / logs | B / logs | C / logs | A / 0 | B / 0 | C / 0 |
|---|---|---|---|---|---|---|
| light | 1.00 / 1.00 / 0.63 | 1.00 / 1.00 / 0.63 | 1.00 / 1.00 / 0.63 | 1.00 / 1.00 / 1.00 | same | same |
| heavy | 1.00 / 1.00 / 0.02 | 1.00 / 0.34 / 0.54 | 1.00 / 1.00 / 0.14 | 1.00 / 1.00 / 0.75 | 1.00 / 0.86 / 0.95 | 1.00 / 1.00 / 0.93 |
| heavy, batch 32 | 1.00 / 1.00 / 0.12 | 1.00 / 1.00 / 0.02 | 1.00 / 1.00 / 0.33 | 1.00 / 1.00 / 0.92 | 1.00 / 1.00 / 0.73 | 1.00 / 1.00 / 0.95 |

(1.00 = at least 99.5 % of draws.) e(S), scenario A, c_x logs: S = 60 Pi 343 [308, 387] / Jetson 351 [340, 362] (light),
425 [390, 470] / 383 [372, 394] (heavy), 380 [346, 425] / 353 [343, 365] (batch); S = 20: 647/873, 764/908, 720/878.

### One-at-a-time ranking (crossover ALL, scenario A, ±1 SD, base 41 heavy / 64 light)

1. Pi residual factor k: heavy 48 / 36, light 79 / 54.
2. Pi c_x window: heavy 46 / 37, light 75 / 56.
3. Jetson c_x window: heavy 39 / 43, light 60 / 68.
4. All bootstrapped parameters (c_v, p_rx, P0, ΔP_m, c_m, c_x runs, r_v): ≤ 1 stream.
Sentence for the paper: "The crossover is dominated by the transfer of stage measurements to concurrent load (residual
factor, feature window), not by measurement noise."

### Robustness labels (robust = holds in ≥ 95 % of draws in every structural scenario: ΔP A/B/C × c_x logs/0)

| Claim | Label | Worst-case share |
|---|---|---|
| G2 (old): Pi cheaper per stream up to ~70 ALL streams with the heaviest model | **fragile** | 0.00 (A/logs); best 0.76 |
| G2 (old): … up to 107 UL streams with the heaviest model | **fragile** | 0.00; best 0.84 |
| G2 (old): Pi cheaper at every load it can capture with light models (60 ALL) | **fragile** | 0.63 |
| … light, up to 170 UL | **fragile** | 0.31 |
| Pi cheaper per stream up to 44 ALL / 103 UL streams, light | **robust** | 0.955 / 0.951 |
| Pi cheaper up to 30 ALL / 57 UL streams, heavy, scenarios A and C (c_x logs or 0) | robust (A, C); not under B | 0.963 / 0.952 |
| Pi cheaper up to 5 streams, heavy, all scenarios incl. B | robust | 0.983 |
| Pi cheaper up to 32 ALL / 55 UL streams, heavy batch 32, all scenarios | **robust** | 0.970 / 0.953 |
| Pi cheaper at 20 ALL streams, all classes | robust in A and C; fragile overall (heavy B: 0.34) | 0.344 |
| G5 live (old): batching saves ~14 % per stream on the Pi at 60 streams (10–18 %) | **fragile** | 0.45 (B/c_x=0 gives 18.2 %; A/logs 10.4 %) |
| G5 live: batching saves ≥ 9 % per stream on the Pi at 60 ALL streams | **robust** | 0.991 |
| G5 live: batching saves ≥ 5 % | robust | 1.00 |

G5 saving at 60 ALL streams, median [5, 95]: A/logs 10.4 [9.4, 11.4] %, B/logs 13.8 [12.3, 15.4], C/logs 10.9 [9.8, 11.9],
A/0 14.2 [12.3, 15.0], B/0 18.2 [15.9, 20.0], C/0 15.2 [12.9, 15.9]. At 30 streams (within Pi capacity): A/logs
7.4 [6.9, 7.9] %, C/logs 7.9 %, A/0 9.2 %, B 25–30 %.

## 7. Claims outside Section VI that must change (for the main session)

**Abstract** — replace the sentence "An additive energy model … predicts measurements not used for calibration within
4.3 % and shows that the CPU-only device remains cheaper per stream up to its capture capacity." with, e.g.:
> "An additive energy model, calibrated per pipeline stage, reproduces capture energy at loss-free loads not used for
> calibration within −9 to +13 % on the Raspberry Pi and 4 % on the Jetson. It shows that feature extraction, not
> inference, dominates the per-stream cost, and that the CPU-only device is cheaper per stream for small monitors
> (robustly up to about 30 streams with the heaviest model and 44 with light ones), while the GPU device pays off
> through capacity and, for heavy models, possibly also through energy beyond 30–60 streams."
Keep "inference adds at most 14 % … for a single live stream" (measured, unchanged).

**Introduction, contribution 2** — "The model predicts measurements excluded from calibration with an error below
4.3 %" → "The model separates baseline power from the per-stream costs of packet reception, dissection, feature
extraction, and prediction, reproduces measurements at loads excluded from calibration within −9 to +13 % (Raspberry Pi)
and +4 % (Jetson), and quantifies the uncertainty of its predictions with a bootstrap and scenario analysis." Also add
"feature extraction" to the list of per-stream costs.

**RQ3 answer** (one explicit paragraph, e.g., end of Sec. VI or in Sec. VII): *Measured:* single-stream energy of all
stages; capture with dissection at 1–200 (Pi) / 1–400 (Jetson) streams; CPU time per stage. *Predicted:* energy per
stream-hour with concurrent inference and feature extraction. *Uncertainty:* crossover heavy ALL 42 [31, 57] streams
(scenario A), 49 [37, 67] (C), 6–8 (B); light 65 [45, 101]; model error −9 … +13 % (Pi). *Capacity:* the Pi processes
28–50 ALL streams with one core per stage (12–27 single-threaded), the Jetson 54–82 (33–48).

**G2** ("Choose the device by the number of streams") — new wording:
> "Per stream, the CPU-only device is the cheaper monitor for small deployments: up to about 44 (ALL) or 103 (UL)
> streams with light models and about 30 (ALL) or 57 (UL) with the heaviest model, in at least 95 % of the model's
> uncertainty draws (Fig. 6). Beyond that, the two devices are equal within the model's uncertainty; once feature
> extraction is counted, the GPU device's lower per-stream cost catches up with its higher baseline at a median of 65
> (light) and 42 (heavy) ALL streams. The CPU-only device also runs out of CPU time first: with one core per pipeline
> stage it processes 28–50 ALL streams, the GPU device 54–82. Choose the GPU device when one monitor must serve more
> than about 30 ALL (60 UL) streams with a heavy model, or more than the CPU-only device can process."
Remove "up to about 70 (ALL) or 107 (UL) streams even with the heaviest one" and "beyond about 60 ALL or 170 UL streams".

**G4** ("For a single live stream, choose by score") — single-stream statements stay (measured). Change the last
sentence "Energy-aware configurations pay off … and with many streams on one monitor" to note that with many streams the
feature preparation costs more than the heaviest model (Pi: 85–135 vs 65 J per stream-hour; Jetson 42–64 vs 32), so
energy-aware *feature sets* (fewer/shorter windows, smaller views) matter at least as much as energy-aware models.
The text in G4 "a step of about 0.3 W … regardless of compute" is unchanged.

**G5** ("Batch predictions") — offline part unchanged. Live part: replace "which the energy model translates into 14 %
less energy per stream on the Raspberry Pi at 60 streams" with "which the energy model translates into 9–18 % less
energy per stream on the Raspberry Pi at 60 streams (at least 9 % in all scenarios; about 7–9 % at 30 streams unless the
live step recurs with every call, where batching saves 25–30 %); batching also cuts the CPU time of inference by a factor
of 3 on the Raspberry Pi, which raises its capacity." Mark as prediction.

**Limitations** — replace "Finally, the energy model is calibrated from stage measurements and single streams; its
predictions for concurrent streams are hypotheses that a full-pipeline load test must confirm." with:
> "Finally, the energy model is calibrated from stage measurements. Its error against loss-free loads not used for
> calibration is −9 to +13 % on the Raspberry Pi, partly due to offsets between measurement campaigns; the cost of
> feature extraction comes from offline single-window runs; the behavior of the live inference step with many streams
> is unknown (we bound it by scenarios); and the batched outputs of the Pi's stalling RF/LGBM are excluded. A pipeline
> that splits traffic per stream and runs all stages under concurrent load does not exist yet, so all predictions
> beyond one stream remain to be confirmed with such a pipeline."

**Conclusion** — replace "predicts that the CPU-only device stays the cheaper monitor per stream up to its capture
capacity, so that the GPU device pays off through capacity" with "predicts that the CPU-only device is the cheaper
monitor for small deployments, that feature extraction rather than inference dominates the per-stream cost, and that
the GPU device pays off through capacity and, for heavy models, possibly also through energy beyond about 30–60 streams;
these predictions carry an uncertainty that we quantify, and validating them with a complete multi-stream pipeline is
the next step." Keep "test whether batching across streams recovers the offline savings".

**Sec. V-E (results.tex, "Traffic Views Under Load")** — the sentence "How inference for many concurrent streams
changes these limits remains to be measured" can point to the CPU-time budget: "Sec. VI estimates from the CPU time
of each stage that the complete pipeline limits the Raspberry Pi to 28–50 ALL streams."

**Table II (measurement.tex)** — caption says the retained packet counts are for the "five-minute, 100-stream replay
trace"; the pcap_stats traces last about 364 s (3.6 M packets at 9,893 packets/s for ALL), and the monitor received
about 3.1 M packets in five minutes. Suggest "in the 100-stream replay trace (about six minutes)". The loss onsets in
Table II are consistent with the logs (ALL 60, UL 170 on the Pi; ALL 120 on the Jetson); with the 98 % rule above, the
Pi's ALL>100B loss starts at 80 (70 is at 98.9 %), the Pi's full-packet capture at 50 (40 is loss-free at 99.4 %), and
the Jetson's ALL>100B at 170 (160 is loss-free) — check against how Table II defined onset.

**Discussion intro** "G2 and the live part of G5 extend them with the model … and are therefore predictions" — keep,
add "with the ranges of Sec. VI-D".

## 8. Open points / not resolved

- c_x is from offline single-window runs (pandas loop); the W2 pipeline computes several windows per update with numpy.
  Their CPU time per stream-second agrees (Sec. 3), but the energy per update under concurrent load is not measured.
- ΔP_m with more than one stream is unknown; scenario B is a pessimistic bound that changes the heavy-model conclusion
  completely (crossover 6–8 streams). Only a multi-stream measurement can decide between A, B and C.
- The campaign offsets (Pi Oct/Nov vs May/June; Jetson fan level) cannot be separated from a load-dependent c_v with
  existing data.
- Table IX candidate ranges recomputed on 13 Nov (consistency pass, see `../CONSISTENCY.md`) with the current P0: only the lower end of c_m (1 per call) on the Pi changed, 0.07 → 0.08 mJ; all other ranges unchanged.
- Inference CPU time for one thread on the Pi is unknown (W1 ran ONNX Runtime with 4 threads); capacities use the
  range W1 wall time … W1 core time.

## Thesis author's answers (16 Nov 2026)

- Settings: Jetson 25 W mode, `jetson_clocks` not set (schedutil, dynamic clocks); Pi `ondemand` governor. Now in Sec. III.
- Live: Pi LGBM(e) UL buffer health and Jetson RF(p) UL stalling were selected by mistake and are excluded
  (`figures_src/plot_figures.py: EXCLUDED_LIVE`, `shortlist_check.py`). New live numbers: Pi 23 candidates, 21 with
  predictions, 10 heavy (all pf) add 63–99 J (8.4–12.8 %), step 0.21–0.33 W, W1 c_m 0.8–42 mJ (factor 50), at most
  10.6 J from predictions; 11 light (all ea) within 6.4 J (0.9 %). Jetson 24 candidates, 22 with predictions, max 14.5 J
  (1.1 %). Table IX: only the heavy Pi step changed (0.35 → 0.33 W). Energy-model inputs (LSTM-pf, RF-ea, FFNN-ea) unchanged.
- Validation: Jetson loads above 200 streams dropped (`energy_model.validation`). Jetson UL/UL>100B single runs at
  150–200: +3.4 to +4.3 %; other Jetson views at onset/saturated (100–180): +2.2 to +9.1 %. Residual factors k unchanged.
- Validation-based shortlist check unchanged: 34 of 47 validation candidates measured.
