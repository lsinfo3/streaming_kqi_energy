"""Additive energy model of the monitoring pipeline, calibrated from the per-run thesis logs.

    P(S)  = P0 + P_inf(S) + S * [ k * (p_rx + r_v * c_v) + f * c_x ]          (power of one monitor)
    e(S)  = 3600 s * P(S) / S                                                (energy per stream-hour)
    P_inf = dP_m + S * f * c_m   (scenario A, default; B and C see inference_power)

Calibration (all values: means of the logged runs after removing outliers outside 1.5 IQR, in power units E/T):
  P0, p_rx  capture without dissection (W3), 1 and 100 streams, mean over the four views
  r_v       packets per second and stream of the 100-stream replay traces (30 seeds, general_data/pcap_stats)
  c_v       slope of capture power with dissection at 100 streams over 100 * r_v (Jetson: four views;
            Pi: UL and UL>100B only, because it loses packets for ALL and ALL>100B at 100 streams)
  c_x       offline feature extraction (100 single-stream files back to back on one core):
            (E - P0 * T) / stream-seconds, for 1 s, 10 s and 60 s windows (range); stream-seconds = feature
            updates with 60 s windows (a 60 s window is almost never empty, so updates = clock ticks)
  dP_m      W2 power increment minus actual W2 prediction rate times unbatched W1 c_m;
            retain a nonnegative residual only when its bootstrap 95% interval excludes zero
  c_m       W1 energy minus P0 * W1 duration, / 100,000 predictions (1 per call or 32 per call)
  k         1 in the point model; the intermediate-load residuals (validation()) feed uncertainty.py

Not used in calibration and used for validation: capture with dissection at all other logged loads.
The figure (Fig. 6, with uncertainty bands) is drawn by uncertainty.py.
"""
import math
import numpy as np
import pandas as pd
from logdata import (capture_power_runs, capture_runs, capture_loads, idle_power, w1_runs, live_power_runs,
                     live_runs, fx_offline_runs, fx_live_runs, pcap_rates, pcap_packets, NCORES, iqr_mask,
                     execution, LIVEVIEW, processing_energy, capture_file)

T5 = 300.0                                         # five-minute run, for J-per-run printouts
F = 1.0                                            # predictions and feature updates per stream-second
VIEWS = ["ALL", "ALL>100B", "UL", "UL>100B"]
DEVS = ("pi", "jn")
WINDOWS = (1, 10, 60)
CLASSES = ("light", "heavy", "heavy, batch 32")
CAND = {"pi": {"light": ("buffer_health", "all", "rf", "ea"), "heavy": ("buffer_health", "all", "lstm", "pf")},
        "jn": {"light": ("buffer_health", "all", "ffnn", "ea"), "heavy": ("buffer_health", "all", "lstm", "pf")}}
CV_FIT = {"pi": [2, 3], "jn": [0, 1, 2, 3]}         # views used for the c_v fit
STEP_SIGNIFICANT = {}                             # (dev, class) -> corrected residual bootstrap 95% CI is positive
S_CAPTURE = {("pi", "ALL"): 60, ("pi", "ALL>100B"): 70, ("pi", "UL"): 170,
             ("jn", "ALL"): 120, ("jn", "ALL>100B"): 150}   # loss onset with dissection (Table II)


# ======================================================================= inputs
def load_inputs():
    """Primitive run-level measurements, grouped by independent measurement series.
    Leaves are 1-D arrays (one value per run) or tuples of equally long arrays (paired per run)."""
    inp = {"pcap": {v: pcap_rates()[v].to_numpy(float) for v in VIEWS}}
    for d in DEVS:
        g = inp[d] = {}
        g["nd1"] = {v: capture_power_runs(d, v, 1, False) for v in VIEWS}
        g["nd100"] = {v: capture_power_runs(d, v, 100, False) for v in VIEWS}
        g["d100"] = {v: capture_power_runs(d, v, 100, True) for v in VIEWS}
        g["w1"] = {"light": w1_runs(d, *CAND[d]["light"])[:2], "heavy": w1_runs(d, *CAND[d]["heavy"])[:2],
                   "heavy, batch 32": w1_runs(d, *CAND[d]["heavy"], batch=True)[:2]}
        g["w2"] = {}
        for c in ("light", "heavy"):
            x = live_runs(d, *CAND[d][c], "model_execution")
            keep = iqr_mask(x.tf_energy)
            # Power and actual prediction rate belong to the same run.
            g["w2"][c] = {
                "me": ((x.tf_energy / x.tf_duration).to_numpy()[keep],
                       (x.predictions / x.tf_duration).to_numpy()[keep]),
                "ref": live_power_runs(d, *CAND[d][c], "feature_extraction")}
        g["fx"] = {}
        for v in VIEWS:
            nss = fx_offline_runs(d, v, 60).total_output_elements.mean()
            for w in WINDOWS:
                x = fx_offline_runs(d, v, w)
                g["fx"][(v, w)] = (x.tf_energy.to_numpy(float) / nss, x.tf_duration.to_numpy(float) / nss)
    # Resolve offsets on the corrected residual, including W1/baseline uncertainty.
    draws = means(inp, 10000, np.random.default_rng(2026))
    for d in DEVS:
        for c in ("light", "heavy"):
            lo, hi = np.percentile(live_residual(draws[d], c), [2.5, 97.5])
            STEP_SIGNIFICANT[(d, c)] = lo > 0
    return inp


def _leaf_mean(leaf, N, rng):
    if isinstance(leaf, tuple):                                          # paired per run: shared indices
        n = len(leaf[0])
        if N is None: return tuple(a.mean() for a in leaf)
        idx = rng.integers(0, n, (N, n)); return tuple(a[idx].mean(1) for a in leaf)
    if N is None: return leaf.mean()
    return leaf[rng.integers(0, len(leaf), (N, len(leaf)))].mean(1)


def means(tree, N=None, rng=None):
    """Point means (N=None) or N bootstrap means of every leaf; same nesting as the input tree."""
    if isinstance(tree, dict): return {k: means(v, N, rng) for k, v in tree.items()}
    return _leaf_mean(tree, N, rng)


# ======================================================================= calibration
def live_residual(g, cls):
    """Residual W2 increment after accounting for predictions actually produced."""
    nd1 = sum(g["nd1"].values()) / 4
    nd100 = sum(g["nd100"].values()) / 4
    P0 = nd1 - (nd100 - nd1) / 99
    E, T = g["w1"][cls]
    cm = (E - P0 * T) / 1e5
    measured, rate = g["w2"][cls]["me"]
    return measured - g["w2"][cls]["ref"] - rate * cm


def calibrate(m):
    """Model parameters from (point or bootstrap) means; works elementwise on arrays of draws."""
    par = {"rv": [m["pcap"][v] for v in VIEWS]}
    rv = [m["pcap"][v] for v in VIEWS]
    for d in DEVS:
        g = m[d]
        nd1 = sum(g["nd1"][v] for v in VIEWS) / 4; nd100 = sum(g["nd100"][v] for v in VIEWS) / 4
        prx = (nd100 - nd1) / 99; P0 = nd1 - prx
        sel = CV_FIT[d]; x = [100 * rv[i] for i in sel]; y = [g["d100"][VIEWS[i]] for i in sel]
        xm = sum(x) / len(x); ym = sum(y) / len(y)
        cv = sum((xi - xm) * (yi - ym) for xi, yi in zip(x, y)) / sum((xi - xm) ** 2 for xi in x)
        d0 = ym - cv * xm
        inf = {}
        for c in CLASSES:
            E, Tw = g["w1"][c]; bat = c.endswith("32")
            src = "heavy" if bat else c
            dP = np.maximum(0, live_residual(g, src))
            if not STEP_SIGNIFICANT.get((d, src), True): dP = 0 * dP      # not measurable: no live step
            inf[c] = dict(dP=dP, cm=(E - P0 * Tw) / 1e5, PW1=E / Tw, tm=Tw / 1e5, batch=32 if bat else 1)
        cx = {v: [g["fx"][(v, w)][0] - P0 * g["fx"][(v, w)][1] for w in WINDOWS] for v in VIEWS}
        tx = {v: [g["fx"][(v, w)][1] for w in WINDOWS] for v in VIEWS}
        par[d] = dict(P0=P0, prx=prx, cv=cv, d0=d0, inf=inf, cx=cx, tx=tx)
    return par


# ======================================================================= model
def cx_value(pd_, view, cx):
    """c_x for scenario cx: 'logs' (10 s window), 'zero', a window length (1/10/60) or a number u in [0,1]
    that interpolates linearly between the 1 s and the 60 s window (used by the uncertainty analysis)."""
    c1, c10, c60 = pd_["cx"][view]
    if isinstance(cx, str):
        return 0.0 if cx == "zero" else c10
    if isinstance(cx, (int,)) and cx in WINDOWS:
        return pd_["cx"][view][WINDOWS.index(cx)]
    return c1 + np.asarray(cx) * (c60 - c1)


def inference_power(pd_, cls, S, dp="A"):
    """Power drawn for inference by S streams.
    A: the live step dP_m is paid once per device, plus c_m per prediction.
    B: the live step is paid per inference call (it grows with the inference duty cycle), but the
       inference power never exceeds the W1 active power above baseline (continuous inference); never below A.
    C: no live step beyond one stream (pure c_m)."""
    p = pd_["inf"][cls]; S = np.asarray(S, float)
    comp = S * F * p["cm"]
    A = p["dP"] + comp
    if dp == "A": return A
    if dp == "C": return np.where(S > 1, comp, A)
    calls = np.ceil(S / p["batch"])
    return np.maximum(A, np.minimum(calls * p["dP"] + comp, p["PW1"] - pd_["P0"]))


def power(par, dev, S, cls, view="ALL", dp="A", cx="logs", k=1.0, b=None):
    """b: optional concave capture term 100 * (p_rx + r_v c_v) * (S/100)^b (N2.4 sensitivity only)."""
    pd_ = par[dev]; v = VIEWS.index(view); S = np.asarray(S, float)
    cap = pd_["prx"] + par["rv"][v] * pd_["cv"]
    capt = S * k * cap if b is None else 100 * k * cap * (S / 100) ** b
    return pd_["P0"] + inference_power(pd_, cls, S, dp) + capt + S * F * cx_value(pd_, view, cx)


def e_stream_hour(par, dev, S, cls, view="ALL", dp="A", cx="logs", k=1.0):
    return 3600 * power(par, dev, S, cls, view, dp, cx, k) / np.asarray(S, float)


def crossover(par, cls, view="ALL", dp="A", cx="logs", Smax=1000):
    """Smallest S at which the Jetson needs less energy per stream than the Pi (None if not up to Smax)."""
    S = np.arange(1, Smax + 1)
    r = e_stream_hour(par, "jn", S, cls, view, dp, cx) < e_stream_hour(par, "pi", S, cls, view, dp, cx)
    return int(S[np.argmax(r)]) if r.any() else None


# ======================================================================= validation at intermediate loads (N2)
REF_RATIO = {}


def loss_ratio(dev, view, x):
    """Captured (or processed) packets / sent, relative to the loss-free ratio of the same view at 100 streams
    without dissection (same overlay construction). Loss-free: >= 0.98; onset: 0.95-0.98; saturated: < 0.95."""
    if (dev, view) not in REF_RATIO:
        # UL>100B: the frame-length filter keeps small packets that are discarded after dissection, so the
        # reference is the (loss-free, Table II) 100-stream run with dissection
        r = capture_runs(dev, view, 100, view == "UL>100B")
        REF_RATIO[(dev, view)] = (r.captured_packets / r.actual).median()
    cap = x["captured_packets"] if "captured_packets" in x and (x["captured_packets"] > 0).all() else x.get("processed_packets")
    if cap is None or (cap <= 0).any(): return np.nan
    return float((cap / x.actual).median() / REF_RATIO[(dev, view)])


def validation(par):
    rows = []
    for d in DEVS:
        pd_ = par[d]
        for view in VIEWS + ["FULL"]:
            vi = VIEWS.index("ALL" if view == "FULL" else view)
            for S in capture_loads(d, view, True):
                if d == "jn" and S > 200: continue      # beyond the replay capacity of the workstation (thesis author)
                if S == 1 or (S == 100 and view != "FULL"):
                    use = "calibration" if S == 100 and vi in CV_FIT[d] else ("single stream" if S == 1 else "100 (not fitted)")
                    if S == 100 and view != "FULL" and use == "calibration": continue
                else:
                    use = "intermediate"
                x = capture_runs(d, view, S, True)
                k = iqr_mask(x.tf_energy) if len(x) >= 4 else np.ones(len(x), bool)
                P = (x.tf_energy / x.tf_duration).to_numpy()[k]
                Pm = pd_["P0"] + S * (pd_["prx"] + par["rv"][vi] * pd_["cv"])
                rr = loss_ratio(d, "ALL" if view == "FULL" else view, x) if S > 1 else 1.0
                state = "loss-free" if rr >= 0.98 else ("onset" if rr >= 0.95 else "saturated")
                cpu = (x.ul_cpu_global_avg.to_numpy()[k] / 100 * NCORES[d]).mean()
                rows.append(dict(dev=d, view=view, S=S, n=int(k.sum()), use=use, rel_captured=rr, state=state,
                                 P_meas=P.mean(), P_model=Pm, err=(Pm - P.mean()) / P.mean() * 100,
                                 k=(P.mean() - pd_["P0"]) / (Pm - pd_["P0"]), cores=cpu,
                                 sem=P.std(ddof=1) / np.sqrt(len(P)) / P.mean() * 100 if len(P) > 1 else np.nan,
                                 month=pd.to_datetime(x.timestamp.median(), unit="s").strftime("%Y-%m")))
    return pd.DataFrame(rows)


def concave_fit(val, par):
    """N2.4: one extra parameter. Per-stream capture power grows as S^b instead of S (b fitted on the
    loss-free Pi points incl. the 100-stream calibration points of UL and UL>100B); returns b and residuals."""
    from scipy.optimize import minimize_scalar
    pd_ = par["pi"]
    pts = val[(val.dev == "pi") & (val.state == "loss-free") & (val.S > 1)]
    def resid(b):
        out = []
        for _, r in pts.iterrows():
            vi = VIEWS.index("ALL" if r["view"] == "FULL" else r["view"])
            per = pd_["prx"] + par["rv"][vi] * pd_["cv"]
            Pm = pd_["P0"] + per * 100 * (r.S / 100) ** b          # anchored at the 100-stream calibration
            out.append((Pm - r.P_meas) / r.P_meas * 100)
        return np.array(out)
    b = minimize_scalar(lambda b: (resid(b) ** 2).sum(), bounds=(0.5, 1.2), method="bounded").x
    return b, pts.assign(err_concave=resid(b))


# ======================================================================= CPU-time budget (N1.3)
def cpu_budget(par, val):
    """Core-time per stream-second of each stage and the resulting stream capacity."""
    out = {}
    for d in DEVS:
        o = out[d] = {}
        # dissection: slope of busy cores over S at loss-free loads (W3 CPU logs), ALL and UL view
        for view in ("ALL", "UL", "FULL"):
            p = val[(val.dev == d) & (val.view == view) & (val.state == "loss-free")]
            one = val[(val.dev == d) & (val.view == view) & (val.S == 1)]
            p = pd.concat([one, p]).drop_duplicates(["S"])
            if len(p) >= 2:
                o[f"td_{view}"] = np.polyfit(p.S, p.cores, 1)[0]
        # feature extraction: one core at 100 % in the offline runs -> core-seconds per stream-second
        o["tx_ALL"] = par[d]["tx"]["ALL"]; o["tx_UL"] = par[d]["tx"]["UL"]
        # inference: W1 wall time per prediction and busy cores during W1
        for c in CLASSES:
            src = CAND[d]["heavy" if c.endswith("32") else c]
            E, Tw, cores = w1_runs(d, *src, batch=c.endswith("32"))
            o[f"tm_{c}"] = Tw.mean() / 1e5; o[f"cores_{c}"] = cores.mean()
    return out


def capacity(cb, d, view, cls, window_idx):
    """(a) all stages serial in one single-threaded process; (b) one core per stage (bottleneck stage);
    (c) all core-time spread perfectly over all cores, dissection still single-threaded."""
    td = cb[d][f"td_{view}"]; tx = cb[d][f"tx_{view}"][window_idx]
    wall = cb[d][f"tm_{cls}"]; core = wall * cb[d][f"cores_{cls}"]
    gpu = d == "jn"
    # one-thread inference time: between W1 wall time and W1 core-time on the Pi (multi-threaded ONNX Runtime);
    # on the Jetson inference runs on the GPU and the W1 wall time is used, the CPU side counted as busy cores
    tm_lo, tm_hi = (wall, wall) if gpu else (wall, core)
    a = (1 / (td + tx + tm_hi), 1 / (td + tx + tm_lo))
    b = (min(1 / td, 1 / tx, 1 / tm_hi), min(1 / td, 1 / tx, 1 / tm_lo))
    tot_hi = td + tx + core
    c = min(1 / td, NCORES[d] / tot_hi)
    return a, b, c


# ======================================================================= report
def fmt(x, nd=1): return "None" if x is None else f"{x:.{nd}f}"


def main():
    inp = load_inputs()
    par = calibrate(means(inp))
    print("idle power (12 h):", {d: round(idle_power(d), 3) for d in DEVS}, "W")
    rv = np.array(list(par["rv"]), float)
    print("packet rate per stream [1/s] (pcap, 30 seeds):", dict(zip(VIEWS, np.round(rv, 2))),
          "| PK/(100*300 s):", dict(zip(VIEWS, np.round(pcap_packets().mean()[VIEWS].to_numpy() / 3e4, 2))))
    for d in DEVS:
        p = par[d]
        print(f"{d}: P0={p['P0']:.3f} W  p_rx={p['prx']*1e3:.2f} mW  c_v={p['cv']*1e6:.0f} uJ/pkt  "
              f"(r_v c_v ALL = {rv[0]*p['cv']*1e3:.1f} mW/stream; intercept {p['d0']:.3f} W vs P0+100p_rx {p['P0']+100*p['prx']:.3f} W)")
        op = np.array([processing_energy(d, v)[0] for v in VIEWS]); pk = 100 * rv * T5
        cop = np.polyfit(pk, op, 1)[0]
        print(f"   deferred processing of the stored 100-stream capture: {cop*1e6:.0f} uJ per packet "
              f"(whole-device energy, slope over the four views; {cop/p['cv']:.1f} x c_v); durations "
              + ", ".join(f"{v} {processing_energy(d, v)[1]:.0f} s" for v in VIEWS))
        for i, v in enumerate(VIEWS):
            m = inp[d]["d100"][v].mean(); pm = p["P0"] + 100 * (p["prx"] + rv[i] * p["cv"])
            print(f"   100 streams {v:9s} measured {m:.3f} W model {pm:.3f} W ({(pm-m)/m*100:+.1f} %)")
        for v in VIEWS:
            print(f"   c_x {v:9s} " + "  ".join(f"{w:2d}s {c*1e3:5.1f} mJ / {t*1e3:5.1f} ms" for w, c, t in zip(WINDOWS, p['cx'][v], p['tx'][v]))
                  + "  per stream-second")
        for c in CLASSES:
            q = p["inf"][c]
            print(f"   {c:16s} dP_m={q['dP']*1e3:6.1f} mW  c_m={q['cm']*1e3:7.3f} mJ  W1 power {q['PW1']:.3f} W  "
                  f"W1 wall time/prediction {q['tm']*1e3:.3f} ms")
    # stream-seconds check
    for d in DEVS:
        print(f"{d} feature updates per offline run (100 files):",
              {f"{v} {w}s": int(fx_offline_runs(d, v, w).total_output_elements.mean()) for v in VIEWS for w in WINDOWS})
    # N1.2: single-stream check of c_x against W2 reference minus W3 single stream with dissection
    print("N1.2: W2 reference (features, no inference) minus W3 single stream with dissection [J per 300 s], 95 % CI (Welch):")
    LV = execution("live")
    for d in DEVS:
        for v in VIEWS:
            lv = LIVEVIEW[v]
            ref = LV[(LV.dev == d) & (LV.view == lv) & (LV["sub"] == "feature_extraction")]
            ref = ref.groupby(["kqi", "fam", "variant"]).apply(lambda g: (g.tf_energy / g.tf_duration)[iqr_mask(g.tf_energy)].mean() * T5,
                                                               include_groups=False)
            w3 = capture_power_runs(d, v, 1, True) * T5
            diff = ref.mean() - w3.mean()
            se = np.sqrt(ref.var(ddof=1) / len(ref) + w3.var(ddof=1) / len(w3))
            exp = [c * T5 for c in (par[d]["cx"][v][0], par[d]["cx"][v][2])]
            print(f"   {d} {v:9s} W2 ref {ref.mean():6.1f} J (n={len(ref)} candidates) - W3 {w3.mean():6.1f} J = {diff:+6.1f} "
                  f"[{diff-1.96*se:+.1f}, {diff+1.96*se:+.1f}] J; expected from c_x: {exp[0]:.1f}-{exp[1]:.1f} J")
    # live cross-check of the feature-extraction CPU time: process CPU of the W2 reference runs (no inference)
    for d in DEVS:
        out = []
        for v in VIEWS:
            r = LV[(LV.dev == d) & (LV.view == LIVEVIEW[v]) & (LV["sub"] == "feature_extraction")]
            g = r.groupby(["kqi", "fam", "variant"]).ul_cpu_process_avg.mean() / 100 * NCORES[d] * 1e3
            out.append(f"{v} {g.median():.0f} ({g.min():.0f}-{g.max():.0f})")
        print(f"   {d} W2 reference, pipeline process CPU [ms per stream-second], median (range over candidates): " + ", ".join(out))
    # live feature-extraction campaign (same pipeline, windows 1/10/60 s): energy and process CPU vs window
    for d in DEVS:
        r = {w: fx_live_runs(d, "ALL", w) for w in WINDOWS}
        e = {w: (x.tf_energy / x.tf_duration).mean() * T5 for w, x in r.items()}
        cpu = {w: x.ul_cpu_process_avg.mean() / 100 * NCORES[d] * 1e3 for w, x in r.items()}
        print(f"   {d} live feature-extraction campaign, ALL: energy per 300 s " + ", ".join(f"{w}s {e[w]:.0f} J" for w in WINDOWS)
              + "; process CPU " + ", ".join(f"{w}s {cpu[w]:.0f} ms/s" for w in WINDOWS))

    # N2 validation
    val = validation(par)
    pd.set_option("display.width", 200)
    print("\nN2: capture with dissection, model vs measured (power; err = (model - measured)/measured):")
    print(val[["dev", "view", "S", "n", "use", "month", "rel_captured", "state", "P_meas", "P_model", "err", "sem", "k", "cores"]]
          .round({"rel_captured": 3, "P_meas": 3, "P_model": 3, "err": 1, "sem": 1, "k": 2, "cores": 2}).to_string(index=False))
    for d in DEVS:
        for st in ("loss-free", "onset", "saturated"):
            q = val[(val.dev == d) & (val.state == st) & (val.use == "intermediate")]
            if len(q): print(f"   {d} intermediate {st:9s}: err {q.err.min():+.1f} .. {q.err.max():+.1f} %  (n points {len(q)}); k {q.k.min():.2f} .. {q.k.max():.2f}")
        q = val[(val.dev == d) & (val.state == "loss-free") & (val.S > 1)]
        print(f"   {d} all loss-free S>1 (incl. 100-stream points not fitted): err {q.err.min():+.1f} .. {q.err.max():+.1f} %, "
              f"median |err| {q.err.abs().median():.1f} %")
    b, cc = concave_fit(val, par)
    print(f"N2.4 concave per-stream term P0 + p_S*100*(S/100)^b: b = {b:.3f}; Pi loss-free residuals linear "
          f"{cc.err.min():+.1f}..{cc.err.max():+.1f} % (rms {np.sqrt((cc.err**2).mean()):.1f}) -> concave "
          f"{cc.err_concave.min():+.1f}..{cc.err_concave.max():+.1f} % (rms {np.sqrt((cc.err_concave**2).mean()):.1f})")

    # campaign offsets and within-campaign marginal cost
    for d in DEVS:
        for view in ("ALL", "UL", "FULL"):
            q = val[(val.dev == d) & (val.view == view) & (val.state == "loss-free") & (val.use == "intermediate") & (val.n >= 8)]
            if len(q) >= 2:
                vi = VIEWS.index("ALL" if view == "FULL" else view)
                print(f"   {d} {view}: marginal power per stream within the later campaign (S = {q.S.min()}-{q.S.max()}): "
                      f"{np.polyfit(q.S, q.P_meas, 1)[0]*1e3:.1f} mW vs model p_rx + r_v c_v = {(par[d]['prx']+par['rv'][vi]*par[d]['cv'])*1e3:.1f} mW")
    import os
    for S in (1, 100):
        f = capture_file("jn", "ALL", S, False).replace("_stats.csv", "_stats_new.csv")
        if os.path.exists(f):
            a = pd.read_csv(f); b0 = capture_runs("jn", "ALL", S, False)
            print(f"   jn campaign offset, capture without dissection, {S} streams: Oct 2026 {(a.tf_energy/a.tf_duration).mean():.3f} W "
                  f"vs Jun 2026 {(b0.tf_energy/b0.tf_duration).mean():.3f} W (fan {a.jn_fan_speed_avg.mean():.0f} vs {b0.jn_fan_speed_avg.mean():.0f} %)")
    for c in ("heavy", "light"):
        Sg = np.arange(1, 1001)
        r = e_stream_hour(par, "jn", Sg, c) < 3600 * power(par, "pi", Sg, c, b=b) / Sg
        print(f"   N2.4 sensitivity: crossover {c} ALL with the concave Pi capture term: {Sg[np.argmax(r)]} (linear: {crossover(par, c)})")

    # N1.5 / N1.3 results
    print("\nResults (scenario A, c_x from the 10 s window unless stated):")
    for cx in ("zero", 1, 10, 60):
        for view in ("ALL", "UL"):
            print(f"   crossover c_x={cx!s:4s} {view:3s}: " + ", ".join(f"{c} {crossover(par, c, view, cx=cx)}" for c in CLASSES))
    for c in CLASSES:
        s = []
        for S in (1, 20, 60):
            s.append(f"S={S}: Pi {float(e_stream_hour(par, 'pi', S, c)):.0f} / Jetson {float(e_stream_hour(par, 'jn', S, c)):.0f}")
        print(f"   e(S) [J per stream-hour] {c:16s} " + "; ".join(s))
    for c in CLASSES:
        s = []
        for S in (60,):
            s.append(f"c_x=0: Pi {float(e_stream_hour(par, 'pi', S, c, cx='zero')):.0f} / Jetson {float(e_stream_hour(par, 'jn', S, c, cx='zero')):.0f}")
        print(f"   e(60) old model {c:16s} " + "; ".join(s))
    sv = 1 - e_stream_hour(par, "pi", 60, "heavy, batch 32") / e_stream_hour(par, "pi", 60, "heavy")
    sv0 = 1 - e_stream_hour(par, "pi", 60, "heavy, batch 32", cx="zero") / e_stream_hour(par, "pi", 60, "heavy", cx="zero")
    print(f"   batching saving per stream, Pi, 60 ALL streams: {float(sv)*100:.1f} % (c_x=0: {float(sv0)*100:.1f} %)")
    for d in DEVS:
        p = par[d]
        for c in CLASSES:
            q = p["inf"][c]
            per = p["prx"] + rv[0] * p["cv"] + F * p["cx"]["ALL"][1] + F * q["cm"]
            per0 = p["prx"] + rv[0] * p["cv"] + F * q["cm"]
            P60 = float(power(par, d, 60, c))
            print(f"   {d} {c:16s}: S* = {(p['P0']+q['dP'])/per:.1f} (c_x=0: {(p['P0']+q['dP'])/per0:.1f}); baseline share S=1 "
                  f"{p['P0']/float(power(par, d, 1, c))*100:.1f} %; at 60 streams inference {(q['dP']+60*q['cm'])/P60*100:.1f} %, "
                  f"features {60*p['cx']['ALL'][1]/P60*100:.1f} % (1-60 s: {60*p['cx']['ALL'][0]/P60*100:.1f}-{60*p['cx']['ALL'][2]/P60*100:.1f} %), "
                  f"dissection+receive {60*(p['prx']+rv[0]*p['cv'])/P60*100:.1f} %")
    print("   per stream-hour [J], ALL: " + "; ".join(
        f"{d}: receive {3600*par[d]['prx']:.0f}, dissection {3600*rv[0]*par[d]['cv']:.0f}, features {3600*par[d]['cx']['ALL'][0]:.0f}-{3600*par[d]['cx']['ALL'][2]:.0f} (10 s: {3600*par[d]['cx']['ALL'][1]:.0f}), "
        f"heavy inference {3600*par[d]['inf']['heavy']['cm']:.0f}, batch {3600*par[d]['inf']['heavy, batch 32']['cm']:.0f}, baseline {3600*par[d]['P0']:.0f}" for d in DEVS))

    cb = cpu_budget(par, val)
    print("\nN1.3 CPU-time budget [ms of core time per stream-second]:")
    for d in DEVS:
        o = cb[d]
        print(f"   {d}: dissection ALL {o['td_ALL']*1e3:.1f}, UL {o['td_UL']*1e3:.1f}" + (f", full packets {o['td_FULL']*1e3:.1f}" if 'td_FULL' in o else "")
              + f" -> one core: {1/o['td_ALL']:.0f} ALL / {1/o['td_UL']:.0f} UL streams"
              + f"; features ALL {o['tx_ALL'][0]*1e3:.1f}-{o['tx_ALL'][2]*1e3:.1f}, UL {o['tx_UL'][0]*1e3:.1f}-{o['tx_UL'][2]*1e3:.1f}; "
              + "; ".join(f"{c}: W1 wall {o['tm_'+c]*1e3:.2f} ms x {o['cores_'+c]:.2f} cores" for c in CLASSES))
        for view in ("ALL", "UL"):
            for c in CLASSES:
                res = [capacity(cb, d, view, c, wi) for wi in (0, 2)]
                a = (min(r[0][0] for r in res), max(r[0][1] for r in res)); b_ = (min(r[1][0] for r in res), max(r[1][1] for r in res))
                cc_ = (min(r[2] for r in res), max(r[2] for r in res))
                print(f"      {view:3s} {c:16s} capacity (a) single thread {a[0]:.0f}-{a[1]:.0f}; (b) core per stage {b_[0]:.0f}-{b_[1]:.0f}; "
                      f"(c) all cores {cc_[0]:.0f}-{cc_[1]:.0f} streams")
    return inp, par, val, cb


if __name__ == "__main__":
    main()
