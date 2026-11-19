"""Uncertainty analysis of the energy model (N3).

Parameter uncertainty: every primitive measurement series of energy_model.load_inputs() (30 runs per configuration,
30 replay seeds) is bootstrapped independently, and the model parameters are recomputed from the resampled means
(energy_model.calibrate). Two further factors are drawn:
  u_cx  position of the feature window between 1 s and 60 s (uniform): c_x = c_x(1 s) + u (c_x(60 s) - c_x(1 s))
  k     factor on the per-stream capture term S (p_rx + r_v c_v), drawn from the residual factors of the
        loss-free intermediate loads (energy_model.validation, >= 8 runs, S >= 20)
Structural uncertainty: enumerated scenarios (energy_model.inference_power): dP_m paid once (A) / per inference call up
to the W1 active power (B) / absent beyond one stream (C); c_x from the logs or c_x = 0; batching is a model class.

Reusable for a global sensitivity analysis: sample_parameters(N, seed, fixed=...) returns parameter draws whose
GROUPS are mutually independent, so pick-freeze (Sobol/Saltelli) designs can be built by drawing two samples A, B and
swapping one group at a time (swap_group). Outputs are vectorized over draws (shape (N, 1)) and stream counts.
"""
import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import energy_model as em

SEED = 2027
N = 10_000
SMAX = 1000
DPS = ("A", "B", "C")
CXS = ("logs", "zero")
OUT_FIG = "../figures/energy_model.pdf"

# independent groups: (device, key in load_inputs) or a drawn factor
GROUPS = {f"{d}:{g}": (d, g) for d in em.DEVS for g in ("nd1", "nd100", "d100", "w1", "w2", "fx")}
GROUPS.update({"pcap": ("pcap", None)}, **{f"{d}:u_cx": (d, "u_cx") for d in em.DEVS}, **{f"{d}:k": (d, "k") for d in em.DEVS})


def residual_factors(val):
    q = val[(val.state == "loss-free") & (val.use == "intermediate") & (val.n >= 8) & (val.S >= 20)]
    return {d: q[q.dev == d].k.to_numpy() for d in em.DEVS}


_CACHE = {}


def _setup():
    if not _CACHE:
        inp = em.load_inputs(); pt = em.means(inp); par = em.calibrate(pt)
        _CACHE.update(inp=inp, point_means=pt, point=par, K=residual_factors(em.validation(par)))
    return _CACHE


def _col(x, N):
    """Reshape draw arrays to (N, 1) so that they broadcast against a vector of stream counts."""
    if isinstance(x, dict): return {k: _col(v, N) for k, v in x.items()}
    if isinstance(x, list): return [_col(v, N) for v in x]
    if isinstance(x, np.ndarray) and x.dtype == object: return [_col(v, N) for v in x]
    if isinstance(x, np.ndarray) and x.ndim == 1 and len(x) == N: return x[:, None]
    return x


def sample_primitives(N, seed=SEED):
    """Bootstrap means of all primitive series plus the drawn factors u_cx and k; dict keyed like GROUPS."""
    c = _setup(); rng = np.random.default_rng(seed)
    m = em.means(c["inp"], N, rng)
    for d in em.DEVS:
        m[d]["u_cx"] = rng.random(N)
        m[d]["k"] = rng.choice(c["K"][d], N)
    return m


def swap_group(mA, mB, group):
    """Copy of draws mA with one independent group taken from mB (pick-freeze designs)."""
    import copy
    out = copy.copy(mA); d, g = GROUPS[group]
    if d == "pcap": out["pcap"] = mB["pcap"]; return out
    out[d] = dict(mA[d]); out[d][g] = mB[d][g]; return out


def parameters(m, N, fixed=()):
    """Model parameters from primitive draws; groups listed in `fixed` are held at their point values."""
    c = _setup(); m = dict(m)
    for grp in fixed:
        d, g = GROUPS[grp]
        if d == "pcap": m["pcap"] = c["point_means"]["pcap"]
        elif g == "u_cx": m[d] = dict(m[d], u_cx=np.full(N, 0.5))
        elif g == "k": m[d] = dict(m[d], k=np.ones(N))
        else: m[d] = dict(m[d], **{g: c["point_means"][d][g]})
    par = em.calibrate(m)
    for d in em.DEVS:
        par[d]["u_cx"] = m[d]["u_cx"]; par[d]["k"] = m[d]["k"]
    par["N"] = N
    return _col(par, N)


def sample_parameters(N=N, seed=SEED, fixed=()):
    return parameters(sample_primitives(N, seed), N, fixed)


# ----------------------------------------------------------------------------- outputs
def e_draws(par, dev, S, cls, view="ALL", dp="A", cx="logs"):
    pd_ = par[dev]
    cxa = pd_["u_cx"] if cx == "logs" else "zero"
    return em.e_stream_hour(par, dev, S, cls, view, dp, cxa, k=pd_["k"])


def crossover_draws(par, cls, view="ALL", dp="A", cx="logs", smax=SMAX, chunk=2500):
    """First S at which the Jetson needs less energy per stream than the Pi, per draw (inf if not up to smax)."""
    S = np.arange(1, smax + 1, dtype=float); n = par["N"]; out = np.empty(n)
    for i in range(0, n, chunk):
        sub = _slice(par, i, min(i + chunk, n))
        r = e_draws(sub, "jn", S, cls, view, dp, cx) < e_draws(sub, "pi", S, cls, view, dp, cx)
        first = np.argmax(r, axis=1).astype(float)
        first[~r.any(axis=1)] = np.inf
        out[i:i + chunk] = S[first.astype(int).clip(0, smax - 1)] * np.isfinite(first) + np.where(np.isfinite(first), 0, np.inf)
    return out


def _slice(x, a, b):
    if isinstance(x, dict): return {k: (_slice(v, a, b) if k != "N" else b - a) for k, v in x.items()}
    if isinstance(x, list): return [_slice(v, a, b) for v in x]
    if isinstance(x, np.ndarray) and x.ndim == 2 and x.shape[1] == 1: return x[a:b]
    return x


def q(x, p): return np.percentile(x, p) if np.isfinite(np.percentile(x, p)) else np.inf


def fmt_range(x):
    f = lambda v: ">1000" if not np.isfinite(v) else f"{v:.0f}"
    return f"{f(q(x, 50))} [{f(q(x, 5))}, {f(q(x, 95))}]"


def oat_ranking(cls="heavy", view="ALL", dp="A"):
    """One-at-a-time: each derived parameter at point +/- 1 SD of its bootstrap distribution, others at point
    (c_x at the window midpoint, k = 1); effect = change of the crossover stream count."""
    c = _setup(); n = 4000
    draws = sample_parameters(n, SEED + 1)
    base = _col(em.calibrate(c["point_means"]), 1)
    for d in em.DEVS: base[d]["u_cx"] = np.full((1, 1), 0.5); base[d]["k"] = np.ones((1, 1))
    base["N"] = 1
    import copy
    def cross(p): return crossover_draws(p, cls, view, dp)[0]
    x0 = cross(base); res = []
    def sd(a): return float(np.std(a))
    items = []
    for d in em.DEVS:
        items += [(f"{d} P0", lambda p, s, d=d: p[d].__setitem__("P0", p[d]["P0"] + s * sd(draws[d]["P0"])))]
        items += [(f"{d} p_rx", lambda p, s, d=d: p[d].__setitem__("prx", p[d]["prx"] + s * sd(draws[d]["prx"])))]
        items += [(f"{d} c_v", lambda p, s, d=d: p[d].__setitem__("cv", p[d]["cv"] + s * sd(draws[d]["cv"])))]
        items += [(f"{d} c_x window", lambda p, s, d=d: p[d].__setitem__("u_cx", p[d]["u_cx"] + s * np.sqrt(1 / 12)))]
        items += [(f"{d} c_x runs", lambda p, s, d=d: p[d].__setitem__("cx", {v: [cc + s * sd(dd) for cc, dd in zip(p[d]["cx"][v], draws[d]["cx"][v])] for v in em.VIEWS}))]
        items += [(f"{d} residual k", lambda p, s, d=d: p[d].__setitem__("k", p[d]["k"] + s * sd(c["K"][d])))]
        items += [(f"{d} dP_m", lambda p, s, d=d: p[d]["inf"][cls].__setitem__("dP", p[d]["inf"][cls]["dP"] + s * sd(draws[d]["inf"][cls]["dP"])))]
        items += [(f"{d} c_m", lambda p, s, d=d: p[d]["inf"][cls].__setitem__("cm", p[d]["inf"][cls]["cm"] + s * sd(draws[d]["inf"][cls]["cm"])))]
    items += [("r_v", lambda p, s: p.__setitem__("rv", [r + s * sd(dd) for r, dd in zip(p["rv"], draws["rv"])]))]
    for name, f in items:
        vals = []
        for s in (-1, 1):
            p = copy.deepcopy(base); f(p, s); vals.append(cross(p))
        res.append((name, vals[0], vals[1], max(abs(vals[0] - x0), abs(vals[1] - x0))))
    return x0, sorted(res, key=lambda r: -r[3])


def figure(par, fname=OUT_FIG):
    plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.linewidth": 0.5,
                         "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, ax = plt.subplots(figsize=(3.45, 2.2))
    cols = {"light": "#2a78d6", "heavy": "#eb6834", "heavy, batch 32": "#1baf7a"}
    ls = {"light": "-", "heavy": "--", "heavy, batch 32": ":"}
    S = np.arange(1, 121, dtype=float)
    labels = {"light": "light", "heavy": "heavy", "heavy, batch 32": "heavy, batch 32"}
    for kind, col in cols.items():
        r = e_draws(par, "jn", S, kind) / e_draws(par, "pi", S, kind)
        lo, med, hi = np.percentile(r, [5, 50, 95], axis=0)
        ax.fill_between(S, lo, hi, color=col, alpha=0.15, lw=0)
        ax.plot(S, med, color=col, lw=1.3, ls=ls[kind], label=labels[kind])
    r0 = em.e_stream_hour(_CACHE["point"], "jn", S, "heavy", cx="zero") / em.e_stream_hour(_CACHE["point"], "pi", S, "heavy", cx="zero")
    ax.plot(S, r0, color="#52514e", lw=0.8, ls=(0, (1, 1.2)), label="heavy, $c_x = 0$")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.01, 0.99), ncol=2, fontsize=6.3, frameon=False, handlelength=2.2,
              borderaxespad=0.1, labelspacing=0.15, columnspacing=1.2)
    ax.axhline(1, color="#52514e", lw=0.6)
    ax.axvspan(60, 120, color="#e9e8e4", lw=0, zorder=0)
    ax.text(62, 1.66, "beyond Pi capture\ncapacity (ALL)", fontsize=6.5, color="#52514e", va="top")
    ax.set_xlim(1, 120); ax.set_ylim(0.6, 1.75)
    ax.set_xticks([1, 20, 40, 60, 80, 100, 120])
    ax.set_xlabel("Concurrent streams $S$ (ALL view)")
    ax.set_ylabel("Energy per stream,\nJetson / Raspberry Pi")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.grid(True, lw=0.3, color="#d9d8d4"); ax.set_axisbelow(True)
    fig.tight_layout(pad=0.3)
    fig.savefig(fname)
    # Removed for publication: the original line also wrote a PNG copy to an absolute
    # scratch directory on the authors' machine, which raises FileNotFoundError anywhere
    # else. See the modifications note in data/README.md.
    fig.savefig(fname.replace(".pdf", ".png"), dpi=300)


def main():
    import time; t0 = time.time()
    c = _setup()
    print("residual factors k (loss-free intermediate loads, >= 8 runs, S >= 20):",
          {d: np.round(np.sort(k), 2).tolist() for d, k in c["K"].items()})
    par = sample_parameters(N, SEED)
    # parameter spread
    for d in em.DEVS:
        p = par[d]
        f = lambda a, s=1: f"{np.median(a)*s:.4g} [{np.percentile(a,5)*s:.4g}, {np.percentile(a,95)*s:.4g}]"
        cxs = p["cx"]["ALL"][0] + p["u_cx"] * (p["cx"]["ALL"][2] - p["cx"]["ALL"][0])
        print(f"{d}: P0 {f(p['P0'])} W; p_rx {f(p['prx'],1e3)} mW; c_v {f(p['cv'],1e6)} uJ; c_x ALL (window+runs) {f(cxs,1e3)} mJ; "
              f"dP heavy {f(p['inf']['heavy']['dP'],1e3)} mW; c_m heavy {f(p['inf']['heavy']['cm'],1e3)} mJ; "
              f"c_m batch {f(p['inf']['heavy, batch 32']['cm'],1e3)} mJ; light dP {f(p['inf']['light']['dP'],1e3)} mW")
    print("r_v:", [f"{np.median(r):.2f} [{np.percentile(r,5):.2f}, {np.percentile(r,95):.2f}]" for r in par["rv"]])

    print("\nCrossover S (Jetson cheaper per stream from S), median [5 %, 95 %]:")
    cross = {}
    for view in ("ALL", "UL"):
        for cls in em.CLASSES:
            row = []
            for cx in CXS:
                for dp in DPS:
                    x = crossover_draws(par, cls, view, dp, cx); cross[(view, cls, dp, cx)] = x
                    row.append(f"{dp}/{cx}: {fmt_range(x)}")
            print(f"   {view:3s} {cls:16s} " + "; ".join(row))

    print("\nP(Pi cheaper per stream) at S = 1 / 20 / 60 (ALL view):")
    for cls in em.CLASSES:
        row = []
        for cx in CXS:
            for dp in DPS:
                pr = [float((e_draws(par, 'pi', S, cls, 'ALL', dp, cx) < e_draws(par, 'jn', S, cls, 'ALL', dp, cx)).mean()) for S in (1, 20, 60)]
                row.append(f"{dp}/{cx}: " + "/".join(f"{v:.2f}" for v in pr))
        print(f"   {cls:16s} " + "; ".join(row))
    print("e(S) [J per stream-hour], scenario A, c_x from logs, median [5 %, 95 %]:")
    for cls in em.CLASSES:
        print(f"   {cls:16s} " + "; ".join(
            f"S={S}: Pi {fmt_range(e_draws(par,'pi',S,cls)[:,0])} / Jetson {fmt_range(e_draws(par,'jn',S,cls)[:,0])}" for S in (1, 20, 60)))

    print("\nG5 (live part): saving per stream from batching 32 predictions per call, Pi, 60 ALL streams:")
    g5 = {}
    for cx in CXS:
        for dp in DPS:
            s = 1 - e_draws(par, "pi", 60, "heavy, batch 32", "ALL", dp, cx)[:, 0] / e_draws(par, "pi", 60, "heavy", "ALL", dp, cx)[:, 0]
            g5[(dp, cx)] = s
            print(f"   {dp}/{cx}: {np.median(s)*100:.1f} % [{np.percentile(s,5)*100:.1f}, {np.percentile(s,95)*100:.1f}]; "
                  f"P(saving >= 5 %) {np.mean(s >= 0.05):.3f}; P(10-18 %) {np.mean((s >= 0.10) & (s <= 0.18)):.3f}")

    for S in (30,):
        for cx in CXS:
            for dp in DPS:
                s_ = 1 - e_draws(par, "pi", S, "heavy, batch 32", "ALL", dp, cx)[:, 0] / e_draws(par, "pi", S, "heavy", "ALL", dp, cx)[:, 0]
                print(f"   at {S} streams {dp}/{cx}: {np.median(s_)*100:.1f} % [{np.percentile(s_,5)*100:.1f}, {np.percentile(s_,95)*100:.1f}]")
    print("\nRobustness (claim holds in >= 95 % of draws in every structural scenario):")
    def label(probs): return "robust" if min(probs) >= 0.95 else "fragile"
    claims = {
        "G2 heavy: Pi cheaper per stream up to 70 ALL streams": [np.mean(cross[("ALL", "heavy", dp, cx)] > 70) for dp in DPS for cx in CXS],
        "G2 heavy: Pi cheaper per stream up to 107 UL streams": [np.mean(cross[("UL", "heavy", dp, cx)] > 107) for dp in DPS for cx in CXS],
        "G2 light: Pi cheaper up to its capture capacity (60 ALL)": [np.mean(cross[("ALL", "light", dp, cx)] > 60) for dp in DPS for cx in CXS],
        "G2 light: Pi cheaper up to its capture capacity (170 UL)": [np.mean(cross[("UL", "light", dp, cx)] > 170) for dp in DPS for cx in CXS],
        "G5 live: batching saves 10-18 % per stream (Pi, 60 ALL)": [np.mean((g5[k] >= 0.10) & (g5[k] <= 0.18)) for k in g5],
        "G5 live: batching saves >= 5 % per stream (Pi, 60 ALL)": [np.mean(g5[k] >= 0.05) for k in g5],
        "G5 live: batching saves >= 9 % per stream (Pi, 60 ALL)": [np.mean(g5[k] >= 0.09) for k in g5],
    }
    for cls in em.CLASSES:
        for view in ("ALL", "UL"):
            lo = min(q(cross[(view, cls, dp, cx)], 5) for dp in DPS for cx in ("logs",))
            claims[f"Pi cheaper per stream up to S < {lo:.0f} ({cls}, {view}, c_x from logs, all dP scenarios)"] = \
                [np.mean(cross[(view, cls, dp, "logs")] > lo - 1) for dp in DPS]
            lo1 = min(q(cross[(view, cls, dp, cx)], 5) for dp in DPS for cx in CXS)
            claims[f"Pi cheaper per stream up to S < {lo1:.0f} ({cls}, {view}, all scenarios)"] = \
                [np.mean(cross[(view, cls, dp, cx)] > lo1 - 1) for dp in DPS for cx in CXS]
        for view in ("ALL", "UL"):
            lo2 = min(q(cross[(view, cls, dp, cx)], 5) for dp in ("A", "C") for cx in CXS)
            claims[f"Pi cheaper per stream up to S < {lo2:.0f} ({cls}, {view}, scenarios A and C, c_x logs or 0)"] = \
                [np.mean(cross[(view, cls, dp, cx)] > lo2 - 1) for dp in ("A", "C") for cx in CXS]
    claims["Pi cheaper per stream at 20 ALL streams, all classes"] = [
        float((e_draws(par, 'pi', 20, cl, 'ALL', dp, cx) < e_draws(par, 'jn', 20, cl, 'ALL', dp, cx)).mean()) for cl in em.CLASSES for dp in DPS for cx in CXS]
    for k, v in claims.items():
        print(f"   {label(v):7s} {k}: min share over scenarios {min(v):.3f} (max {max(v):.3f})")

    print("\nOne-at-a-time ranking (crossover, ALL view, scenario A, c_x window midpoint, k = 1; +/- 1 SD):")
    for cls in ("heavy", "light"):
        x0, rk = oat_ranking(cls)
        print(f"   {cls}: base crossover {x0:.0f}; " + "; ".join(f"{n} {a:.0f}/{b:.0f}" for n, a, b, _ in rk[:8]))

    figure(par)
    print(f"saved figure {OUT_FIG}; runtime {time.time()-t0:.1f} s")


if __name__ == "__main__":
    main()
