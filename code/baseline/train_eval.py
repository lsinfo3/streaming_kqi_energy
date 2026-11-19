"""Same-split linear baselines for stalling (logistic regression) and buffer health (ridge).

Features: log1p, then standardized on the training split. Hyperparameters chosen on the
validation split (AP for stalling, R^2 for buffer health); the F1 threshold is fixed on
validation. Test metrics with a run-level bootstrap (runs resampled with replacement).
"""
import sys, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, f1_score, r2_score, mean_absolute_error
sys.path.insert(0, str(Path(__file__).parent))
from features import feature_names

DATA = Path(sys.argv[1]); OUT = Path(sys.argv[2])
RNG = np.random.default_rng(1234)
NAMES = feature_names()

def cols(view, counts_only=False):
    idx = []
    for i, n in enumerate(NAMES):
        d = "_ul_" in n
        gt = n.endswith("_gt_100")
        ok = {"ALL": True, "ALL>100B": gt, "UL": d, "UL>100B": d and gt}[view]
        if counts_only:
            ok = ok and ("packet_count" in n or "total_bytes" in n)
        if ok: idx.append(i)
    return np.array(idx)

def load(stage):
    X = np.load(DATA / f"{stage}_X.npy", mmap_mode="r")
    return X, np.load(DATA / f"{stage}_stalling.npy"), np.load(DATA / f"{stage}_bh.npy"), np.load(DATA / f"{stage}_run.npy")

def prep(X, idx, mu=None, sd=None):
    Z = np.log1p(np.asarray(X[:, idx], dtype=np.float64))
    if mu is None:
        mu, sd = Z.mean(0), Z.std(0); sd[sd == 0] = 1.0
    return (Z - mu) / sd, mu, sd

def best_f1_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f = 2 * pr * rc / np.maximum(pr + rc, 1e-12)
    return th[np.nanargmax(f[:-1])]

def boot(fn, run, B=200):
    runs = np.unique(run); pos = {r: np.where(run == r)[0] for r in runs}
    vals = []
    for _ in range(B):
        pick = RNG.choice(runs, len(runs), replace=True)
        ix = np.concatenate([pos[r] for r in pick]); vals.append(fn(ix))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

Xtr, Str, Btr, Rtr = load("training"); Xva, Sva, Bva, Rva = load("validation"); Xte, Ste, Bte, Rte = load("test")
info = {s: {"samples": int(len(y)), "runs": int(len(np.unique(r))), "stalling_share": float(y.mean()),
            "bh_valid": int(np.isfinite(b).sum())}
        for s, y, b, r in (("training", Str, Btr, Rtr), ("validation", Sva, Bva, Rva), ("test", Ste, Bte, Rte))}
print(json.dumps(info, indent=1), flush=True)

N_FIT = 1_000_000                       # seeded subsample of the training split for fitting
fit = np.sort(RNG.choice(len(Str), size=min(N_FIT, len(Str)), replace=False))
results = {"data": info, "n_fit": int(len(fit)), "stalling": {}, "buffer_health": {}}

for view in ("ALL", "ALL>100B", "UL", "UL>100B"):
    for counts_only in (False, True):
        key = view + (" counts" if counts_only else "")
        idx = cols(view, counts_only)
        Ztr, mu, sd = prep(Xtr[fit], idx); Zva, _, _ = prep(Xva, idx, mu, sd); Zte, _, _ = prep(Xte, idx, mu, sd)
        # stalling: logistic regression
        best = None
        for C in (0.01, 0.1, 1.0, 10.0):
            m = LogisticRegression(C=C, max_iter=2000).fit(Ztr, Str[fit])
            ap = average_precision_score(Sva, m.predict_proba(Zva)[:, 1])
            if best is None or ap > best[0]: best = (ap, C, m)
        ap_va, C, m = best
        pva, pte = m.predict_proba(Zva)[:, 1], m.predict_proba(Zte)[:, 1]
        th = best_f1_threshold(Sva, pva)
        yhat = (pte >= th).astype(int)
        results["stalling"][key] = {
            "features": int(len(idx)), "C": C, "val_AP": float(ap_va), "threshold": float(th),
            "test_AP": float(average_precision_score(Ste, pte)),
            "test_AP_ci": boot(lambda ix: average_precision_score(Ste[ix], pte[ix]), Rte),
            "test_F1": float(f1_score(Ste, yhat)),
            "test_F1_ci": boot(lambda ix: f1_score(Ste[ix], yhat[ix]), Rte)}
        # buffer health: ridge regression in closed form on the full training split
        okt = np.isfinite(Btr)
        ZZ = np.zeros((len(idx) + 1, len(idx) + 1)); Zy = np.zeros(len(idx) + 1)
        for s in range(0, len(Btr), 500_000):
            sl = slice(s, s + 500_000); k = okt[sl]
            Z, _, _ = prep(Xtr[sl], idx, mu, sd); Z = np.hstack([Z[k], np.ones((k.sum(), 1))])
            ZZ += Z.T @ Z; Zy += Z.T @ Btr[sl][k]
        okv, oke = np.isfinite(Bva), np.isfinite(Bte)
        Zva1 = np.hstack([Zva, np.ones((len(Zva), 1))]); Zte1 = np.hstack([Zte, np.ones((len(Zte), 1))])
        best = None
        for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
            reg = alpha * np.eye(len(idx) + 1); reg[-1, -1] = 0
            w = np.linalg.solve(ZZ + reg, Zy)
            r2 = r2_score(Bva[okv], np.clip(Zva1[okv] @ w, 0, 1))
            if best is None or r2 > best[0]: best = (r2, alpha, w)
        r2_va, alpha, w = best
        yte = np.clip(Zte1[oke] @ w, 0, 1); gt = Bte[oke]; rr = Rte[oke]
        results["buffer_health"][key] = {
            "features": int(len(idx)), "alpha": alpha, "val_R2": float(r2_va),
            "test_R2": float(r2_score(gt, yte)), "test_R2_ci": boot(lambda ix: r2_score(gt[ix], yte[ix]), rr),
            "test_MAE_s": float(mean_absolute_error(gt, yte) * 120),
            "test_MAE_s_ci": [v * 120 for v in boot(lambda ix: mean_absolute_error(gt[ix], yte[ix]), rr)]}
        print(key, json.dumps({k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in results[k][key].items()} for k in ("stalling", "buffer_health")}), flush=True)

OUT.write_text(json.dumps(results, indent=1))
print("saved", OUT)
