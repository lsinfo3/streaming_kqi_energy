import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from features import run_samples, finalize
base = Path("yt_dataset/dataset")  # point this at your copy of the public dataset
for r in (1, 2, 500, 5000):
    t0 = time.perf_counter(); out = run_samples(base / f"run_{r}")
    dt = time.perf_counter() - t0
    if out is None: print(r, "none", dt); continue
    X, s, b = out; Xf, keep = finalize(X)
    print(f"run {r}: {X.shape} kept {keep.sum()} stall {s[keep].mean():.3f} bhNaN {int((b!=b).sum())} {dt:.2f}s")
