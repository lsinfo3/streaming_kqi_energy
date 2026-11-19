"""Build train/validation/test feature sets for all 11,142 runs with the thesis split."""
import sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from features import run_samples, finalize

BASE = Path("yt_dataset/dataset")  # point this at your copy of the public dataset
OUT = Path(sys.argv[1])

# thesis split (master_thesis/src/helpers.py and preprocessing.build_datasets_from_config)
RNG = np.random.default_rng(1234)
NUMBER_OF_RUNS = 11143
RND_RUNS = RNG.choice(a=range(1, NUMBER_OF_RUNS), size=NUMBER_OF_RUNS - 1, replace=False)
SPLITS = {"training": RND_RUNS[0:9018], "validation": RND_RUNS[9018:10051], "test": RND_RUNS[10051:11142]}


def work(run):
    out = run_samples(BASE / f"run_{run}")
    if out is None:
        return run, None
    X, s, b = out
    Xf, keep = finalize(X)
    return run, (Xf[keep].astype(np.float32), s[keep], b[keep])


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for stage, runs in SPLITS.items():
        Xs, Ss, Bs, R, empty = [], [], [], [], []
        with Pool(7) as p:
            for i, (run, res) in enumerate(p.imap(work, runs, chunksize=8)):
                if res is None or len(res[0]) == 0:
                    empty.append(int(run)); continue
                X, s, b = res
                Xs.append(X); Ss.append(s); Bs.append(b); R.append(np.full(len(X), run, dtype=np.int32))
                if i % 500 == 0:
                    print(stage, i, len(runs), flush=True)
        np.save(OUT / f"{stage}_X.npy", np.concatenate(Xs))
        np.save(OUT / f"{stage}_stalling.npy", np.concatenate(Ss))
        np.save(OUT / f"{stage}_bh.npy", np.concatenate(Bs))
        np.save(OUT / f"{stage}_run.npy", np.concatenate(R))
        print(stage, "done", sum(len(x) for x in Xs), "samples; runs without samples:", len(empty), flush=True)
