"""Reproduce the run-level dataset split of the accompanying paper.

The split is a fixed permutation of the 11,142 playback runs of the public
YouTube dataset, drawn with NumPy's default_rng and seed 1234, cut into
9,018 training / 1,033 validation / 1,091 test runs (81/9/10 %).

Source of truth: master_thesis/src/helpers.py lines 11-14 (SEED, RND_RUNS) and
master_thesis/src/processing/preprocessing.py line 334 (the cut points).

Usage:
    python split.py                 # print the split sizes and a checksum
    python split.py --out split.csv # write run_id,stage for all 11,142 runs
"""

import argparse

import numpy as np

SEED = 1234
NUMBER_OF_RUNS = 11143  # exclusive upper bound; run IDs are 1..11142
CUTS = (9018, 10051, 11142)  # train | validation | test


def run_split():
    """Return (train, validation, test) arrays of run IDs, in pipeline order.

    The order matters: the per-run sample blocks of the stored datasets and
    prediction arrays are concatenated in exactly this order.
    """
    rng = np.random.default_rng(SEED)
    runs = rng.choice(a=range(1, NUMBER_OF_RUNS), size=NUMBER_OF_RUNS - 1, replace=False)
    train, validation, test = runs[: CUTS[0]], runs[CUTS[0] : CUTS[1]], runs[CUTS[1] : CUTS[2]]
    return train, validation, test


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", help="write run_id,stage to this CSV")
    args = ap.parse_args()

    train, validation, test = run_split()
    assert (len(train), len(validation), len(test)) == (9018, 1033, 1091)
    assert not (set(train) & set(test)) and not (set(train) & set(validation))

    print(f"train      {len(train):>6} runs")
    print(f"validation {len(validation):>6} runs")
    print(f"test       {len(test):>6} runs")
    print(f"first five test runs, in pipeline order: {list(test[:5])}")
    print(f"checksum (sum of test run IDs): {int(test.sum())}")

    if args.out:
        with open(args.out, "w") as fh:
            fh.write("run_id,stage\n")
            for stage, arr in (("train", train), ("validation", validation), ("test", test)):
                for run in arr:
                    fh.write(f"{run},{stage}\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
