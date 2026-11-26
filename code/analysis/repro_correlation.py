"""Reproduce the parameter-count and cross-device rank correlations (Section IV-C).

Spearman correlations between parameter count and unbatched W1 energy, within and across model
families per device, and between the two devices' W1 energy rankings, over all 96 configurations.

Usage:  python repro_correlation.py [../../data]
"""
import sys
from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "data"
DEVS = {"raspberry_pi": "Raspberry Pi", "jetson_nano_orin": "Jetson"}


def main():
    w = pd.read_csv(DATA / "derived" / "repro_w1.csv")
    w = w[w["mode"] == "offline"]
    print("== Section IV-C: Spearman correlation, parameter count vs. W1 energy ==")
    for dev, name in DEVS.items():
        g = w[w.dev == dev]
        within = {f: spearmanr(x.n_parameters, x.E).statistic for f, x in g.groupby("fam")}
        print(f"  {name:12s} across families rho = {spearmanr(g.n_parameters, g.E).statistic:.2f} (n={len(g)})")
        print(f"  {'':12s} within: " + ", ".join(f"{f} {r:.2f}" for f, r in within.items()))
    p = w.pivot_table(index=["kqi", "view", "fam", "sel"], columns="dev", values="E")
    rho = spearmanr(p.raspberry_pi, p.jetson_nano_orin).statistic
    print(f"\n== Section IV-C: W1 energy ranking, Raspberry Pi vs. Jetson: rho = {rho:.2f} (n={len(p)}) ==")


if __name__ == "__main__":
    main()
