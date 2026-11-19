"""Re-implementation of the thesis sample construction for the same-split baseline.

Mirrors master_thesis/src: processing/preprocessing.py (convert_to_matrix_format,
create_link_table, worker, build_datasets_from_config) and
feature_extraction_offline.compute_features, for the ALL view with windows
1, 3, 5, 10, 20, 30 s and a 1 s execution interval.

Per window, 48 features in thesis order:
  for direction in (ul, dl):            # i = 0 uplink, i = 1 downlink
    for subset in (gt_100, all):        # j = 0 size > 100 B, j = 1 all
      iat  [mean, median, min, max, std]
      size [mean, median, min, max, std]
      packet_count, total_bytes
Size is the transport payload length (udpLen / tcpLen), as in training.
"""
import numpy as np
import pandas as pd

CLIENT_IPS = {"10.10.0.140", "10.10.0.10", "10.10.0.15"}
WINDOWS_S = (1, 3, 5, 10, 20, 30)
INTERVAL_NS = 10**9
AGGS = ("mean", "median", "min", "max", "std")


def feature_names():
    names = []
    for w in WINDOWS_S:
        for d in ("ul", "dl"):
            for suffix in ("_gt_100", ""):
                names += [f"{w}s_{d}_{a}_iat{suffix}" for a in AGGS]
                names += [f"{w}s_{d}_{a}_size{suffix}" for a in AGGS]
                names += [f"{w}s_{d}_packet_count{suffix}", f"{w}s_{d}_total_bytes{suffix}"]
    return names


def _agg(x):
    n = len(x)
    if n == 0:
        return [np.nan] * 5
    std = np.std(x, ddof=1) if n > 1 else np.nan  # pandas default ddof=1
    return [x.mean(), np.median(x), x.min(), x.max(), std]


def load_run(run_dir):
    tr = pd.read_csv(run_dir / "video_traffic.csv",
                     usecols=["timestamp", "ipSrc", "ipDst", "tcpLen", "udpLen", "payloadProtocolNumber"])
    up = tr["ipSrc"].isin(CLIENT_IPS)
    down = tr["ipDst"].isin(CLIENT_IPS)
    size = np.where(tr["payloadProtocolNumber"] == 17, tr["udpLen"],
                    np.where(tr["payloadProtocolNumber"] == 6, tr["tcpLen"], np.nan))
    df = pd.DataFrame({"ts": tr["timestamp"], "up": np.where(up, 1.0, np.where(down, 0.0, np.nan)), "size": size})
    df = df.dropna()
    df["ts"] = np.round(df["ts"].to_numpy() * 1e9).astype(np.int64)
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    app = pd.read_csv(run_dir / "application_data.csv", usecols=["timestamp", "bh", "phase"])
    app["ts"] = pd.to_datetime(app["timestamp"]).astype("int64")
    app = app.sort_values("ts", kind="stable").reset_index(drop=True)
    return df, app


def run_samples(run_dir):
    """Return (features [n, 288] float64 with NaN, stalling [n], bh [n]) before ffill."""
    df, app = load_run(run_dir)
    if len(df) == 0 or len(app) == 0:
        return None
    ts = df["ts"].to_numpy()
    up = df["up"].to_numpy().astype(bool)
    size = df["size"].to_numpy().astype(np.float64)
    # link every packet to the nearest application row (merge_asof, nearest)
    link = pd.merge_asof(pd.DataFrame({"ts": ts}), app[["ts"]].assign(row=np.arange(len(app))),
                         on="ts", direction="nearest")["row"].to_numpy()
    subsets = []
    for d in (True, False):
        m_dir = up == d
        for gt in (True, False):
            m = m_dir & (size > 100) if gt else m_dir
            subsets.append((ts[m], size[m]))
    feats, rows = [], []
    t, last = ts[0], ts[-1]
    while t < last:
        t += INTERVAL_NS
        lo30 = np.searchsorted(ts, t - WINDOWS_S[-1] * 10**9, "left")
        hi30 = np.searchsorted(ts, t, "right")
        if hi30 - lo30 == 0:            # thesis: skip when the last (30 s) window is empty
            continue
        vec = []
        for w in WINDOWS_S:
            lo = np.searchsorted(ts, t - w * 10**9, "left")
            hi = np.searchsorted(ts, t, "right")
            if hi - lo == 0:            # compute_features returns None -> NA block
                vec += [np.nan] * 48
                continue
            for sts, ssz in subsets:
                a = np.searchsorted(sts, t - w * 10**9, "left")
                b = np.searchsorted(sts, t, "right")
                wt, wsz = sts[a:b], ssz[a:b]
                iat = np.diff(wt).astype(np.float64) if len(wt) > 1 else np.empty(0)
                vec += _agg(iat) if len(wt) != 1 else [np.nan] * 5
                vec += _agg(wsz)
                vec += [float(len(wt)), float(wsz.sum())]
        feats.append(vec)
        rows.append(link[hi30 - 1])     # id of the last packet in the 30 s window
    if not feats:
        return None
    X = np.asarray(feats, dtype=np.float64)
    r = np.asarray(rows)
    stalling = (app["phase"].to_numpy()[r] == "stalling").astype(np.float32)
    bh = app["bh"].to_numpy().astype(np.float64)[r]
    bh = np.where(bh < 0, np.nan, np.clip(bh, 0, 120_000) / 120_000)
    return X, stalling, bh.astype(np.float32)


def finalize(X):
    """ffill per run, then the thesis drops rows that still contain NaN and clips at 0."""
    Xf = pd.DataFrame(X).ffill().to_numpy()
    keep = ~np.isnan(Xf).any(axis=1)
    return np.clip(Xf, 0, None), keep
