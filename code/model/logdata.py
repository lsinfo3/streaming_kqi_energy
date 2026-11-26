"""Loaders for the per-run measurement logs in data_masters_thesis/measurements.
Means after removing outliers outside 1.5 IQR (as stated in Section III-E).

Scalar helpers (capture, w1, live_step, ...) return cleaned means; the *_runs helpers return the
cleaned per-run values that the bootstrap in uncertainty.py resamples."""
import glob, os, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '../../data_masters_thesis')
ROOT = os.path.join(DATA, 'measurements')
DEV = {'raspberry_pi': 'pi', 'jetson_nano_orin': 'jn'}
DEVDIR = {'pi': 'raspberry_pi', 'jn': 'jetson_nano_orin'}
NCORES = {'pi': 4, 'jn': 6}
VIEWDIR = {'ALL': 'header_only', 'ALL>100B': 'header_only_gt_100', 'UL': 'header_only_uplink_only',
           'UL>100B': 'header_only_uplink_only_gt_100', 'FULL': 'complete_capture'}
LIVEVIEW = {'ALL': 'all', 'ALL>100B': 'gt_100', 'UL': 'uplink_only', 'UL>100B': 'uplink_only_gt_100'}


def iqr_mask(x):
    x = np.asarray(x, float); q1, q3 = np.percentile(x, [25, 75]); i = q3 - q1
    return (x >= q1 - 1.5 * i) & (x <= q3 + 1.5 * i)


def clean(x):
    x = np.asarray(x, float)
    return x[iqr_mask(x)] if len(x) >= 4 else x


# ---------------------------------------------------------------- capture (W3)
def capture_file(dev, view, S, dissection):
    v = VIEWDIR[view]; suf = '_dissection' if dissection else ''
    return f"{ROOT}/capturing/{DEVDIR[dev]}/{v}{suf}/{v}_{S}_overlaid{suf}_stats.csv"


def capture_runs(dev, view, S, dissection, col='tf_energy'):
    """Raw per-run log rows (not cleaned) of the main file (no _old/_new variants)."""
    f = capture_file(dev, view, S, dissection)
    if not os.path.exists(f): raise FileNotFoundError(f)
    return pd.read_csv(f)


def capture_power_runs(dev, view, S, dissection):
    """Per-run mean power E/T [W], IQR-cleaned on energy."""
    x = capture_runs(dev, view, S, dissection)
    k = iqr_mask(x.tf_energy) if len(x) >= 4 else np.ones(len(x), bool)
    return (x.tf_energy / x.tf_duration).to_numpy()[k]


def capture(dev, view, S, dissection):
    return clean(capture_runs(dev, view, S, dissection).tf_energy).mean()


def capture_loads(dev, view, dissection):
    v = VIEWDIR[view]; suf = '_dissection' if dissection else ''
    out = []
    for f in glob.glob(f"{ROOT}/capturing/{DEVDIR[dev]}/{v}{suf}/{v}_*_overlaid{suf}_stats.csv"):
        s = os.path.basename(f)[len(v) + 1:].split('_')[0]
        if s.isdigit(): out.append(int(s))
    return sorted(out)


def idle_power(dev):
    j = json.load(open(f"{ROOT}/idle_baseline/{DEVDIR[dev]}_idle_baseline_12h.json"))
    return j['tf_power_avg']


# ---------------------------------------------------------------- model execution (W1, W2)
def execution(mode):
    rows = []
    for f in glob.glob(f'{ROOT}/model_execution/*/{mode}/**/*.csv', recursive=True):
        p = f[len(ROOT) + len('/model_execution/'):].split('/')
        x = pd.read_csv(f); b = p[-1].replace('_stats.csv', '')
        x['dev'] = DEV[p[0]]; x['sub'] = p[2] if mode == 'live' else ''
        x['variant'] = 'pf' if b.endswith('performance') else 'ea'
        x['kqi'] = 'stalling' if b.startswith('stalling') else 'buffer_health'
        x['fam'] = p[-2]; x['view'] = p[-3]
        rows.append(x)
    return pd.concat(rows, ignore_index=True)


def w1(mode='offline'):
    x = execution(mode); g = ['dev', 'kqi', 'view', 'fam', 'variant']
    return x.groupby(g).agg(E=('tf_energy', lambda s: clean(s).mean()), T=('tf_duration', lambda s: clean(s).mean())).reset_index()


def _exec_file(dev, mode, kqi, view, fam, variant, sub=None):
    name = f"{kqi}_{view}_{fam}_{'performance' if variant == 'pf' else 'efficiency'}_stats.csv"
    mid = f"{mode}/{sub}" if sub else mode
    return f"{ROOT}/model_execution/{DEVDIR[dev]}/{mid}/{kqi}/{view}/{fam}/{name}"


def w1_runs(dev, kqi, view, fam, variant, batch=False):
    """Per-run (E [J], T [s], cores busy) of the W1 job (100,000 predictions), IQR-cleaned on energy."""
    x = pd.read_csv(_exec_file(dev, 'offline_fixed_batch' if batch else 'offline', kqi, view, fam, variant))
    k = iqr_mask(x.tf_energy)
    return (x.tf_energy.to_numpy()[k], x.tf_duration.to_numpy()[k],
            (x.ul_cpu_global_avg.to_numpy()[k] / 100 * NCORES[dev]))


def live_runs(dev, kqi, view, fam, variant, sub='model_execution'):
    """Per-run log rows of a W2 configuration (sub = model_execution or feature_extraction reference)."""
    return pd.read_csv(_exec_file(dev, 'live', kqi, view, fam, variant, sub))


def live_power_runs(dev, kqi, view, fam, variant, sub='model_execution'):
    x = live_runs(dev, kqi, view, fam, variant, sub); k = iqr_mask(x.tf_energy)
    return (x.tf_energy / x.tf_duration).to_numpy()[k]


def live_step(dev, kqi, view, fam, variant):
    x = execution('live'); k = (x.dev == dev) & (x.kqi == kqi) & (x.view == view) & (x.fam == fam) & (x.variant == variant)
    me = x[k & (x['sub'] == 'model_execution')]; ref = x[k & (x['sub'] == 'feature_extraction')]
    return (clean(me.tf_energy).mean() - clean(ref.tf_energy).mean()) / me.tf_duration.median()


# ---------------------------------------------------------------- feature extraction
def fx_offline_runs(dev, view, window, interval=None):
    """Offline feature extraction: 100 single-stream files processed back to back on one core.
    interval: None for the per-window logs, or 0.5/1/2 for the execution-interval variants (1 s window, ALL)."""
    v = VIEWDIR[view]; d = f"{ROOT}/feature_extraction/{DEVDIR[dev]}/offline"
    if interval is None:
        f = f"{d}/{v}/{v}_{window}s_feature_extraction_stats.csv"
    else:
        f = f"{d}/{v}_execution_interval/{v}_1s_{interval:g}s_execution_interval_feature_extraction_stats.csv"
    x = pd.read_csv(f); k = iqr_mask(x.tf_energy)
    return x[k].reset_index(drop=True)


def fx_live_runs(dev, view, window):
    v = VIEWDIR[view]
    x = pd.read_csv(f"{ROOT}/feature_extraction/{DEVDIR[dev]}/live/{v}/{v}_{window}s_feature_extraction_stats.csv")
    return x[iqr_mask(x.tf_energy)].reset_index(drop=True)


# ---------------------------------------------------------------- replay traces
PCAP_SCEN = {'ALL': 'header_only', 'ALL>100B': 'header_only_gt_100', 'UL': 'header_only_uplink_only',
             'UL>100B': 'header_only_uplink_only_gt_100', 'FULL': 'complete_capture'}


def pcap_rates(overlay=100):
    """Packets per second and stream for each of the 30 replay seeds: DataFrame seeds x views."""
    rows = {}
    for f in sorted(glob.glob(f"{DATA}/general_data/pcap_stats/stats_s*_r{overlay}.csv")):
        s = os.path.basename(f).split('_')[1]
        x = pd.read_csv(f).set_index('scenario')
        rows[s] = {v: x.loc[sc, 'pkts/s'] / overlay for v, sc in PCAP_SCEN.items()}
    return pd.DataFrame(rows).T


def pcap_packets(overlay=100):
    rows = {}
    for f in sorted(glob.glob(f"{DATA}/general_data/pcap_stats/stats_s*_r{overlay}.csv")):
        s = os.path.basename(f).split('_')[1]
        x = pd.read_csv(f).set_index('scenario')
        rows[s] = {v: x.loc[sc, 'packets'] for v, sc in PCAP_SCEN.items()}
    return pd.DataFrame(rows).T


def processing_energy(dev, view, S=100, kind='static'):
    """Deferred processing of a stored five-minute capture (whole-device energy [J], duration [s]), cleaned means."""
    v = VIEWDIR[view]
    x = pd.read_csv(f"{ROOT}/processing/{DEVDIR[dev]}/{v}_{kind}_processing/{v}_{S}_{kind}_processing_stats.csv")
    k = iqr_mask(x.tf_energy)
    return x.tf_energy[k].mean(), x.tf_duration[k].mean()
