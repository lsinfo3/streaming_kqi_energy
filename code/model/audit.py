"""Recompute the live-offset correction (Section III-F) and its downstream projections.

Run from any directory with Python, NumPy, pandas, SciPy and Matplotlib.
The report is written beside this script; original thesis data remain unchanged.
"""
from pathlib import Path
import json
import numpy as np
import uncertainty as u


def main():
    c = u._setup()
    em, point = u.em, c['point']
    draws = u.sample_parameters()
    screening = em.means(c['inp'], 10000, np.random.default_rng(2026))
    report = {'draws': 10000, 'projection_seed': u.SEED, 'offset_screening_seed': 2026,
              'offsets': {}, 'crossovers': {}, 'nominal_energy': {}, 'batch_savings_percent': {}}
    for d in em.DEVS:
        for cls in ('light', 'heavy'):
            g = c['point_means'][d]
            measured, rate = g['w2'][cls]['me']
            delta = measured - g['w2'][cls]['ref']
            cm = point[d]['inf'][cls]['cm']
            residual = float(em.live_residual(g, cls))
            np.testing.assert_allclose(residual + rate * cm, delta, atol=1e-12)
            used = float(point[d]['inf'][cls]['dP'])
            if used:
                # At the observed W2 rate, the corrected decomposition recovers the increment.
                np.testing.assert_allclose(used + rate * cm, delta, atol=1e-12)
            report['offsets'][f'{d}:{cls}'] = dict(observed_W=float(delta), actual_rate_Hz=float(rate),
                residual_W=residual, residual_CI95_W=np.percentile(em.live_residual(screening[d], cls),[2.5,97.5]).tolist(), used_W=used)
        assert point[d]['inf']['heavy']['dP'] == point[d]['inf']['heavy, batch 32']['dP']
        # All structural scenarios agree at one stream for unbatched inference.
        for cls in ('light', 'heavy'):
            a=em.inference_power(point[d],cls,1,'A')
            for scenario in ('B','C'):
                np.testing.assert_allclose(a,em.inference_power(point[d],cls,1,scenario))
    for view in ('ALL','UL'):
        for cls in em.CLASSES:
            for scenario in u.DPS:
                for features in u.CXS:
                    key=f'{view}:{cls}:{scenario}:{features}'
                    x=u.crossover_draws(draws,cls,view,scenario,features)
                    report['crossovers'][key]=np.percentile(x,[5,50,95]).tolist()
    for S in (20,30,60):
        for d in em.DEVS:
            for cls in em.CLASSES:
                for features in u.CXS:
                    report['nominal_energy'][f'{d}:{cls}:{S}:{features}']=float(em.e_stream_hour(point,d,S,cls,cx=features))
        for scenario in u.DPS:
            for features in u.CXS:
                saving=100*(1-u.e_draws(draws,'pi',S,'heavy, batch 32',dp=scenario,cx=features)/u.e_draws(draws,'pi',S,'heavy',dp=scenario,cx=features))
                report['batch_savings_percent'][f'{S}:{scenario}:{features}']=np.percentile(saving,[5,50,95]).tolist()
    report['oat_heavy']=u.oat_ranking()
    report['checks']='Offset accounting, shared batch residual and single-stream scenario consistency passed.'
    path=Path(__file__).with_name('audit_results.json')
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(report['checks'])
    print(path)

if __name__=='__main__':
    main()
