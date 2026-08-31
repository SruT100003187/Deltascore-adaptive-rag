import os
import numpy as np
import pandas as pd
try:
    from scipy.stats import binomtest

    def _binom(k, n):
        return binomtest(k, n, 0.5, alternative='two-sided').pvalue
except ImportError:
    from scipy.stats import binom_test

    def _binom(k, n):
        return binom_test(k, n, 0.5, alternative='two-sided')
BASE_CSV = 'baseline_full.csv'
KS = [2, 3]

def need(path):
    if not os.path.exists(path):
        raise SystemExit(f'Missing {path}. Run sweep_two_graphs.py with LIMIT = None first.')
    return path
base = pd.read_csv(need(BASE_CSV))
jb = pd.read_csv(need('judge_baseline.csv'))
baseline_v = dict(zip(jb['question'], jb['faithful'].astype(int)))
fast_v = {K: dict(zip(pd.read_csv(need(f'judge_fastK{K}.csv'))['question'], pd.read_csv(need(f'judge_fastK{K}.csv'))['faithful'].astype(int))) for K in KS}
d = base['delta_score'].to_numpy()
mean, sd = (float(d.mean()), float(d.std(ddof=1)))
THRESHOLDS = [('mean+2sd', mean + 2 * sd), ('mean+1sd', mean + sd), ('mean', mean), ('mean-1sd', mean - sd), ('mean-2sd', mean - 2 * sd)]
questions = list(base['question'])
base_rate = np.mean([baseline_v[q] for q in questions])
print(f'Baseline faithfulness (this judge run): {base_rate:.3f}  on {len(questions)} queries\n')
rows = []
for K in KS:
    for label, tau in THRESHOLDS:
        b = c = 0
        adapt = []
        for _, r in base.iterrows():
            q = r['question']
            bv = baseline_v[q]
            av = fast_v[K][q] if r['delta_score'] >= tau else bv
            adapt.append(av)
            if bv == 1 and av == 0:
                b += 1
            elif bv == 0 and av == 1:
                c += 1
        n_disc = b + c
        p = _binom(min(b, c), n_disc) if n_disc > 0 else 1.0
        rate = float(np.mean(adapt))
        rows.append({'K': K, 'threshold': label, 'tau': round(tau, 4), 'fast_n': int((d >= tau).sum()), 'faith': round(rate, 3), 'diff': round(rate - base_rate, 3), 'lost': b, 'gained': c, 'discordant': n_disc, 'p_value': round(p, 4), 'verdict': 'NOT significant' if p >= 0.05 else 'SIGNIFICANT drop' if b > c else 'SIGNIFICANT gain'})
res = pd.DataFrame(rows)
res.to_csv('significance_results.csv', index=False)
print('=================  McNEMAR TEST vs BASELINE  =================')
print(res.to_string(index=False))
print('==============================================================')
print('\nHow to read this:')
print("  'lost'   = queries the baseline got right and the adaptive system did not")
print("  'gained' = queries the adaptive system got right and the baseline did not")
print('  p >= 0.05 means the faithfulness difference is NOT statistically significant.\n')
ok = res[res['p_value'] >= 0.05]
if len(ok):
    print('Operating points with NO significant faithfulness drop (success criterion met):')
    for _, r in ok.iterrows():
        print(f'   K={r['K']}  {r['threshold']:9}  faith {r['faith']}  (diff {r['diff']:+.3f})  p={r['p_value']}')
else:
    print('Every operating point shows a significant drop. Report that honestly.')
print('\nSaved significance_results.csv')
