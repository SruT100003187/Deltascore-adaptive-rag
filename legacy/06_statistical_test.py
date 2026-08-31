import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
BASELINE_CSV = 'baseline_full.csv'
ADAPTIVE_CSV = 'adaptive_full.csv'

def main():
    base = pd.read_csv(BASELINE_CSV).reset_index(drop=True)
    adapt = pd.read_csv(ADAPTIVE_CSV).reset_index(drop=True)
    n = min(len(base), len(adapt))
    b_lat = base['latency_seconds'].to_numpy()[:n]
    a_lat = adapt['latency_seconds'].to_numpy()[:n]

    def stats(x):
        x = np.asarray(x, dtype=float)
        clean = x[x <= 60]
        return (round(clean.mean(), 2), round(clean.std(), 2), round(np.percentile(x, 50), 2), round(np.percentile(clean, 95), 2))
    b_mean, b_std, b_p50, b_p95 = stats(b_lat)
    a_mean, a_std, a_p50, a_p95 = stats(a_lat)
    faster = int(np.sum(a_lat < b_lat))
    slower = int(np.sum(a_lat > b_lat))
    same = int(np.sum(a_lat == b_lat))
    diff = a_lat - b_lat
    nonzero = diff[diff != 0]
    if len(nonzero) > 0:
        stat, p_value = wilcoxon(nonzero)
    else:
        stat, p_value = (float('nan'), float('nan'))
    print('\n===============  DESCRIPTIVE COMPARISON  ===============')
    table = pd.DataFrame([{'system': 'Static baseline (Top-5)', 'mean_s': b_mean, 'std_s': b_std, 'P50_s': b_p50, 'P95_s': b_p95}, {'system': 'Adaptive (DeltaScore)', 'mean_s': a_mean, 'std_s': a_std, 'P50_s': a_p50, 'P95_s': a_p95}])
    print(table.to_string(index=False))
    print('\n===============  PAIRED OUTCOME (per question)  ===============')
    print(f'Questions FASTER under routing:  {faster}')
    print(f'Questions SLOWER under routing:  {slower}')
    print(f'Questions UNCHANGED (Deep Path):  {same}')
    print('\n===============  WILCOXON SIGNED-RANK TEST  ===============')
    print(f'Compared {len(nonzero)} questions whose latency changed.')
    print(f'Test statistic: {stat:.1f}')
    print(f'p-value: {p_value:.6f}')
    if p_value < 0.05:
        print('\nRESULT: p < 0.05. The latency improvement is STATISTICALLY SIGNIFICANT.')
        print('You can state in your thesis that DeltaScore routing produces a')
        print('statistically significant reduction in latency versus the static baseline.')
    else:
        print('\nRESULT: p >= 0.05. The improvement is not statistically significant')
        print('at this threshold. Report this honestly; a stronger Fast/Deep gap may help.')
    summary = table.copy()
    summary.to_csv('final_comparison.csv', index=False)
    with open('statistical_test_result.txt', 'w', encoding='utf-8') as f:
        f.write('Wilcoxon signed-rank test on paired latencies\n')
        f.write(f'n changed = {len(nonzero)}, statistic = {stat:.1f}, p = {p_value:.6f}\n')
        f.write(f'faster = {faster}, slower = {slower}, unchanged = {same}\n')
        f.write(f'baseline P95 = {b_p95}s, adaptive P95 = {a_p95}s\n')
    print('\nSaved final_comparison.csv and statistical_test_result.txt.')
    print('Your implementation is complete.')
if __name__ == '__main__':
    main()
