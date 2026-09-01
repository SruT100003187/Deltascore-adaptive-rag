import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from _parse_allseeds import parse_final_results_allseeds
results_1b = parse_final_results_allseeds('../results/final/FINAL_results_allseeds.txt')
seeds = [42, 7, 123, 2024]
mean_pct_1b = {r['seed']: r['mean_pct_change'] for r in results_1b}
mean_pct_3b = {}
for seed in seeds:
    df = pd.read_csv(f'../results/supplementary_3b/final_results_3b_seed{seed}.csv')
    base = df[df['set'] == 'heldout_baseline']['mean'].iloc[0]
    adapt = df[df['set'] == 'heldout_adaptive']['mean'].iloc[0]
    mean_pct_3b[seed] = (adapt - base) / base * 100
x = np.arange(len(seeds))
width = 0.35
fig, ax = plt.subplots(figsize=(7, 4.3))
ax.bar(x - width / 2, [mean_pct_1b[s] for s in seeds], width, label='1B generator', color='#4C72B0')
ax.bar(x + width / 2, [mean_pct_3b[s] for s in seeds], width, label='3B generator', color='#8172B2')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f'Seed {s}' for s in seeds])
ax.set_ylabel('Mean latency change vs static baseline (%)')
ax.set_title('Held-out mean latency reduction,\ncompared between the 1B and 3B generators')
ax.legend()
fig.tight_layout()
fig.savefig('fig_5_9_generator_comparison.png', dpi=200)
print('saved fig_5_9_generator_comparison.png')
for s in seeds:
    print(f'seed {s}: 1B {mean_pct_1b[s]:.1f}%   3B {mean_pct_3b[s]:.1f}%')
