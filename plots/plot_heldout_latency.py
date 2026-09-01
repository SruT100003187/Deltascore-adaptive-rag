import matplotlib.pyplot as plt
import numpy as np
from _parse_allseeds import parse_final_results_allseeds
results = parse_final_results_allseeds('../results/final/FINAL_results_allseeds.txt')
seeds = [str(r['seed']) for r in results]
p95_pct = [r['p95_pct_change'] for r in results]
mean_pct = [r['mean_pct_change'] for r in results]
x = np.arange(len(seeds))
width = 0.35
fig, ax = plt.subplots(figsize=(7, 4.3))
ax.bar(x - width / 2, p95_pct, width, label='P95 latency change', color='#4C72B0')
ax.bar(x + width / 2, mean_pct, width, label='Mean latency change', color='#DD8452')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f'Seed {s}' for s in seeds])
ax.set_ylabel('Change vs static baseline (%)')
ax.set_title('Held-out P95 and mean latency reductions\nacross the four random splits')
ax.legend()
fig.tight_layout()
fig.savefig('fig_5_6_latency_reduction.png', dpi=200)
print('saved fig_5_6_latency_reduction.png')
