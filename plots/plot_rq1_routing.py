import matplotlib.pyplot as plt
import numpy as np
from _parse_allseeds import parse_final_results_allseeds
results = parse_final_results_allseeds('../results/final/FINAL_results_allseeds.txt')
seeds = [str(r['seed']) for r in results]
fast_rate = [r['deep_needed_rate_fast_routed'] for r in results]
deep_rate = [r['deep_needed_rate_deep_routed'] for r in results]
x = np.arange(len(seeds))
width = 0.35
fig, ax = plt.subplots(figsize=(7, 4.3))
ax.bar(x - width / 2, fast_rate, width, label='Among Fast-routed', color='#55A868')
ax.bar(x + width / 2, deep_rate, width, label='Among Deep-routed', color='#C44E52')
ax.set_xticks(x)
ax.set_xticklabels([f'Seed {s}' for s in seeds])
ax.set_ylabel('Deep-needed rate (%)')
ax.set_title('Deep-needed rates among Fast-routed and Deep-routed\nquestions on each held-out split')
ax.legend()
fig.tight_layout()
fig.savefig('fig_5_5_deep_needed_rates.png', dpi=200)
print('saved fig_5_5_deep_needed_rates.png')
