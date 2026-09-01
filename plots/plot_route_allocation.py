import matplotlib.pyplot as plt
import numpy as np
from _parse_allseeds import parse_final_results_allseeds
results = parse_final_results_allseeds('../results/final/FINAL_results_allseeds.txt')
seeds = [str(r['seed']) for r in results]
fast_n = [r['fast_n'] for r in results]
deep_n = [r['deep_n'] for r in results]
x = np.arange(len(seeds))
fig, ax = plt.subplots(figsize=(7, 4.3))
ax.bar(x, fast_n, label='Routed Fast', color='#55A868')
ax.bar(x, deep_n, bottom=fast_n, label='Routed Deep', color='#C44E52')
ax.set_xticks(x)
ax.set_xticklabels([f'Seed {s}' for s in seeds])
ax.set_ylabel('Number of queries (of 200 held-out)')
ax.set_title('Held-out Fast/Deep route allocation after freezing\nthe development-selected threshold')
ax.legend()
fig.tight_layout()
fig.savefig('fig_5_8_route_allocation.png', dpi=200)
print('saved fig_5_8_route_allocation.png')
