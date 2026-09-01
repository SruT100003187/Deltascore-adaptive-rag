import matplotlib.pyplot as plt
import numpy as np
from _parse_allseeds import parse_final_results_allseeds
results = parse_final_results_allseeds('../results/final/FINAL_results_allseeds.txt')
seeds = [str(r['seed']) for r in results]
static_faith = [r['static_faith'] for r in results]
adaptive_faith = [r['adaptive_faith'] for r in results]
x = np.arange(len(seeds))
width = 0.35
fig, ax = plt.subplots(figsize=(7, 4.3))
ax.bar(x - width / 2, static_faith, width, label='Static Top-5', color='#4C72B0')
ax.bar(x + width / 2, adaptive_faith, width, label='Adaptive (K=3)', color='#55A868')
ax.set_xticks(x)
ax.set_xticklabels([f'Seed {s}' for s in seeds])
ax.set_ylabel('Supportedness')
ax.set_ylim(0, 1)
ax.set_title('Static and adaptive supportedness\non the four held-out splits')
ax.legend()
fig.tight_layout()
fig.savefig('fig_5_7_supportedness_comparison.png', dpi=200)
print('saved fig_5_7_supportedness_comparison.png')
