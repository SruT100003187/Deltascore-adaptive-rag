import matplotlib.pyplot as plt
from _parse_allseeds import parse_final_results_allseeds
results = parse_final_results_allseeds('../results/final/FINAL_results_allseeds.txt')
fig, ax = plt.subplots(figsize=(7, 4.6))
for r in results:
    faith_pct_change = (r['adaptive_faith'] - r['static_faith']) / r['static_faith'] * 100
    ax.scatter(r['mean_pct_change'], faith_pct_change, s=90, color='#4C72B0')
    ax.annotate(f'seed {r['seed']}', (r['mean_pct_change'], faith_pct_change), textcoords='offset points', xytext=(8, 5), fontsize=9)
ax.axhline(0, color='black', linewidth=0.6)
ax.set_xlabel('Mean latency change vs static baseline (%)')
ax.set_ylabel('Supportedness change vs static baseline (%)')
ax.set_title('Held-out latency-quality trade-off\nacross the four robustness splits')
fig.tight_layout()
fig.savefig('fig_6_1_latency_quality_tradeoff.png', dpi=200)
print('saved fig_6_1_latency_quality_tradeoff.png')
print('Note: four overlapping re-partitions of the same 400-question pool, not independent')
print('replications - the thesis explicitly cautions against reading this as a correlation test.')
