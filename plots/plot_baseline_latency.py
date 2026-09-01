import pandas as pd
import matplotlib.pyplot as plt
OUTLIER_SECONDS = 60
df = pd.read_csv('../results/final/baseline_full.csv')
clean = df[df['latency_seconds'] <= OUTLIER_SECONDS]
print(f'total queries: {len(df)}')
print(f'excluded as outliers (> {OUTLIER_SECONDS}s): {len(df) - len(clean)}')
print(f'cleaned n: {len(clean)}')
fig, ax = plt.subplots(figsize=(7, 4.3))
ax.hist(clean['latency_seconds'], bins=30, color='#4C72B0', edgecolor='white')
ax.set_xlabel('Latency (seconds)')
ax.set_ylabel('Number of queries')
ax.set_title(f'Static Top-5 baseline latency distribution (cleaned, n={len(clean)})')
fig.tight_layout()
fig.savefig('fig_5_1_baseline_latency.png', dpi=200)
print('saved fig_5_1_baseline_latency.png')
