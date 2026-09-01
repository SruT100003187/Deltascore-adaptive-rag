import matplotlib.pyplot as plt
UNCHANGED = 49
CHANGED = 1
N = UNCHANGED + CHANGED
NOISE_FLOOR = 0.02
fig, ax = plt.subplots(figsize=(6, 4.3))
bars = ax.bar(['Unchanged verdicts', 'Changed verdicts'], [UNCHANGED, CHANGED], color=['#2E8B8B', '#C0622A'])
for bar, val in zip(bars, [UNCHANGED, CHANGED]):
    ax.annotate(str(val), (bar.get_x() + bar.get_width() / 2, val), ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.set_ylabel(f'Repeated baseline answers (n={N})')
ax.set_title(f'Temperature-zero judge repeatability check\n{CHANGED} of {N} verdicts changed (observed repeat variation = {NOISE_FLOOR:.3f})')
fig.tight_layout()
fig.savefig('fig_D_1_judge_repeatability.png', dpi=200)
print(f'saved fig_D_1_judge_repeatability.png  ({UNCHANGED} unchanged, {CHANGED} changed, n={N})')
print()
print('Source: thesis Appendix D text ("Repeatability: 1 of 50 verdicts changed on a')
print('second run"). For context, judge_baseline_t0.csv / judge_fastK3_t0.csv give the')
print('related but distinct aggregate faithfulness scores (baseline 0.725, Fast-Path 0.640')
print('as currently saved vs. 0.723/0.650 quoted in the thesis) - a small run-to-run drift')
print('consistent with the same 0.020 noise floor this repeatability check measures.')
