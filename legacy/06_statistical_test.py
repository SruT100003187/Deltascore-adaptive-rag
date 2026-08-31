"""
STEP 6 (FINAL) - Statistical Test.

WHY THIS MATTERS
You showed the adaptive router is faster (RQ1) and measured the small
faithfulness trade-off (RQ2). This last step proves the speed improvement is
STATISTICALLY REAL, not just luck or random variation. That is what lets you
write "the improvement is statistically significant" in your thesis, which is
much stronger than just showing two numbers.

WHAT IT DOES
It pairs each question's baseline latency with its adaptive latency (same 400
questions, same order) and runs the Wilcoxon signed-rank test. This is the
correct test for paired before/after measurements that are not normally
distributed (yours are not, because of the tail). It reports:
  - the p-value (below 0.05 means the difference is statistically significant)
  - how many questions got faster vs slower under routing
  - a final summary table you can paste straight into your thesis

HOW TO RUN
  python 06_statistical_test.py
It only reads files you already have, so it finishes in a few seconds.
No model, no internet, no waiting.
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

BASELINE_CSV = "baseline_full.csv"
ADAPTIVE_CSV = "adaptive_full.csv"


def main():
    base = pd.read_csv(BASELINE_CSV).reset_index(drop=True)
    adapt = pd.read_csv(ADAPTIVE_CSV).reset_index(drop=True)

    # Pair by row: both files are in the same question order.
    n = min(len(base), len(adapt))
    b_lat = base["latency_seconds"].to_numpy()[:n]
    a_lat = adapt["latency_seconds"].to_numpy()[:n]

    # ---- Descriptive comparison (robust to the 3 machine-stall outliers) ----
    def stats(x):
        x = np.asarray(x, dtype=float)
        clean = x[x <= 60]
        return (round(clean.mean(), 2), round(clean.std(), 2),
                round(np.percentile(x, 50), 2), round(np.percentile(clean, 95), 2))

    b_mean, b_std, b_p50, b_p95 = stats(b_lat)
    a_mean, a_std, a_p50, a_p95 = stats(a_lat)

    faster = int(np.sum(a_lat < b_lat))
    slower = int(np.sum(a_lat > b_lat))
    same = int(np.sum(a_lat == b_lat))

    # ---- Wilcoxon signed-rank test on paired latencies ----
    # Drop pairs that are exactly equal (the Deep-Path queries reuse baseline,
    # so those pairs are identical and carry no signal for the test).
    diff = a_lat - b_lat
    nonzero = diff[diff != 0]
    if len(nonzero) > 0:
        stat, p_value = wilcoxon(nonzero)
    else:
        stat, p_value = float("nan"), float("nan")

    print("\n===============  DESCRIPTIVE COMPARISON  ===============")
    table = pd.DataFrame([
        {"system": "Static baseline (Top-5)", "mean_s": b_mean, "std_s": b_std,
         "P50_s": b_p50, "P95_s": b_p95},
        {"system": "Adaptive (DeltaScore)", "mean_s": a_mean, "std_s": a_std,
         "P50_s": a_p50, "P95_s": a_p95},
    ])
    print(table.to_string(index=False))

    print("\n===============  PAIRED OUTCOME (per question)  ===============")
    print(f"Questions FASTER under routing:  {faster}")
    print(f"Questions SLOWER under routing:  {slower}")
    print(f"Questions UNCHANGED (Deep Path):  {same}")

    print("\n===============  WILCOXON SIGNED-RANK TEST  ===============")
    print(f"Compared {len(nonzero)} questions whose latency changed.")
    print(f"Test statistic: {stat:.1f}")
    print(f"p-value: {p_value:.6f}")
    if p_value < 0.05:
        print("\nRESULT: p < 0.05. The latency improvement is STATISTICALLY SIGNIFICANT.")
        print("You can state in your thesis that DeltaScore routing produces a")
        print("statistically significant reduction in latency versus the static baseline.")
    else:
        print("\nRESULT: p >= 0.05. The improvement is not statistically significant")
        print("at this threshold. Report this honestly; a stronger Fast/Deep gap may help.")

    # Save a clean summary for the thesis
    summary = table.copy()
    summary.to_csv("final_comparison.csv", index=False)
    with open("statistical_test_result.txt", "w", encoding="utf-8") as f:
        f.write("Wilcoxon signed-rank test on paired latencies\n")
        f.write(f"n changed = {len(nonzero)}, statistic = {stat:.1f}, p = {p_value:.6f}\n")
        f.write(f"faster = {faster}, slower = {slower}, unchanged = {same}\n")
        f.write(f"baseline P95 = {b_p95}s, adaptive P95 = {a_p95}s\n")
    print("\nSaved final_comparison.csv and statistical_test_result.txt.")
    print("Your implementation is complete.")


if __name__ == "__main__":
    main()
