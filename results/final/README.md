# `results/final/`: what backs the thesis's reported numbers

## Confirmatory four-seed evaluation (the headline numbers in the abstract/Chapter 5)

`FINAL_results_allseeds.txt` was produced by running `src/finalize_experiment_v3.py` four times (seeds 42, 7, 123, 2024), reading:
- `baseline_full.csv`: static Top-5 baseline, latency plus DeltaScore per query
- `fast_cache_K3.csv`: Fast-Path (K=3) answers plus latency per query
- `judge_baseline.csv`: supportedness verdicts for the baseline answers
- `judge_fastK3.csv`: supportedness verdicts for the Fast-Path answers

**This has been independently re-run and verified**: re-executing `finalize_experiment_v3.py` unmodified against these exact four CSVs reproduces every reported figure exactly. P95 change (-7.6%, -12.5%, -11.7%, -12.3%), mean latency change (-19.5%, -24.9%, -24.2%, -21.1%), and McNemar p-values (0.8746, 0.0789, 0.1263, 0.6587) across the four seeds, matching `FINAL_results_allseeds.txt` and the abstract's stated ranges.

`final_results.csv` / `final_dev_sweep.csv` are overwritten on each run, so as saved they only reflect the last run (seed 2024), also saved separately as `FINAL_seed2024_results.csv`.

## About `judge_baseline_t0.csv` / `judge_fastK3_t0.csv`

These were produced by `src/rejudge_temp0.py`: a separate deterministic repeatability check, re-scoring the same answers with the judge fixed at temperature 0, to measure how much the (originally non-deterministic) judge could disagree with itself on a repeat pass. This produced the noise-floor figure reported in Appendix D. They were not fed back into a re-run of `finalize_experiment_v3.py`, and the confirmatory headline numbers are not derived from them.

The current aggregate scores in these files are baseline 0.7225 and fastK3 0.6500, matching the thesis Appendix D figures of 0.723 and 0.650. If you want the headline numbers to instead be based on the deterministic t0 judge, `finalize_experiment_v3.py` would need to be re-run pointing at `judge_baseline_t0.csv` / `judge_fastK3_t0.csv` instead. That would very likely shift the exact percentages slightly (the noise floor between the two judge passes was measured at ~0.02), so it's worth being clear in a defense about which pass the reported numbers come from.

## `significance_results.csv` (calibration-stage, K=2 vs K=3 threshold sweep)

From an earlier stage of the DeltaScore threshold calibration (Section 4.3.3 / RQ3), using a now-overwritten judge pass. Attempting to re-run `significance_test.py` against the *current* `judge_baseline.csv`/`judge_fastK2.csv`/`judge_fastK3.csv` on disk produces different numbers, because those verdict files were regenerated later by a fresh, non-deterministic LLM judge call. This is expected, not a bug, and is exactly the instability that motivated moving to the temperature-0 judge. `significance_results.csv` stands as a historical record of that calibration step; it isn't independently re-derivable from what's currently saved.
