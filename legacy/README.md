# Legacy / superseded material

This folder is kept for transparency and an audit trail of how the final pipeline in `src/` was arrived at. **None of this backs the numbers reported in the thesis.** If you're trying to reproduce the thesis results, start from the top-level `README.md` instead.

## Superseded scripts

| File | Why it's here, not in `src/` |
|---|---|
| `03_evaluate_baseline.py` | Early evaluation script using RAGAS + an OpenAI-key-gated faithfulness metric. Superseded once the study moved to a fully local judge (`rejudge_temp0.py`), per thesis Section 4.3.4. |
| `05_faithfulness.py` | First implementation of the local judge — non-deterministic (no fixed temperature/seed). Superseded by `rejudge_temp0.py`, which fixes temperature to 0 and seed to 42 after this version's repeat-run instability was discovered. |
| `06_statistical_test.py` | Early significance test using a Wilcoxon signed-rank test on latency only. Superseded by `significance_test.py`, which runs McNemar's exact test on paired binary supportedness outcomes (see thesis Section 4.3.5). |
| `finalize_experiment.py`, `finalize_experiment_v2.py` | Earlier iterations of the four-seed confirmatory evaluation logic. Superseded by `finalize_experiment_v3.py`, the version actually cited in Table 4.1 and used for the reported results. |
| `fix_sample_size.py`, `set_full_run.py` | One-off dev utilities that toggled a setting inside `02_run_baseline.py` between a 20-query smoke test and the full 400-query run. Not part of the analysis pipeline itself. |
| `results_chapter.tex` | An earlier LaTeX draft of the results chapter, from before the thesis was finalised in Word. Kept for reference only. |

## `exploratory_csvs/`

Intermediate and smoke-test outputs from development (e.g. `*_smoke8.csv`, `*_noisy.csv`, `baseline_sample.csv`, `judge_baseline_noisy.csv`). These are process artefacts, not the curated evidence set — that lives in `results/final/` and `results/supplementary_3b/`.

**Correction:** an earlier pass at organising this repo mistakenly placed `judge_baseline.csv` and `judge_fastK3.csv` in this folder, assuming they'd been superseded by the temperature-0 rejudge (`judge_baseline_t0.csv` / `judge_fastK3_t0.csv`). A file-timestamp check showed the opposite: `judge_baseline.csv`/`judge_fastK3.csv` are dated 9 Aug, the same run that produced `FINAL_results_allseeds.txt`, while the `_t0` files are dated 19–20 Aug — created *after* the confirmatory numbers were already final, as a separate repeatability check (see `results/final/README.md`). They've been moved back to `results/final/` where they belong.
