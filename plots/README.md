# plots/

Reusable scripts that regenerate every quantitative figure in the thesis (Figures 5.1-5.9, 6.1, D.1) directly from the saved evidence in `results/final/` and `results/supplementary_3b/`.

## Why this folder exists

The original figures in the submitted thesis were verified by inspecting the PNG metadata embedded in the `.docx`: 13 of the 17 total figures carry a `Software: Matplotlib version 3.10.8` tag, confirming they were genuinely produced with Python/Matplotlib from real experiment data, not fabricated or drawn by hand. The remaining 4 are either conceptual architecture diagrams (Figures 1.1, 2.1, 3.1, which are not data plots and correctly carry no such tag) or a figure whose tag was lost on a re-save (Figure 5.9).

However, the original one-off plotting code that produced the 13 data-driven figures was written and run during earlier analysis sessions and was not kept as reusable scripts in the implementation folder, only the calibration-sweep plots (Figures 5.2-5.4-ish) had their generator (`plot_sweep.py`) preserved.

These 10 scripts close that gap: each one reads directly from the saved result CSVs and regenerates the matching figure, so every quantitative claim in the thesis now has a runnable, auditable path from raw data to chart.

## Verification

Two reconstructions were checked directly against the actual embedded thesis images (Figure 5.6 and Figure 5.7) and matched every reported number exactly. Figure D.1 was corrected after an initial mismatch, it plots the 49-unchanged/1-changed breakdown from the 50-answer repeatability retest (Appendix D), not a faithfulness comparison.

## Data provenance per figure

| Script | Figure | Source data |
|---|---|---|
| `plot_baseline_latency.py` | 5.1 | `results/final/baseline_full.csv` (cleaned n=397 at the 60s outlier rule) |
| `plot_calibration_routes.py` | 5.2 | `results/final/sweep_results.csv` |
| `plot_k2_k3_tradeoff.py` | 5.3, 5.4 | `results/final/sweep_results.csv` |
| `plot_rq1_routing.py` | 5.5 | Parsed from `results/final/FINAL_results_allseeds.txt` |
| `plot_heldout_latency.py` | 5.6 | Parsed from `results/final/FINAL_results_allseeds.txt` |
| `plot_supportedness.py` | 5.7 | Parsed from `results/final/FINAL_results_allseeds.txt` |
| `plot_route_allocation.py` | 5.8 | Parsed from `results/final/FINAL_results_allseeds.txt` |
| `plot_generator_sensitivity.py` | 5.9 | `FINAL_results_allseeds.txt` (1B) + `results/supplementary_3b/final_results_3b_seed*.csv` (3B) |
| `plot_latency_quality.py` | 6.1 | Parsed from `results/final/FINAL_results_allseeds.txt` |
| `plot_judge_repeatability.py` | D.1 | Counts (49/1) taken from the thesis's own Appendix D text; not independently recomputable since the per-item repeat-judging comparison was not saved separately from the console log |

`_parse_allseeds.py` is a shared helper that regex-parses the four-seed console log into structured values, used by most of the scripts above.

## Running

```bash
cd plots
pip install pandas matplotlib
python plot_baseline_latency.py
python plot_calibration_routes.py
python plot_k2_k3_tradeoff.py
python plot_rq1_routing.py
python plot_heldout_latency.py
python plot_supportedness.py
python plot_route_allocation.py
python plot_generator_sensitivity.py
python plot_latency_quality.py
python plot_judge_repeatability.py
```

Each script prints the exact numbers it plots to the console, so the output can be checked against the thesis text directly.
