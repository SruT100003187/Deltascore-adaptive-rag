# DeltaScore: Adaptive Retrieval-Depth Routing for Enterprise RAG

Master's thesis project — M.Sc. Computer Science (Big Data and AI), SRH University of Applied Sciences, Leipzig.
Author: Sruthi | Supervisors: Prof. Dr. Fakhteh Ghanbarnejad, Prof. Dr.-Ing. Joel Dokmegang

> **Status:** the state of this repo at thesis submission is tagged [`v1.0-thesis-submission`](../../releases) — see that tag/release for the exact snapshot the thesis results are based on. `main` may move ahead of it afterwards.

## Overview

Enterprise RAG systems typically retrieve a fixed number of context chunks (Top-K) for every query, regardless of how much evidence the query actually needs. This project introduces **DeltaScore**, a training-free routing signal computed from retrieval-score separation, to decide — before generation — whether a query needs a shallow ("Fast Path", K=3) or deep ("Deep Path", K=5) retrieval pass.

- **DeltaScore** = top retrieval score − mean of the remaining top-K scores
- **Retriever:** all-MiniLM-L6-v2 embeddings, FAISS `IndexFlatIP`
- **Generator / judge:** local Llama 3.2 1B via Ollama
- **Corpus:** WixQA (6,221 articles, 23,271 chunks)
- **Evaluation:** four-seed confirmatory held-out evaluation (seeds 42, 7, 123, 2024), McNemar's exact test for paired supportedness comparisons

## Key results

- Routing ~60% of queries to the Fast Path cut mean latency by ~19.5–24.9% and P95 latency by ~7.6–12.5%, with no statistically detectable drop in answer supportedness (McNemar p-values in [0.079, 0.875]) across all four seeds.
- DeltaScore is a **weak individual-query classifier** (deep-needed queries appeared at similar rates in both paths: 25.1% Deep-routed vs 24.4% Fast-routed), but a **useful system-level routing signal** — the ~60% operating region is development-calibrated rather than a closed-form optimum.
- A supplementary check with a larger (3B) generator confirmed the latency pattern held; the quality comparison at 3B was inconclusive due to judge instability (see `results/supplementary_3b/`).

## Repository structure

```
.
├── src/                          # Final pipeline, run in this order
│   ├── 01_build_index.py         #   builds the FAISS index over WixQA
│   ├── 02_run_baseline.py        #   static Top-5 baseline + DeltaScore logging
│   ├── 04_adaptive_router.py     #   applies DeltaScore routing, Fast/Deep paths
│   ├── sweep_two_graphs.py       #   RQ3 threshold sweep (K=2 vs K=3, 5-point ladder)
│   ├── plot_sweep.py             #   renders the sweep graphs
│   ├── rejudge_temp0.py          #   deterministic (temp-0) supportedness judge
│   ├── significance_test.py      #   McNemar's exact test on paired verdicts
│   └── finalize_experiment_v3.py #   final four-seed confirmatory evaluation (headline numbers)
│
├── src/supplementary_3b/         # 3B-generator comparison (Section 5.8 / Appendix F)
│
├── demos/                        # Interactive/manual scripts, not part of the evaluation pipeline
│   ├── routing_demo.py           #   live DeltaScore routing demo (used for supervisor meetings)
│   └── show_index.py             #   ad-hoc retrieval inspection tool
│
├── results/
│   ├── final/                    # Curated evidence backing the thesis's reported numbers
│   └── supplementary_3b/         # 3B comparison results (latency only, see thesis Appendix F)
│
├── figures/                      # Sweep graphs (Chapter 5 / Appendix B)
│
├── configs/
│   └── reproducibility_manifest.md   # Fixed components, prompts, and known limitations (Appendix A)
│
├── legacy/                       # Superseded scripts and exploratory intermediate files, kept
│                                  # for transparency/audit trail — NOT the pipeline behind the
│                                  # thesis's final numbers. See legacy/README.md.
│
├── data/                         # Large generated artefacts excluded from git — see data/README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Reproducing the pipeline

**Important:** the scripts are unmodified from what actually produced the thesis's reported results — they read and write files by bare filename (`"baseline_full.csv"`, `"wixqa_index.faiss"`, etc.) in whatever directory they're run from, rather than a `src/`/`results/`/`data/` layout. That's deliberate: reorganising the files into folders for readability shouldn't mean silently editing the analysis code itself. So to actually re-run the pipeline, work in one flat folder rather than relying on the folder structure above for execution:

```bash
mkdir run && cd run
cp ../src/*.py .                        # copy the pipeline scripts into one working folder
pip install -r ../requirements.txt

# Ollama must be installed separately: https://ollama.com
ollama pull llama3.2:1b

python 01_build_index.py                # -> wixqa_index.faiss, chunks.pkl, chunk_embeddings.npy
python 02_run_baseline.py               # -> baseline_full.csv (static Top-5 + DeltaScore)
python sweep_two_graphs.py              # -> sweep_results.csv (RQ3 calibration sweep)
python plot_sweep.py                    # -> graph_K2.png, graph_K3.png, graph_faithfulness_K2_vs_K3.png
python 04_adaptive_router.py            # -> fast_cache_K3.csv (Fast-Path answers at chosen K)
python rejudge_temp0.py                 # -> judge_baseline_t0.csv, judge_fastK3_t0.csv
python significance_test.py             # -> significance_results.csv (McNemar test)
python finalize_experiment_v3.py        # -> FINAL_results_allseeds.txt, final_results.csv (headline numbers)
```

For the 3B supplementary comparison, copy the scripts from `src/supplementary_3b/` into the same working folder and run them in the same order, after the 1B pipeline has produced its index and baseline (they read the 1B outputs as inputs).

The `results/` and `figures/` folders in this repo already contain the actual saved outputs from these runs — re-running is only necessary if you want to regenerate or extend the experiment, not to inspect what was reported in the thesis.

Exact model configuration, the judge and generator prompt templates, and known reproducibility limitations (e.g. unpinned library/model versions) are documented in `configs/reproducibility_manifest.md`, mirroring Appendix A of the thesis.

## Note on data

The WixQA corpus itself is not redistributed here — see the [WixQA benchmark](https://huggingface.co/datasets) for access via Hugging Face `datasets`. The FAISS index, chunk embeddings, and chunk store built from it are regenerated locally by `01_build_index.py`; see `data/README.md`.

## Citation

If referencing this work:

```
Sruthi. (2026). DeltaScore: Adaptive Retrieval-Depth Routing for Enterprise RAG Systems.
M.Sc. Thesis, SRH University of Applied Sciences, Leipzig.
```

## License

See `LICENSE`.
