"""
STEP 2 of 4 (3B generator comparison) — Generate Fast-Path (K=3) answers
with llama3.2:3b, and produce the initial (non-deterministic) judge
verdicts needed by the confirmatory pipeline.

This is a trimmed version of your original sweep_two_graphs.py:
  - Only K=3 is generated (not K=2), since K=3 is your established
    final Fast-Path depth and is all finalize_experiment_v3.py needs.
  - Points at baseline_full_3b.csv (from script 1) instead of the
    original 1B baseline.
  - All filenames are suffixed with _3b so nothing from your original
    1B run is touched.

Requires 05_faithfulness.py to be in the same folder (same judge logic
you already used for the original 1B judging pass).

Safe to interrupt and resume: cached progress is checkpointed to disk
as it goes, same as your original script.
"""

import os
import importlib.util
import numpy as np
import pandas as pd

# ---------------- settings you may touch ----------------
LIMIT = None                 # set to e.g. 10 for a quick smoke test first
CHECKPOINT_EVERY = 10
K_FAST = 3                   # only the established final Fast-Path depth
GENERATOR_MODEL = "llama3.2:3b"
BASELINE_CSV = "baseline_full_3b.csv"
ADAPTIVE_CSV = None           # no existing 3B Fast answers to reuse yet
# ----------------------------------------------------------

SUF = "_3b" if LIMIT is None else f"_3b_smoke{LIMIT}"


def load_module(name, path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}. Run this from your Implementation_RAG folder.")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

print("Loading router (04) and judge (05) modules...")
router = load_module("router_mod", "04_adaptive_router.py")
judge = load_module("judge_mod", "05_faithfulness.py")

import faiss, pickle
from sentence_transformers import SentenceTransformer

print("Loading index, chunks and embedder...")
index = faiss.read_index("wixqa_index.faiss")
with open("chunks.pkl", "rb") as f:
    saved = pickle.load(f)
all_chunks = saved["chunks"]
chunk_to_article = saved["article_ids"]
embedder = SentenceTransformer(router.EMBEDDING_MODEL)

base = pd.read_csv(BASELINE_CSV)
if LIMIT is not None:
    base = base.head(LIMIT).copy()
questions = list(base["question"])
print(f"\nGenerator for Fast-Path answers: {GENERATOR_MODEL}")
print(f"Loaded {len(base)} baseline queries from {BASELINE_CSV}\n")


def ctx_to_text(contexts):
    if isinstance(contexts, list):
        return "\n\n".join(str(x) for x in contexts)
    return judge.parse_contexts(contexts)


def generate_fast_answers(K):
    cache = f"fast_cache_K{K}{SUF}.csv"
    done = {}
    if os.path.exists(cache):
        c = pd.read_csv(cache)
        for _, r in c.iterrows():
            done[r["question"]] = {"answer": r["answer"], "contexts": r["contexts"],
                                   "latency_seconds": float(r["latency_seconds"])}
    todo = [q for q in questions if q not in done]
    print(f"[generate] K={K}: {len(done)} cached, {len(todo)} to generate with {GENERATOR_MODEL}")

    def flush():
        rows = [{"question": q, "answer": v["answer"], "contexts": v["contexts"],
                 "latency_seconds": v["latency_seconds"]} for q, v in done.items()]
        pd.DataFrame(rows).to_csv(cache, index=False)

    for i, q in enumerate(todo, 1):
        # answer_with_depth uses router.GENERATOR_MODEL internally, so we override it here
        router.GENERATOR_MODEL = GENERATOR_MODEL
        out = router.answer_with_depth(embedder, index, all_chunks, chunk_to_article, q, K)
        done[q] = {"answer": out["answer"], "contexts": out["contexts"],
                   "latency_seconds": out["latency_seconds"]}
        if i % CHECKPOINT_EVERY == 0 or i == len(todo):
            flush()
            print(f"    K={K}: generated {i}/{len(todo)} (checkpointed)")
    return done


def judge_set(name, get_ctx_answer):
    cache = f"judge_{name}{SUF}.csv"
    done = {}
    if os.path.exists(cache):
        c = pd.read_csv(cache)
        done = dict(zip(c["question"], c["faithful"].astype(int)))
    todo = [q for q in questions if q not in done]
    print(f"[judge] {name}: {len(done)} cached, {len(todo)} to judge")

    def flush():
        pd.DataFrame([{"question": q, "faithful": v} for q, v in done.items()]).to_csv(cache, index=False)

    for i, q in enumerate(todo, 1):
        ctx, ans = get_ctx_answer(q)
        done[q] = int(judge.judge_faithful(ctx_to_text(ctx), str(ans)))
        if i % CHECKPOINT_EVERY == 0 or i == len(todo):
            flush()
            print(f"    {name}: judged {i}/{len(todo)} (checkpointed)")
    return done


# ---------------- Phase 1: generate Fast (K=3) answers with 3B ----------------
fast_answers = generate_fast_answers(K_FAST)

# ---------------- Phase 2: initial judging pass (cached) ----------------
base_by_q = {r["question"]: r for _, r in base.iterrows()}
baseline_faith = judge_set("baseline_3b" if LIMIT is None else f"baseline_3b_smoke{LIMIT}",
                           lambda q: (base_by_q[q]["contexts"], base_by_q[q]["answer"]))
fast_faith = judge_set(f"fastK{K_FAST}_3b" if LIMIT is None else f"fastK{K_FAST}_3b_smoke{LIMIT}",
                       lambda q: (fast_answers[q]["contexts"], fast_answers[q]["answer"]))

print("\nDone with initial judging pass.")
print(f"Files written: fast_cache_K{K_FAST}{SUF}.csv, judge_baseline_3b.csv, judge_fastK{K_FAST}_3b.csv")
print("\nNext: run rejudge_temp0_3b.py to stabilise these verdicts at temperature 0,")
print("then run finalize_experiment_3b.py for the four-seed confirmatory result.")
