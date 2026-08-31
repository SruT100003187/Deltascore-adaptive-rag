"""
STEP 4 of the thesis - The DeltaScore Adaptive Router.

WHAT THIS DOES
This is the core experiment of your thesis. It takes the retrieval scores
already saved in baseline_full.csv, applies the DeltaScore routing rule, and
re-runs ONLY the queries whose retrieval depth changes. Then it compares the
adaptive system's P95 latency against your static baseline (24.0s).

THE ROUTING RULE
  DeltaScore = top_score - mean(remaining top-K scores)   [already in your CSV]
  If DeltaScore >= tau  -> FAST PATH  (retrieve fewer chunks)
  If DeltaScore <  tau  -> DEEP PATH  (retrieve more chunks)

CHOICES MADE FOR YOU (defensible defaults, change at the top if you want):
  FAST_K = 2     (Fast Path retrieves Top-2)
  DEEP_K = 5     (Deep Path retrieves Top-5, same as baseline)
  tau    = median DeltaScore on this data (a simple, data-driven threshold)

HOW TO RUN
  1) First run leaves SWEEP = False. It runs ONE threshold (the median),
     produces the adaptive result, and prints the headline comparison.
  2) On Day 3, set SWEEP = True to test several thresholds and produce the
     trade-off table for RQ3.

This reuses your existing index and the same generator, so the comparison is fair.
"""

import pickle
import time
import numpy as np
import pandas as pd
import faiss
import ollama
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---- Settings you may touch -------------------------------------------------
FAST_K = 2                      # Fast Path retrieval depth
DEEP_K = 5                      # Deep Path retrieval depth (matches baseline)
GENERATOR_MODEL = "llama3.2:1b" # same model as the baseline (keep it fixed)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BASELINE_CSV = "baseline_full.csv"
SWEEP = False                   # Day 1: False (single run). Day 3: True (sweep).
SWEEP_TAUS = [0.03, 0.0422, 0.05, 0.07]  # used only when SWEEP = True


def retrieve(embedder, index, all_chunks, chunk_to_article, question, k):
    q_emb = embedder.encode([question], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, k)
    return [
        {"score": float(s), "chunk": all_chunks[i], "article_id": chunk_to_article[i]}
        for s, i in zip(scores[0], indices[0])
    ]


def build_prompt(question, retrieved_chunks):
    context = "\n\n".join(c["chunk"] for c in retrieved_chunks)
    return (
        "You are a helpful support assistant. Answer the question using "
        "only the context below. If the answer is not in the context, "
        'say "I don\'t know."\n\n'
        f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:"
    )


def answer_with_depth(embedder, index, all_chunks, chunk_to_article, question, k):
    """Run one query end to end at retrieval depth k; return answer, contexts, latency."""
    start = time.time()
    retrieved = retrieve(embedder, index, all_chunks, chunk_to_article, question, k)
    prompt = build_prompt(question, retrieved)
    resp = ollama.chat(model=GENERATOR_MODEL, messages=[{"role": "user", "content": prompt}])
    latency = time.time() - start
    return {
        "answer": resp["message"]["content"],
        "contexts": [c["chunk"] for c in retrieved],
        "latency_seconds": latency,
    }


def run_adaptive(tau, base, embedder, index, all_chunks, chunk_to_article):
    """
    Apply DeltaScore routing at threshold tau.
    Fast Path (delta >= tau) -> retrieve FAST_K. Deep Path -> retrieve DEEP_K.
    The baseline already used DEEP_K (5), so Deep-Path queries reuse the
    baseline latency directly. Only Fast-Path queries are re-run.
    """
    results = []
    fast_count = 0
    to_rerun = base[base["delta_score"] >= tau]
    print(f"  tau={tau:.4f}: {len(to_rerun)} queries go FAST (re-run at Top-{FAST_K}), "
          f"{len(base) - len(to_rerun)} stay DEEP (reuse baseline).")

    for _, row in tqdm(base.iterrows(), total=len(base), desc=f"  routing (tau={tau:.4f})"):
        if row["delta_score"] >= tau:
            fast_count += 1
            out = answer_with_depth(embedder, index, all_chunks, chunk_to_article,
                                    row["question"], FAST_K)
            results.append({
                "question": row["question"],
                "path": "fast",
                "answer": out["answer"],
                "contexts": out["contexts"],
                "latency_seconds": out["latency_seconds"],
                "delta_score": row["delta_score"],
            })
        else:
            # Deep Path = same depth as baseline, so reuse the baseline result
            results.append({
                "question": row["question"],
                "path": "deep",
                "answer": row["answer"],
                "contexts": row["contexts"],
                "latency_seconds": row["latency_seconds"],
                "delta_score": row["delta_score"],
            })
    return pd.DataFrame(results), fast_count


def summarise(name, latencies):
    lat = np.asarray(latencies, dtype=float)
    clean = lat[lat <= 60]  # drop the same machine-stall outliers as the baseline
    return {
        "system": name,
        "n": len(lat),
        "mean_s": round(clean.mean(), 2),
        "std_s": round(clean.std(), 2),
        "P50_s": round(np.percentile(lat, 50), 2),
        "P95_s": round(np.percentile(clean, 95), 2),
    }


def main():
    print("Loading index, chunks, and baseline results...")
    index = faiss.read_index("wixqa_index.faiss")
    with open("chunks.pkl", "rb") as f:
        saved = pickle.load(f)
    all_chunks = saved["chunks"]
    chunk_to_article = saved["article_ids"]
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    base = pd.read_csv(BASELINE_CSV)

    base_summary = summarise("Static baseline (Top-5)", base["latency_seconds"])

    if not SWEEP:
        tau = float(np.median(base["delta_score"]))
        print(f"\nRunning adaptive router at the median threshold tau = {tau:.4f}\n")
        adf, fast_count = run_adaptive(tau, base, embedder, index, all_chunks, chunk_to_article)
        adf.to_csv("adaptive_full.csv", index=False)
        adapt_summary = summarise(f"Adaptive (tau={tau:.4f})", adf["latency_seconds"])

        table = pd.DataFrame([base_summary, adapt_summary])
        print("\n==================  RESULT  ==================")
        print(table.to_string(index=False))
        print("==============================================")
        pct = 100 * (base_summary["P95_s"] - adapt_summary["P95_s"]) / base_summary["P95_s"]
        print(f"\nFast Path used on {fast_count} of {len(base)} queries "
              f"({100*fast_count/len(base):.0f}%).")
        print(f"P95 latency: baseline {base_summary['P95_s']}s -> "
              f"adaptive {adapt_summary['P95_s']}s  ({pct:+.1f}%).")
        print("\nSaved adaptive_full.csv. This is your core result for RQ1.")
        table.to_csv("comparison_summary.csv", index=False)
    else:
        print("\nSWEEP MODE: testing several thresholds for the trade-off table (RQ3)\n")
        rows = [base_summary]
        for tau in SWEEP_TAUS:
            adf, fast_count = run_adaptive(tau, base, embedder, index, all_chunks, chunk_to_article)
            s = summarise(f"Adaptive (tau={tau:.4f})", adf["latency_seconds"])
            s["fast_pct"] = round(100 * fast_count / len(base))
            rows.append(s)
            adf.to_csv(f"adaptive_tau_{tau:.4f}.csv", index=False)
        table = pd.DataFrame(rows)
        print("\n==================  TRADE-OFF TABLE  ==================")
        print(table.to_string(index=False))
        print("======================================================")
        table.to_csv("threshold_sweep.csv", index=False)
        print("\nSaved threshold_sweep.csv. This is your calibration evidence for RQ3.")


if __name__ == "__main__":
    main()
