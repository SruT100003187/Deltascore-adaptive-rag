"""
STEP 1 of 3 — Build the WixQA retrieval index.

What this script does, in order:
  1. Loads all four WixQA configs from Hugging Face
  2. Chunks the knowledge-base corpus with a FIXED chunk size
     (this choice is locked in now and should not change for the
     rest of the thesis — see Chapter 1 / Methodology)
  3. Embeds every chunk with a Sentence-BERT bi-encoder
  4. Builds a FAISS index over the embeddings (cosine similarity
     via normalized inner product)
  5. Saves everything to disk so 02_run_baseline.py can load it
  6. Runs one sanity-check retrieval so you can see it actually
     working before moving on

Run this once. Expect a few minutes depending on your machine —
the embedding step is the slow part.
"""

import pickle
import numpy as np
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# ── Fixed methodology choices — do not change these once you start ─────
# evaluating, or you will not be able to compare runs fairly.
CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 100    # characters of overlap between adjacent chunks
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


def main():
    print("Step 1/6 — loading WixQA from Hugging Face...")
    kb = load_dataset("Wix/WixQA", "wix_kb_corpus")["train"]
    expert = load_dataset("Wix/WixQA", "wixqa_expertwritten")["train"]
    simulated = load_dataset("Wix/WixQA", "wixqa_simulated")["train"]
    synthetic = load_dataset("Wix/WixQA", "wixqa_synthetic")["train"]

    print(f"  KB corpus:      {len(kb)} articles")
    print(f"  ExpertWritten:  {len(expert)} queries")
    print(f"  Simulated:      {len(simulated)} queries")
    print(f"  Synthetic:      {len(synthetic)} queries")

    print("\nStep 2/6 — chunking the knowledge-base corpus...")
    all_chunks = []
    chunk_to_article = []
    for row in kb:
        for piece in chunk_text(row["contents"]):
            all_chunks.append(piece)
            chunk_to_article.append(row["id"])
    print(f"  Produced {len(all_chunks)} chunks from {len(kb)} articles")

    print(f"\nStep 3/6 — loading the embedding model ({EMBEDDING_MODEL})...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    print("\nStep 4/6 — embedding all chunks (slow step, be patient)...")
    chunk_embeddings = embedder.encode(
        all_chunks, show_progress_bar=True, convert_to_numpy=True
    ).astype("float32")

    print("\nStep 5/6 — building the FAISS index...")
    faiss.normalize_L2(chunk_embeddings)  # makes inner product == cosine similarity
    dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(chunk_embeddings)
    print(f"  Indexed {index.ntotal} chunks at dimension {dimension}")

    print("\nStep 6/6 — saving everything to disk...")
    faiss.write_index(index, "wixqa_index.faiss")
    with open("chunks.pkl", "wb") as f:
        pickle.dump({"chunks": all_chunks, "article_ids": chunk_to_article}, f)
    np.save("chunk_embeddings.npy", chunk_embeddings)
    print("  Saved: wixqa_index.faiss, chunks.pkl, chunk_embeddings.npy")

    # ── Sanity check: retrieve for one real question ────────────────────
    print("\n--- Sanity check ---")
    test_question = expert[0]["question"]
    q_emb = embedder.encode([test_question], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, 5)

    print(f"QUESTION: {test_question}\n")
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        print(f"  #{rank}  score={score:.3f}  article={chunk_to_article[idx][:12]}...")
        print(f"        {all_chunks[idx][:120]}...")

    print(
        "\nIf the top result above looks topically related to the "
        "question, your index is working. Move on to 02_run_baseline.py."
    )


if __name__ == "__main__":
    main()
