"""
LIVE ROUTING DEMO - show the professor the DeltaScore decision happening live.

Type any question. It shows:
  - the top 5 chunks and their scores
  - the DeltaScore (top score minus the average of the rest)
  - the threshold
  - the routing decision: FAST PATH or DEEP PATH

Two good questions to try in the meeting:
  FAST:  how can I request a refund for Premium or Studio plans
  DEEP:  How do I reset my password?

Type 'quit' to exit.
"""
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TAU = 0.0422          # my threshold: the median DeltaScore from the experiment
FAST_K = 2            # Fast Path retrieves 2 chunks
DEEP_K = 5            # Deep Path retrieves 5 chunks

print("Loading the FAISS index and knowledge-base chunks...")
index = faiss.read_index("wixqa_index.faiss")
with open("chunks.pkl", "rb") as f:
    saved = pickle.load(f)
all_chunks = saved["chunks"]
article_ids = saved["article_ids"]
embedder = SentenceTransformer(EMBEDDING_MODEL)

print()
print("=" * 62)
print(f"  Index ready: {index.ntotal:,} chunks   |   Threshold (tau) = {TAU}")
print("=" * 62)
print("Type a question and press Enter. Type 'quit' to exit.")
print()

while True:
    question = input("Your question > ").strip()
    if question.lower() in ("quit", "exit", "q", ""):
        print("Done.")
        break

    q_emb = embedder.encode([question], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, idxs = index.search(q_emb, 5)
    s = scores[0]

    # DeltaScore = top score minus the average of the rest
    top = float(s[0])
    rest_mean = float(np.mean(s[1:]))
    delta = top - rest_mean

    print()
    print(f"  Question: {question}")
    print("  " + "-" * 58)
    print("  Top 5 match scores:")
    for rank, (sc, ix) in enumerate(zip(s, idxs[0]), start=1):
        snippet = all_chunks[ix][:70].replace("\n", " ")
        marker = "  <- top match" if rank == 1 else ""
        print(f"    #{rank}  {sc:.3f}   {snippet}...{marker}")

    print()
    print(f"  DeltaScore = top - average(rest)")
    print(f"             = {top:.3f} - {rest_mean:.3f}")
    print(f"             = {delta:.3f}")
    print(f"  Threshold  = {TAU}")
    print()

    if delta >= TAU:
        print(f"  DECISION:  DeltaScore {delta:.3f} >= {TAU}")
        print(f"  >>> FAST PATH  (retrieve {FAST_K} chunks, shorter prompt, faster answer)")
    else:
        print(f"  DECISION:  DeltaScore {delta:.3f} < {TAU}")
        print(f"  >>> DEEP PATH  (retrieve {DEEP_K} chunks, more context, safer answer)")
    print()
