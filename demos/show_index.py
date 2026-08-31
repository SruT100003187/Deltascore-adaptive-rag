"""
 
Run, type any question, Prints the top matching chunks from the
WixQA knowledge base with their similarity scores. Type 'quit' to stop.

Shows the index is built and retrieving correctly, live.
"""
import pickle
import faiss
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

print("Loading the FAISS index and knowledge-base chunks...")
index = faiss.read_index("wixqa_index.faiss")
with open("chunks.pkl", "rb") as f:
    saved = pickle.load(f)
all_chunks = saved["chunks"]
article_ids = saved["article_ids"]
embedder = SentenceTransformer(EMBEDDING_MODEL)

print()
print("=" * 60)
print(f"  Index ready:  {index.ntotal:,} chunks searchable")
print("=" * 60)
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

    print()
    print(f"  Top 5 matches for: {question}")
    print("  " + "-" * 56)
    for rank, (sc, ix) in enumerate(zip(scores[0], idxs[0]), start=1):
        snippet = all_chunks[ix][:110].replace("\n", " ")
        print(f"  #{rank}  score={sc:.3f}   article {article_ids[ix][:10]}...")
        print(f"       {snippet}...")
    print()
