import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
TAU = 0.0422
FAST_K = 2
DEEP_K = 5
print('Loading the FAISS index and knowledge-base chunks...')
index = faiss.read_index('wixqa_index.faiss')
with open('chunks.pkl', 'rb') as f:
    saved = pickle.load(f)
all_chunks = saved['chunks']
article_ids = saved['article_ids']
embedder = SentenceTransformer(EMBEDDING_MODEL)
print()
print('=' * 62)
print(f'  Index ready: {index.ntotal:,} chunks   |   Threshold (tau) = {TAU}')
print('=' * 62)
print("Type a question and press Enter. Type 'quit' to exit.")
print()
while True:
    question = input('Your question > ').strip()
    if question.lower() in ('quit', 'exit', 'q', ''):
        print('Done.')
        break
    q_emb = embedder.encode([question], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(q_emb)
    scores, idxs = index.search(q_emb, 5)
    s = scores[0]
    top = float(s[0])
    rest_mean = float(np.mean(s[1:]))
    delta = top - rest_mean
    print()
    print(f'  Question: {question}')
    print('  ' + '-' * 58)
    print('  Top 5 match scores:')
    for rank, (sc, ix) in enumerate(zip(s, idxs[0]), start=1):
        snippet = all_chunks[ix][:70].replace('\n', ' ')
        marker = '  <- top match' if rank == 1 else ''
        print(f'    #{rank}  {sc:.3f}   {snippet}...{marker}')
    print()
    print(f'  DeltaScore = top - average(rest)')
    print(f'             = {top:.3f} - {rest_mean:.3f}')
    print(f'             = {delta:.3f}')
    print(f'  Threshold  = {TAU}')
    print()
    if delta >= TAU:
        print(f'  DECISION:  DeltaScore {delta:.3f} >= {TAU}')
        print(f'  >>> FAST PATH  (retrieve {FAST_K} chunks, shorter prompt, faster answer)')
    else:
        print(f'  DECISION:  DeltaScore {delta:.3f} < {TAU}')
        print(f'  >>> DEEP PATH  (retrieve {DEEP_K} chunks, more context, safer answer)')
    print()
