import pickle
import time
import numpy as np
import faiss
import pandas as pd
import ollama
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
SAMPLE_SIZE = 'all'
TOP_K = 5
GENERATOR_MODEL = 'llama3.2:3b'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
OUTPUT_FILE = 'baseline_full_3b.csv'

def retrieve(embedder, index, all_chunks, chunk_to_article, question, k=TOP_K):
    q_emb = embedder.encode([question], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, k)
    return [{'score': float(s), 'chunk': all_chunks[i], 'article_id': chunk_to_article[i]} for s, i in zip(scores[0], indices[0])]

def build_prompt(question, retrieved_chunks):
    context = '\n\n'.join((c['chunk'] for c in retrieved_chunks))
    return f"""You are a helpful support assistant. Answer the question using only the context below. If the answer is not in the context, say "I don't know."\n\nCONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:"""

def answer_query(embedder, index, all_chunks, chunk_to_article, question, k=TOP_K):
    start = time.time()
    retrieved = retrieve(embedder, index, all_chunks, chunk_to_article, question, k=k)
    prompt = build_prompt(question, retrieved)
    response = ollama.chat(model=GENERATOR_MODEL, messages=[{'role': 'user', 'content': prompt}])
    latency = time.time() - start
    top_score = retrieved[0]['score']
    rest_scores = [c['score'] for c in retrieved[1:]]
    delta_score = top_score - (np.mean(rest_scores) if rest_scores else 0.0)
    return {'question': question, 'answer': response['message']['content'], 'contexts': [c['chunk'] for c in retrieved], 'retrieved_article_ids': [c['article_id'] for c in retrieved], 'top_score': top_score, 'delta_score': delta_score, 'latency_seconds': latency}

def main():
    print(f'SAMPLE_SIZE setting: {SAMPLE_SIZE}')
    print(f'Generator for this run: {GENERATOR_MODEL}')
    print('Loading saved index and chunks...')
    index = faiss.read_index('wixqa_index.faiss')
    with open('chunks.pkl', 'rb') as f:
        saved = pickle.load(f)
    all_chunks = saved['chunks']
    chunk_to_article = saved['article_ids']
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    print('Loading evaluation queries (ExpertWritten + Simulated)...')
    expert = load_dataset('Wix/WixQA', 'wixqa_expertwritten')['train']
    simulated = load_dataset('Wix/WixQA', 'wixqa_simulated')['train']
    queries = [{'question': r['question'], 'gold_article_ids': r['article_ids']} for r in expert] + [{'question': r['question'], 'gold_article_ids': r['article_ids']} for r in simulated]
    if SAMPLE_SIZE != 'all':
        queries = queries[:SAMPLE_SIZE]
    print(f'Running baseline on {len(queries)} queries with K={TOP_K}, generator={GENERATOR_MODEL}...\n')
    results = []
    for q in tqdm(queries, desc='Answering queries'):
        try:
            results.append(answer_query(embedder, index, all_chunks, chunk_to_article, q['question']))
        except Exception as e:
            print(f'  Skipped a query due to error: {e}')
    df = pd.DataFrame(results)
    out_name = 'baseline_sample_3b.csv' if SAMPLE_SIZE != 'all' else OUTPUT_FILE
    df.to_csv(out_name, index=False)
    avg_latency = df['latency_seconds'].mean()
    print(f'\nDone. Saved {len(df)} results to {out_name}')
    print(f'Average latency this run: {avg_latency:.2f} seconds per query')
    if SAMPLE_SIZE != 'all':
        estimated_total_minutes = avg_latency * 400 / 60
        print(f"""\nEstimate for the full 400-query run: ~{estimated_total_minutes:.0f} minutes. Set SAMPLE_SIZE = "all" at the top of this file and rerun when you're ready.""")
if __name__ == '__main__':
    main()
