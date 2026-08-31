import pickle
import time
import numpy as np
import pandas as pd
import faiss
import ollama
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
FAST_K = 2
DEEP_K = 5
GENERATOR_MODEL = 'llama3.2:1b'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
BASELINE_CSV = 'baseline_full.csv'
SWEEP = False
SWEEP_TAUS = [0.03, 0.0422, 0.05, 0.07]

def retrieve(embedder, index, all_chunks, chunk_to_article, question, k):
    q_emb = embedder.encode([question], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, k)
    return [{'score': float(s), 'chunk': all_chunks[i], 'article_id': chunk_to_article[i]} for s, i in zip(scores[0], indices[0])]

def build_prompt(question, retrieved_chunks):
    context = '\n\n'.join((c['chunk'] for c in retrieved_chunks))
    return f"""You are a helpful support assistant. Answer the question using only the context below. If the answer is not in the context, say "I don't know."\n\nCONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:"""

def answer_with_depth(embedder, index, all_chunks, chunk_to_article, question, k):
    start = time.time()
    retrieved = retrieve(embedder, index, all_chunks, chunk_to_article, question, k)
    prompt = build_prompt(question, retrieved)
    resp = ollama.chat(model=GENERATOR_MODEL, messages=[{'role': 'user', 'content': prompt}])
    latency = time.time() - start
    return {'answer': resp['message']['content'], 'contexts': [c['chunk'] for c in retrieved], 'latency_seconds': latency}

def run_adaptive(tau, base, embedder, index, all_chunks, chunk_to_article):
    results = []
    fast_count = 0
    to_rerun = base[base['delta_score'] >= tau]
    print(f'  tau={tau:.4f}: {len(to_rerun)} queries go FAST (re-run at Top-{FAST_K}), {len(base) - len(to_rerun)} stay DEEP (reuse baseline).')
    for _, row in tqdm(base.iterrows(), total=len(base), desc=f'  routing (tau={tau:.4f})'):
        if row['delta_score'] >= tau:
            fast_count += 1
            out = answer_with_depth(embedder, index, all_chunks, chunk_to_article, row['question'], FAST_K)
            results.append({'question': row['question'], 'path': 'fast', 'answer': out['answer'], 'contexts': out['contexts'], 'latency_seconds': out['latency_seconds'], 'delta_score': row['delta_score']})
        else:
            results.append({'question': row['question'], 'path': 'deep', 'answer': row['answer'], 'contexts': row['contexts'], 'latency_seconds': row['latency_seconds'], 'delta_score': row['delta_score']})
    return (pd.DataFrame(results), fast_count)

def summarise(name, latencies):
    lat = np.asarray(latencies, dtype=float)
    clean = lat[lat <= 60]
    return {'system': name, 'n': len(lat), 'mean_s': round(clean.mean(), 2), 'std_s': round(clean.std(), 2), 'P50_s': round(np.percentile(lat, 50), 2), 'P95_s': round(np.percentile(clean, 95), 2)}

def main():
    print('Loading index, chunks, and baseline results...')
    index = faiss.read_index('wixqa_index.faiss')
    with open('chunks.pkl', 'rb') as f:
        saved = pickle.load(f)
    all_chunks = saved['chunks']
    chunk_to_article = saved['article_ids']
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    base = pd.read_csv(BASELINE_CSV)
    base_summary = summarise('Static baseline (Top-5)', base['latency_seconds'])
    if not SWEEP:
        tau = float(np.median(base['delta_score']))
        print(f'\nRunning adaptive router at the median threshold tau = {tau:.4f}\n')
        adf, fast_count = run_adaptive(tau, base, embedder, index, all_chunks, chunk_to_article)
        adf.to_csv('adaptive_full.csv', index=False)
        adapt_summary = summarise(f'Adaptive (tau={tau:.4f})', adf['latency_seconds'])
        table = pd.DataFrame([base_summary, adapt_summary])
        print('\n==================  RESULT  ==================')
        print(table.to_string(index=False))
        print('==============================================')
        pct = 100 * (base_summary['P95_s'] - adapt_summary['P95_s']) / base_summary['P95_s']
        print(f'\nFast Path used on {fast_count} of {len(base)} queries ({100 * fast_count / len(base):.0f}%).')
        print(f'P95 latency: baseline {base_summary['P95_s']}s -> adaptive {adapt_summary['P95_s']}s  ({pct:+.1f}%).')
        print('\nSaved adaptive_full.csv. This is your core result for RQ1.')
        table.to_csv('comparison_summary.csv', index=False)
    else:
        print('\nSWEEP MODE: testing several thresholds for the trade-off table (RQ3)\n')
        rows = [base_summary]
        for tau in SWEEP_TAUS:
            adf, fast_count = run_adaptive(tau, base, embedder, index, all_chunks, chunk_to_article)
            s = summarise(f'Adaptive (tau={tau:.4f})', adf['latency_seconds'])
            s['fast_pct'] = round(100 * fast_count / len(base))
            rows.append(s)
            adf.to_csv(f'adaptive_tau_{tau:.4f}.csv', index=False)
        table = pd.DataFrame(rows)
        print('\n==================  TRADE-OFF TABLE  ==================')
        print(table.to_string(index=False))
        print('======================================================')
        table.to_csv('threshold_sweep.csv', index=False)
        print('\nSaved threshold_sweep.csv. This is your calibration evidence for RQ3.')
if __name__ == '__main__':
    main()
