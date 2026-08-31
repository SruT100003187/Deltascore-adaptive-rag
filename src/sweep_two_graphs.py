import os
import importlib.util
import numpy as np
import pandas as pd
LIMIT = None
CHECKPOINT_EVERY = 10
KS = [2, 3]
BASELINE_CSV = 'baseline_full.csv'
ADAPTIVE_CSV = 'adaptive_full.csv'
SUF = '' if LIMIT is None else f'_smoke{LIMIT}'

def load_module(name, path):
    if not os.path.exists(path):
        raise FileNotFoundError(f'Could not find {path}. Run this script from your Implementation_RAG folder.')
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
print('Loading your router (04) and judge (05) so behaviour is identical...')
router = load_module('router_mod', '04_adaptive_router.py')
judge = load_module('judge_mod', '05_faithfulness.py')
import faiss, pickle
from sentence_transformers import SentenceTransformer
print('Loading index, chunks and embedder...')
index = faiss.read_index('wixqa_index.faiss')
with open('chunks.pkl', 'rb') as f:
    saved = pickle.load(f)
all_chunks = saved['chunks']
chunk_to_article = saved['article_ids']
embedder = SentenceTransformer(router.EMBEDDING_MODEL)
base = pd.read_csv(BASELINE_CSV)
if LIMIT is not None:
    base = base.head(LIMIT).copy()
questions = list(base['question'])
d = base['delta_score'].to_numpy()
mean, sd = (float(d.mean()), float(d.std(ddof=1)))
THRESHOLDS = [('mean+2sd', mean + 2 * sd), ('mean+1sd', mean + 1 * sd), ('mean', mean), ('mean-1sd', mean - 1 * sd), ('mean-2sd', mean - 2 * sd)]
print(f'\nDeltaScore mean={mean:.4f}  sd={sd:.4f}  (from {len(base)} queries)')
for name, tau in THRESHOLDS:
    fast_n = int((d >= tau).sum())
    print(f'  {name:9} tau={tau:8.4f}  ->  Fast {fast_n:3d} / Deep {len(base) - fast_n:3d}')
print()

def ctx_to_text(contexts):
    if isinstance(contexts, list):
        return '\n\n'.join((str(x) for x in contexts))
    return judge.parse_contexts(contexts)

def generate_fast_answers(K):
    cache = f'fast_cache_K{K}{SUF}.csv'
    done = {}
    if os.path.exists(cache):
        c = pd.read_csv(cache)
        for _, r in c.iterrows():
            done[r['question']] = {'answer': r['answer'], 'contexts': r['contexts'], 'latency_seconds': float(r['latency_seconds'])}
    if K == 2 and os.path.exists(ADAPTIVE_CSV):
        adf = pd.read_csv(ADAPTIVE_CSV)
        for _, r in adf[adf['path'] == 'fast'].iterrows():
            if r['question'] in questions and r['question'] not in done:
                done[r['question']] = {'answer': r['answer'], 'contexts': r['contexts'], 'latency_seconds': float(r['latency_seconds'])}
    todo = [q for q in questions if q not in done]
    print(f'[generate] K={K}: {len(done)} reused/cached, {len(todo)} to generate')

    def flush():
        rows = [{'question': q, 'answer': v['answer'], 'contexts': v['contexts'], 'latency_seconds': v['latency_seconds']} for q, v in done.items()]
        pd.DataFrame(rows).to_csv(cache, index=False)
    for i, q in enumerate(todo, 1):
        out = router.answer_with_depth(embedder, index, all_chunks, chunk_to_article, q, K)
        done[q] = {'answer': out['answer'], 'contexts': out['contexts'], 'latency_seconds': out['latency_seconds']}
        if i % CHECKPOINT_EVERY == 0 or i == len(todo):
            flush()
            print(f'    K={K}: generated {i}/{len(todo)} (checkpointed)')
    return done

def judge_set(name, get_ctx_answer):
    cache = f'judge_{name}{SUF}.csv'
    done = {}
    if os.path.exists(cache):
        c = pd.read_csv(cache)
        done = dict(zip(c['question'], c['faithful'].astype(int)))
    todo = [q for q in questions if q not in done]
    print(f'[judge] {name}: {len(done)} cached, {len(todo)} to judge')

    def flush():
        pd.DataFrame([{'question': q, 'faithful': v} for q, v in done.items()]).to_csv(cache, index=False)
    for i, q in enumerate(todo, 1):
        ctx, ans = get_ctx_answer(q)
        done[q] = int(judge.judge_faithful(ctx_to_text(ctx), str(ans)))
        if i % CHECKPOINT_EVERY == 0 or i == len(todo):
            flush()
            print(f'    {name}: judged {i}/{len(todo)} (checkpointed)')
    return done
fast_answers = {K: generate_fast_answers(K) for K in KS}
base_by_q = {r['question']: r for _, r in base.iterrows()}
baseline_faith = judge_set('baseline', lambda q: (base_by_q[q]['contexts'], base_by_q[q]['answer']))
fast_faith = {K: judge_set(f'fastK{K}', lambda q, K=K: (fast_answers[K][q]['contexts'], fast_answers[K][q]['answer'])) for K in KS}

def latency_stats(vals):
    a = np.asarray(vals, dtype=float)
    clean = a[a <= 60]
    if len(clean) == 0:
        clean = a
    return (round(clean.mean(), 2), round(np.percentile(clean, 50), 2), round(np.percentile(clean, 95), 2))
rows = []
for K in KS:
    for label, tau in THRESHOLDS:
        lat, faith = ([], [])
        for _, row in base.iterrows():
            q = row['question']
            if row['delta_score'] >= tau:
                lat.append(fast_answers[K][q]['latency_seconds'])
                faith.append(fast_faith[K][q])
            else:
                lat.append(float(row['latency_seconds']))
                faith.append(baseline_faith[q])
        m, p50, p95 = latency_stats(lat)
        rows.append({'K': K, 'threshold': label, 'tau': round(tau, 4), 'fast_n': int((d >= tau).sum()), 'deep_n': int((d < tau).sum()), 'mean_s': m, 'P50_s': p50, 'P95_s': p95, 'faithfulness': round(float(np.mean(faith)), 3)})
res = pd.DataFrame(rows)
res.to_csv(f'sweep_results{SUF}.csv', index=False)
bm, bp50, bp95 = latency_stats(base['latency_seconds'])
bfaith = round(float(np.mean([baseline_faith[q] for q in questions])), 3)
pd.DataFrame([{'system': 'baseline_Top5', 'mean_s': bm, 'P50_s': bp50, 'P95_s': bp95, 'faithfulness': bfaith}]).to_csv(f'baseline_reference{SUF}.csv', index=False)
print('\n==================  SWEEP RESULTS  ==================')
print(res.to_string(index=False))
print('----------------------------------------------------')
print(f'BASELINE (Top-5):  P95={bp95}s   faithfulness={bfaith}')
print('====================================================')
for K in KS:
    sub = res[res['K'] == K]
    best = sub.loc[sub['faithfulness'].idxmax()]
    verdict = 'BEATS or matches baseline' if best['faithfulness'] >= bfaith else 'still below baseline'
    print(f'K={K}: best faithfulness {best['faithfulness']} at {best['threshold']} (P95 {best['P95_s']}s)  ->  {verdict} ({bfaith}).')
print(f'\nSaved sweep_results{SUF}.csv and baseline_reference{SUF}.csv.')
print('Next: run  python plot_sweep.py   to draw the two graphs.')
if LIMIT is not None:
    print('\nThis was a SMOKE TEST on', LIMIT, 'queries. If it worked, set LIMIT = None and run again.')
