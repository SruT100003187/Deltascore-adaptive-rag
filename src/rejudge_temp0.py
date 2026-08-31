import csv, json, sys, time, urllib.request
JUDGE_MODEL = 'llama3.2:1b'
OLLAMA_URL = 'http://localhost:11434/api/generate'
TEMPERATURE = 0
SEED = 42
REPEAT_CHECK_N = 50
CONTEXT_CHAR_CAP = 8000
TIMEOUT = 180
JUDGE_PROMPT = 'You are checking whether an ANSWER is supported by the given CONTEXT.\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}\n\nIs the ANSWER supported by the CONTEXT above? Reply with exactly one word: YES or NO.'

def judge(context, answer):
    context = str(context)[:CONTEXT_CHAR_CAP]
    answer = str(answer)
    prompt = JUDGE_PROMPT.format(context=context, answer=answer)
    payload = json.dumps({'model': JUDGE_MODEL, 'prompt': prompt, 'stream': False, 'options': {'temperature': TEMPERATURE, 'seed': SEED}}).encode('utf-8')
    req = urllib.request.Request(OLLAMA_URL, data=payload, headers={'Content-Type': 'application/json'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                out = json.loads(resp.read().decode('utf-8')).get('response', '')
            text = out.strip().lower()
            return 1 if text.startswith('yes') or ' yes' in text[:20] else 0
        except Exception as e:
            if attempt == 2:
                print(f'\n  judge call failed after 3 tries: {e}')
                return 0
            time.sleep(2)

def read_rows(path):
    try:
        with open(path, newline='', encoding='utf-8') as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        sys.exit(f'Missing {path}. Run this in your Implementation_RAG folder.')

def judge_file(path, label):
    rows = read_rows(path)
    n = len(rows)
    print(f'\nJudging {label}: {n} answers ...')
    verdicts = []
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        v = judge(r.get('contexts', ''), r.get('answer', ''))
        verdicts.append((r.get('question', ''), v))
        if i % 25 == 0 or i == n:
            rate = i / max(time.time() - t0, 1e-06)
            eta = (n - i) / max(rate, 1e-06)
            print(f'  {i}/{n}   faithful so far: {sum((v for _, v in verdicts))}   ~{eta:0.0f}s left')
    return verdicts

def write_verdicts(path, verdicts):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['question', 'faithful'])
        w.writerows(verdicts)
    print(f'  wrote {path}')
print('=' * 66)
print('DETERMINISTIC RE-JUDGE  (temperature 0, fixed seed)')
print('=' * 66)
base_v = judge_file('baseline_full.csv', 'baseline (Deep, K=5)')
write_verdicts('judge_baseline_t0.csv', base_v)
fast_v = judge_file('fast_cache_K3.csv', 'Fast Path (K=3)')
write_verdicts('judge_fastK3_t0.csv', fast_v)
base_faith = sum((v for _, v in base_v)) / max(len(base_v), 1)
fast_faith = sum((v for _, v in fast_v)) / max(len(fast_v), 1)
print(f'\nRepeatability check: re-judging the first {REPEAT_CHECK_N} baseline answers ...')
rows = read_rows('baseline_full.csv')[:REPEAT_CHECK_N]
disagree = 0
for r in rows:
    v2 = judge(r.get('contexts', ''), r.get('answer', ''))
    v1 = dict(base_v).get(r.get('question', ''))
    if v1 is not None and v1 != v2:
        disagree += 1
print('\n' + '=' * 66)
print('SUMMARY')
print('=' * 66)
print(f'Baseline (Deep) faithfulness at temp 0 : {base_faith:.3f}  (n={len(base_v)})')
print(f'Fast Path (K=3) faithfulness at temp 0 : {fast_faith:.3f}  (n={len(fast_v)})')
print(f'Repeatability: {disagree} of {REPEAT_CHECK_N} verdicts changed on a second run (noise floor = {disagree / max(REPEAT_CHECK_N, 1):.3f}).')
print()
print('NEXT: swap these steady verdicts in and rerun the final experiment across seeds:')
print('  copy judge_baseline.csv judge_baseline_noisy.csv')
print('  copy judge_fastK3.csv  judge_fastK3_noisy.csv')
print('  copy judge_baseline_t0.csv judge_baseline.csv')
print('  copy judge_fastK3_t0.csv  judge_fastK3.csv')
print('  (then rerun finalize_experiment_v2.py for SEED = 42, 7, 123, 2024)')
