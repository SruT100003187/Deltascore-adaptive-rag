import pandas as pd
import ollama
JUDGE_MODEL = 'llama3.2:1b'
TEMPERATURE = 0.0
SEED = 42
CONTEXT_CHAR_LIMIT = 8000
JUDGE_PROMPT = 'You are checking whether an ANSWER is supported by the given CONTEXT.\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}\n\nIs the ANSWER supported by the CONTEXT above? Reply with exactly one word: YES or NO.'

def parse_contexts(raw):
    if isinstance(raw, list):
        return '\n\n'.join((str(x) for x in raw))
    try:
        import ast
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return '\n\n'.join((str(x) for x in parsed))
    except Exception:
        pass
    return str(raw)

def judge_faithful(context_text, answer_text):
    context_text = context_text[:CONTEXT_CHAR_LIMIT]
    prompt = JUDGE_PROMPT.format(context=context_text, answer=answer_text)
    resp = ollama.chat(model=JUDGE_MODEL, messages=[{'role': 'user', 'content': prompt}], options={'temperature': TEMPERATURE, 'seed': SEED})
    verdict = resp['message']['content'].strip().upper()
    return 1 if verdict.startswith('YES') else 0

def rejudge_file(answers_csv, output_csv, label):
    print(f'\nRe-judging {label} ({answers_csv}) at temperature 0...')
    df = pd.read_csv(answers_csv)
    rows = []
    for i, r in df.iterrows():
        ctx = parse_contexts(r['contexts'])
        verdict = judge_faithful(ctx, str(r['answer']))
        rows.append({'question': r['question'], 'faithful': verdict})
        if (i + 1) % 25 == 0 or i + 1 == len(df):
            print(f'  {label}: {i + 1}/{len(df)}')
    out = pd.DataFrame(rows)
    out.to_csv(output_csv, index=False)
    faith_rate = out['faithful'].mean()
    print(f'{label} faithfulness at temp 0: {faith_rate:.3f}  (n={len(out)})')
    return out

def main():
    print(f'Judge model: {JUDGE_MODEL} (unchanged) | temperature={TEMPERATURE} | seed={SEED}')
    print('Generator being evaluated: llama3.2:3b\n')
    rejudge_file('baseline_full_3b.csv', 'judge_baseline_t0_3b.csv', 'Baseline (Deep, 3B)')
    rejudge_file('fast_cache_K3_3b.csv', 'judge_fastK3_t0_3b.csv', 'Fast Path (K=3, 3B)')
    print('\nDone. Saved judge_baseline_t0_3b.csv and judge_fastK3_t0_3b.csv.')
    print('Next: run finalize_experiment_3b.py for the four-seed confirmatory result.')
if __name__ == '__main__':
    main()
