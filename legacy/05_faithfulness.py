import ast
import time
import numpy as np
import pandas as pd
import ollama
from tqdm import tqdm
SAMPLE_SIZE = 'all'
JUDGE_MODEL = 'llama3.2:1b'
BASELINE_CSV = 'baseline_full.csv'
ADAPTIVE_CSV = 'adaptive_full.csv'

def parse_contexts(cell):
    try:
        val = ast.literal_eval(cell)
        if isinstance(val, list):
            return '\n\n'.join((str(x) for x in val))
        return str(val)
    except Exception:
        return str(cell)

def judge_faithful(context, answer):
    prompt = f'You are checking whether an ANSWER is supported by the CONTEXT.\nReply with exactly one word: YES if every claim in the answer is supported by the context, or NO if any claim is not supported.\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}\n\nIs the answer fully supported by the context? Reply YES or NO:'
    try:
        resp = ollama.chat(model=JUDGE_MODEL, messages=[{'role': 'user', 'content': prompt}])
        text = resp['message']['content'].strip().lower()
        if 'yes' in text[:10]:
            return 1
        if 'no' in text[:10]:
            return 0
        return 1 if 'yes' in text else 0
    except Exception as e:
        print('  (judge error, counting as unsupported):', e)
        return 0

def score_frame(df, label):
    rows = df if SAMPLE_SIZE == 'all' else df.head(SAMPLE_SIZE)
    scores = []
    for _, row in tqdm(rows.iterrows(), total=len(rows), desc=f'  judging {label}'):
        ctx = parse_contexts(row['contexts'])
        scores.append(judge_faithful(ctx, str(row['answer'])))
    return (np.mean(scores), len(scores))

def main():
    print('Loading baseline and adaptive answers...')
    base = pd.read_csv(BASELINE_CSV)
    adapt = pd.read_csv(ADAPTIVE_CSV)
    n = 'all 400' if SAMPLE_SIZE == 'all' else f'the first {SAMPLE_SIZE}'
    print(f'Scoring faithfulness on {n} answers of each system.\n')
    base_score, base_n = score_frame(base, 'baseline')
    adapt_score, adapt_n = score_frame(adapt, 'adaptive')
    table = pd.DataFrame([{'system': 'Static baseline (Top-5)', 'n_scored': base_n, 'faithfulness': round(base_score, 3)}, {'system': 'Adaptive (DeltaScore)', 'n_scored': adapt_n, 'faithfulness': round(adapt_score, 3)}])
    print('\n==============  FAITHFULNESS  ==============')
    print(table.to_string(index=False))
    print('===========================================')
    diff = adapt_score - base_score
    print(f'\nDifference (adaptive minus baseline): {diff:+.3f}')
    if abs(diff) <= 0.05:
        print('Interpretation: faithfulness is essentially unchanged. The adaptive system is faster WITHOUT harming answer quality. This is a strong RQ2 result.')
    elif diff > 0.05:
        print('Interpretation: the adaptive system is actually slightly MORE faithful. Report this honestly and cautiously.')
    else:
        print('Interpretation: faithfulness dropped somewhat. Note this as a real trade-off; the threshold sweep on Day 3 can help find a better balance.')
    out = 'faithfulness_sample.csv' if SAMPLE_SIZE != 'all' else 'faithfulness_full.csv'
    table.to_csv(out, index=False)
    print(f'\nSaved {out}.')
    if SAMPLE_SIZE != 'all':
        print('When this looks right, set SAMPLE_SIZE = "all" at the top and run again for the final number.')
if __name__ == '__main__':
    main()
