import os
from ast import literal_eval
import numpy as np
import pandas as pd

def main():
    df = pd.read_csv('baseline_full.csv')
    df['contexts'] = df['contexts'].apply(literal_eval)
    p50 = np.percentile(df['latency_seconds'], 50)
    p95 = np.percentile(df['latency_seconds'], 95)
    p99 = np.percentile(df['latency_seconds'], 99)
    print('=== Static Top-5 Baseline: Latency ===')
    print(f'P50: {p50:.2f}s')
    print(f'P95: {p95:.2f}s   <- this is your primary thesis metric')
    print(f'P99: {p99:.2f}s')
    if 'OPENAI_API_KEY' not in os.environ:
        print("\nHeads up: RAGAS needs an LLM to judge faithfulness, and by default it looks for OPENAI_API_KEY. If you don't have one yet, that's fine, your P95 number above is already saved. Come back before running this part again and we'll wire RAGAS to use your local Ollama model instead, no OpenAI key needed.")
        faithfulness_mean = None
        relevancy_mean = None
    else:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        ragas_ds = Dataset.from_pandas(df[['question', 'answer', 'contexts']])
        ragas_result = evaluate(ragas_ds, metrics=[faithfulness, answer_relevancy])
        ragas_df = ragas_result.to_pandas()
        ragas_df.to_csv('baseline_ragas_scores.csv', index=False)
        faithfulness_mean = ragas_df['faithfulness'].mean()
        relevancy_mean = ragas_df['answer_relevancy'].mean()
        print('\n=== Static Top-5 Baseline: Faithfulness ===')
        print(f'Mean faithfulness:     {faithfulness_mean:.3f}')
        print(f'Mean answer_relevancy: {relevancy_mean:.3f}')
    print('\n=== Summary table for your progress report ===')
    summary = pd.DataFrame([{'system': 'Static Top-5 baseline', 'n_queries': len(df), 'P50_latency_s': round(p50, 2), 'P95_latency_s': round(p95, 2), 'P99_latency_s': round(p99, 2), 'mean_faithfulness': faithfulness_mean, 'mean_answer_relevancy': relevancy_mean}])
    print(summary.to_string(index=False))
    summary.to_csv('baseline_summary.csv', index=False)
    print('\nSaved baseline_summary.csv: this is your two-week deliverable.')
if __name__ == '__main__':
    main()
