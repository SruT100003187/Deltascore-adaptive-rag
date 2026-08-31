"""
STEP 5 of the thesis - Faithfulness Check (answer quality).

WHY THIS MATTERS
You proved the router is faster (RQ1). Now you must prove it did NOT make
answers worse. "Faithfulness" means: are the claims in the answer actually
supported by the documents that were retrieved? If the adaptive system stays
about as faithful as the baseline while being faster, that is your RQ2 result.

HOW IT WORKS (no API key needed)
For each answer, a local judge model (your Ollama llama3.2) is shown the
retrieved context and the answer, and asked whether the answer is supported
by the context. We do this for the baseline answers and the adaptive answers,
then compare the two faithfulness rates.

HOW TO RUN
  1) First run leaves SAMPLE_SIZE = "all". This scores 40 baseline + 40 adaptive
     answers so you can confirm it works and see an early number (~15-20 min).
  2) Then set SAMPLE_SIZE = "all" to score all 400 of each for the final
     number. This is slower (it makes 800 judge calls), so start it and leave it.

This uses the SAME local model, so it costs nothing and needs no internet.
"""

import ast
import time
import numpy as np
import pandas as pd
import ollama
from tqdm import tqdm

# ---- Settings you may touch -------------------------------------------------
SAMPLE_SIZE = "all"                 # start at 40; set to "all" for the final run
JUDGE_MODEL = "llama3.2:1b"      # local judge, same model family as generation
BASELINE_CSV = "baseline_full.csv"
ADAPTIVE_CSV = "adaptive_full.csv"


def parse_contexts(cell):
    """The contexts column is stored as a string that looks like a Python list."""
    try:
        val = ast.literal_eval(cell)
        if isinstance(val, list):
            return "\n\n".join(str(x) for x in val)
        return str(val)
    except Exception:
        return str(cell)


def judge_faithful(context, answer):
    """
    Ask the local model a yes/no question: is the answer supported by the context?
    Returns 1 if faithful, 0 if not. Robust to messy model output.
    """
    prompt = (
        "You are checking whether an ANSWER is supported by the CONTEXT.\n"
        "Reply with exactly one word: YES if every claim in the answer is "
        "supported by the context, or NO if any claim is not supported.\n\n"
        f"CONTEXT:\n{context}\n\nANSWER:\n{answer}\n\n"
        "Is the answer fully supported by the context? Reply YES or NO:"
    )
    try:
        resp = ollama.chat(model=JUDGE_MODEL, messages=[{"role": "user", "content": prompt}])
        text = resp["message"]["content"].strip().lower()
        # take the first yes/no we see
        if "yes" in text[:10]:
            return 1
        if "no" in text[:10]:
            return 0
        # fallback: look anywhere
        return 1 if "yes" in text else 0
    except Exception as e:
        print("  (judge error, counting as unsupported):", e)
        return 0


def score_frame(df, label):
    rows = df if SAMPLE_SIZE == "all" else df.head(SAMPLE_SIZE)
    scores = []
    for _, row in tqdm(rows.iterrows(), total=len(rows), desc=f"  judging {label}"):
        ctx = parse_contexts(row["contexts"])
        scores.append(judge_faithful(ctx, str(row["answer"])))
    return np.mean(scores), len(scores)


def main():
    print("Loading baseline and adaptive answers...")
    base = pd.read_csv(BASELINE_CSV)
    adapt = pd.read_csv(ADAPTIVE_CSV)

    n = "all 400" if SAMPLE_SIZE == "all" else f"the first {SAMPLE_SIZE}"
    print(f"Scoring faithfulness on {n} answers of each system.\n")

    base_score, base_n = score_frame(base, "baseline")
    adapt_score, adapt_n = score_frame(adapt, "adaptive")

    table = pd.DataFrame([
        {"system": "Static baseline (Top-5)", "n_scored": base_n,
         "faithfulness": round(base_score, 3)},
        {"system": "Adaptive (DeltaScore)", "n_scored": adapt_n,
         "faithfulness": round(adapt_score, 3)},
    ])

    print("\n==============  FAITHFULNESS  ==============")
    print(table.to_string(index=False))
    print("===========================================")
    diff = adapt_score - base_score
    print(f"\nDifference (adaptive minus baseline): {diff:+.3f}")
    if abs(diff) <= 0.05:
        print("Interpretation: faithfulness is essentially unchanged. The adaptive "
              "system is faster WITHOUT harming answer quality. This is a strong RQ2 result.")
    elif diff > 0.05:
        print("Interpretation: the adaptive system is actually slightly MORE faithful. "
              "Report this honestly and cautiously.")
    else:
        print("Interpretation: faithfulness dropped somewhat. Note this as a real "
              "trade-off; the threshold sweep on Day 3 can help find a better balance.")

    out = "faithfulness_sample.csv" if SAMPLE_SIZE != "all" else "faithfulness_full.csv"
    table.to_csv(out, index=False)
    print(f"\nSaved {out}.")
    if SAMPLE_SIZE != "all":
        print('When this looks right, set SAMPLE_SIZE = "all" at the top and run again '
              "for the final number.")


if __name__ == "__main__":
    main()
