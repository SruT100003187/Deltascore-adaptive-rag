# Reproducibility Configuration Manifest

Mirrors Appendix A of the thesis. Every fixed component of the study, so the pipeline can be reconstructed component by component.

| Component | Study value | Note |
|---|---|---|
| Dataset | Wix/WixQA | Exact revision hash was not pinned during the study — a known reproducibility limitation. |
| Corpus | 6,221 Wix knowledge-base articles | Matches the released WixQA corpus (Cohen et al., 2025). |
| Chunking | 800 characters, 100 overlap | Fixed. |
| Chunks | 23,271 | — |
| Embedding model | all-MiniLM-L6-v2 | Specific library version not pinned — known limitation. |
| Embedding dimension | 384 | Fixed. |
| Index | FAISS IndexFlatIP | Specific library version/index checksum not pinned — known limitation. |
| Vector normalisation | L2; inner product used as cosine | Fixed. |
| Baseline | Static Top-5 | Fixed. |
| Generator | llama3.2:1b via Ollama | Specific model digest / Ollama version not pinned — known limitation. |
| Judge | llama3.2:1b, binary YES/NO, temperature 0 | Measured repeat variation (noise floor): 0.020. Preserve exact prompt and model digest for reuse. |
| Final question protocol | 400 questions (ExpertWritten + Simulated); four seeded 200/200 development/held-out splits | Seeds: 42, 7, 123, 2024. |
| Outlier rule | 60-second cleaned sensitivity | Pre-registered; both raw and cleaned figures reported in the thesis. |

## Judge prompt template

Exact prompt used by the deterministic, temperature-zero supportedness judge (`rejudge_temp0.py`). Model: llama3.2:1b, temperature 0, fixed seed 42, context capped at 8,000 characters per query.

```
You are checking whether an ANSWER is supported by the given CONTEXT.

CONTEXT:
{context}

ANSWER:
{answer}

Is the ANSWER supported by the CONTEXT above? Reply with exactly one word: YES or NO.
```

## Generator prompt template

Exact prompt used for both the static baseline and the adaptive Fast/Deep generator. Only the amount of retrieved context in the CONTEXT block changes between conditions; instructions, structure, and decoding model are held fixed.

```
You are a helpful support assistant. Answer the question using only the context below. If the answer is not in the context, say "I don't know."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
```
