# Data artefacts (not tracked in git)

Three large generated files are deliberately excluded from this repository (see `.gitignore`):

| File | Size | Produced by |
|---|---|---|
| `wixqa_index.faiss` | ~35 MB | `src/01_build_index.py` |
| `chunk_embeddings.npy` | ~35 MB | `src/01_build_index.py` |
| `chunks.pkl` | ~16 MB | `src/01_build_index.py` |

These are fully reproducible by running `python src/01_build_index.py`, which downloads the WixQA corpus, chunks it, embeds it, and builds the index. They are not needed to review the code or read the results in `results/`.

If you want them versioned anyway (e.g. so collaborators don't have to rebuild the index), the cleanest option is **Git LFS**:

```bash
git lfs install
git lfs track "*.faiss" "*.npy" "*.pkl"
git add .gitattributes
```

Do this before adding the files themselves, or GitHub's plain 100 MB-per-file limit and repo-bloat warnings will apply retroactively.
