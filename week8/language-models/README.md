# Week 8 (revised): Language models and the transformer

Six-part worksheet from the revised 24-page learning sheet (snapshot SHA256
`4e33bea929459c01511efc0298308a93d08d99ee7df38372bbf52e89be69cc35`). Correctness
contract: `SPEC.md`; phase plan: `PLAN.md`; live status: `TASK_STATUS.md`.

Environment: Python 3.12 (uv), numpy 2.2.6, JAX 0.6.2 CPU, matplotlib 3.10.3,
sentencepiece 0.2.2, safetensors 0.8.0, scipy 1.17.1.

```bash
uv sync
uv run pytest tests/ -q
```

Pinned inputs (full provenance in `data/MANIFEST.md`; files git-ignored):

- `minishlab/potion-base-8M` @ `bf8b056` — `model.safetensors`, `tokenizer.json`
- `roneneldan/TinyStories` @ `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` —
  `TinyStoriesV2-GPT4-valid.txt`
- `karpathy/tinyllamas` @ `0bd21da` — `stories260K/tok512.model`
- `epoch-research/analyzing-chinchilla` @ `92258837425e1b5f2851d624287f0120583a3d0e`
  — `data/svg_extracted_data.csv`

Raw runs and checkpoints live in git-ignored `artifacts/`; evidence figures and
result JSON are committed under `evidence/`.

## Parts

| Part | topic | status |
|---|---|---|
| 1 | embeddings: relations as directions | input prep + module in progress |
| 2 | count models (Markov chains, n-grams) | input prep in progress |
| 3 | MLP as key–value memory | pending |
| 4 | attention: softmax store recalls 算 | pending |
| 5 | transformer: training and sampling | pending |
| 6 | Chinchilla scaling law refit | pending |

Statuses are updated as formal runs complete; every reported gate value is a
measured number.
