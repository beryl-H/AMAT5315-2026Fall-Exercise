# Pinned inputs — provenance

All files live in git-ignored `week8/language-models/data/`. This document and
`evidence/input-manifest.json` are the published provenance record.

| # | item | repo @ pinned revision | resolved full SHA | file | bytes | SHA256 |
|---|---|---|---|---|---|---|
| 1 | embedding table | `minishlab/potion-base-8M` @ `bf8b056` | not resolved (HF API timed out within bounded 30 s request; pinned short revision kept) | `model.safetensors` | 30236760 | `f65d0f325faadc1e121c319e2faa41170d3fa07d8c89abd48ca5358d9a223de2` |
| 2 | tokenizer (Part 1) | `minishlab/potion-base-8M` @ `bf8b056` | not resolved (same) | `tokenizer.json` | 683666 | `e67e803f624fb4d67dea1c730d06e1067e1b14d830e2c2202569e3ef0f70bb50` |
| 3 | corpus | `roneneldan/TinyStories` @ `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` | `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` (API-confirmed) | `TinyStoriesV2-GPT4-valid.txt` | 22502601 | `6874bae9a4c1a4e7edcf0e53b86c17817e9cf881fc75ff2368da457b80c0585d` |
| 4 | BPE tokenizer | `karpathy/tinyllamas` @ `0bd21da` | not resolved (HF API timed out within bounded 30 s request; pinned short revision kept) | `tok512.model` | 7645 | `dfff07d929db979913f166ec94a6f5ecad4c70cfed8eb5c9cbe7e464455e46f5` |
| 5 | scaling runs | `epoch-research/analyzing-chinchilla` @ `92258837425e1b5f2851d624287f0120583a3d0e` | `92258837425e1b5f2851d624287f0120583a3d0e` (API-confirmed) | `chinchilla-runs.csv` | 23224 | `193478df4e90f040016431996c45682659306840ecdadf83cfb9cd5ca0aa1b30` |

Primary URLs (all public, no credentials):

1. `https://huggingface.co/minishlab/potion-base-8M/resolve/bf8b056/model.safetensors`
2. `https://huggingface.co/minishlab/potion-base-8M/resolve/bf8b056/tokenizer.json`
3. `https://huggingface.co/datasets/roneneldan/TinyStories/resolve/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/TinyStoriesV2-GPT4-valid.txt`
4. `https://huggingface.co/karpathy/tinyllamas/resolve/0bd21da/stories260K/tok512.model`
5. `https://raw.githubusercontent.com/epoch-research/analyzing-chinchilla/92258837425e1b5f2851d624287f0120583a3d0e/data/svg_extracted_data.csv`

## Schemas

- `model.safetensors`: single tensor `embeddings`, shape (29528, 256), float32
  (29528 × 256 = the sheet's potion-base-8M table).
- `tokenizer.json`: Hugging Face tokenizers JSON; `model.type = WordPiece`,
  vocab size 29528; WordPiece pre-tokenizer splits at spaces and drops them
  (consistent with sheet UNDERSTAND).
- `TinyStoriesV2-GPT4-valid.txt`: UTF-8 plain text, stories separated by
  `<|endoftext|>` lines; 27,630 separators in file.
- `tok512.model`: SentencePiece model, vocab 512.
- `chinchilla-runs.csv`: 245 data rows; columns `x, y, color, Model Size,
  Training FLOP, hex_color, loss` (N = `Model Size` [parameters], C = `Training
  FLOP` [FLOP], L = `loss` [nats/token]).

## Retrieval record

- Files 1–4: downloaded 2026-10-28 by native WSL `pi` (this task, provider
  `volcengine-plan` / `glm-5-3-flash`) with `curl -L`.
- File 5: first WSL attempt failed with **curl exit code 55** (network error
  during transfer; failure preserved here and in the session log — not
  treated as success). Retrieved by the root coordinator on Windows via
  Python `urllib` from the same pinned primary URL; receipt:
  `support/chinchilla-download-receipt.json` (SHA256 and byte size match this
  file exactly); copied byte-identical into `data/chinchilla-runs.csv`.
- Commit-resolution API lookups: bounded 30 s requests on 2026-10-28; datasets
  and GitHub endpoints resolved to the pinned full SHAs; the two HF model-repo
  endpoints did not respond in time and are recorded as unresolved (short
  pinned revisions), per protocol.

No credentials were used or stored. No full model framework (e.g., PyTorch)
was downloaded.
