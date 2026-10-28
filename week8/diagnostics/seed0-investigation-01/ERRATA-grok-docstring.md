# ERRATA — stale module docstring in frozen `week8/grok.py`

Status: documentation erratum only. `grok.py` is a frozen formal source
(sha256 `d0501c3baf88a1194665123e47abc0a5b0035d0b203a56ca0dcc92853bc1d447`)
and is NOT modified; this document records the discrepancy for reviewers.

## Discrepancy

The module docstring of `week8/grok.py` (lines 1–12) still describes the
**invalidated formal-01 symmetric variant**, not the code below it:

| Docstring text (stale) | Actual frozen code / PART4_DECODER_PROTOCOL.md |
|---|---|
| "no positional embeddings" | learned positional embeddings `pos` of shape (3, 64), initialized `normal(kpos, (3, WIDTH))`, added to token embeddings in `forward` and `attention_weights` (`x = emb[tokens] + pos[:T]`) |
| "no causal mask" | causal tril mask applied before softmax in both `forward` and `attention_weights` (`scores = where(tril, scores, -inf)`) |
| "Parameters: … = 62,305" | 62,497 total: 62,305 + 192 positional (`count_params` / protocol "62,497 = 62,305 + 192") |

## Why it is stale

`grok.py` was corrected from the invalidated symmetric variant to the
decoder structure (Part 2 minus LayerNorm, with positional embeddings and
causal mask) when the symmetry defect was proven
(`week8-support/part4-symmetry-defect.json`); the executable code and
`init_params` were updated at that time, but the module docstring was not
rewritten. The authoritative architecture description is
`week8/PART4_DECODER_PROTOCOL.md` ("Architecture (frozen)") and the code
itself; diag04 verified the saved initializations bitwise-match
`init_params(PRNGKey(seed))`, and diag03 verified all 12 formal runs share
one source-hash set — the trained artifacts correspond to the decoder
structure with positional embeddings and causal mask, i.e. the docstring is
wrong, not the model.

## Required action (deferred, not executed here)

A docstring-only correction to `grok.py` (no executable change) should be
applied in a future non-frozen revision, only when the coordinator authorizes
touching frozen sources and with a recorded re-hash. Until then this erratum
governs interpretation.
