# Part 4 formal-decoder-02 — per-main-seed evidence

Producing command: `uv run python make_part4_evidence.py` — strict-gated on manager COMPLETE + 12 hash-matched metrics.

| seed | memorize step | grok step | delay | final train acc | final test acc | final top-5 share | key5 | keep key | remove key | remove other (10 draws) | keep other (10 draws) | all gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 200 | null | null | 1.0000 | 0.0954 | 0.3964 | 35,31,43,22,24 | 0.1456 | 0.0112 | min 0.0844 / max 0.0935 | max 0.0137 | FAIL |
| 1 | 200 | 5000 | 25.00 | 1.0000 | 1.0000 | 0.9295 | 40,24,21,48,42 | 0.9997 | 0.0103 | min 1.0000 / max 1.0000 | max 0.0103 | PASS |
| 2 | 200 | 5500 | 27.50 | 1.0000 | 1.0000 | 0.8798 | 6,7,28,30,20 | 0.9998 | 0.0138 | min 1.0000 / max 1.0000 | max 0.0138 | PASS |
| 3 | 200 | 4500 | 22.50 | 1.0000 | 1.0000 | 0.9283 | 7,40,1,2,14 | 1.0000 | 0.0108 | min 1.0000 / max 1.0000 | max 0.0125 | PASS |
| 4 | 200 | 6700 | 33.50 | 1.0000 | 1.0000 | 0.9443 | 40,13,28,17,23 | 0.9991 | 0.0112 | min 1.0000 / max 1.0000 | max 0.0112 | PASS |

Gates: grok within 10000, delay > 10, final top-5 share > 0.8, keep key5 > 0.95, remove key5 < 0.05, remove random other5 > 0.95 in all 10 draws, keep only random other5 < 0.05 in all 10 draws. Failed gates stay failed; controls are observed results.
