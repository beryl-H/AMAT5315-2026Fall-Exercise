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
| 1 | embeddings: relations as directions | inference DONE, all mandatory gates PASS |
| 2 | count models (Markov chains, n-grams) | code+driver ready (CONTROL_PART2_READY); formal-01 invalidated (multi-writer custody); formal run dispatched by root |
| 3 | MLP as key–value memory | sweep driver + smoke done (d16); full sweep reserved for root dispatch |
| 4 | attention: softmax store recalls 算 | formal 20-draw validation DONE, all mandatory gates PASS |
| 5 | transformer: training and sampling | pending |
| 6 | Chinchilla scaling law refit | fitter + smoke done (3/243 starts); full fit + bootstrap reserved for root dispatch |

Part 1 measured results (`evidence/part1-results.json`, seed protocol in
`SPEC.md`): table mean LOO cosines 0.466 / 0.549 / 0.433 (gate ≥ 0.3); control
seed-1730 means +0.030 / −0.015 / +0.009 (gate |·| ≤ 0.15); 20-shuffle means in
[−0.089, +0.085] (descriptive). Figure: `evidence/part1-relations.png`.
**Visual correction:** the first figure used unequal x/y scaling, so drawn
angles were not arccos(cosine). Re-rendered from the same saved numbers with
`set_aspect("equal")` into `evidence/part1-relations-equal-aspect.png`
(producing command: `uv run python src/render_part1_figure.py`); data and
gates unchanged. Pair numbering key —
country_capital: 1 iran→tehran, 2 turkey→ankara, 3 russia→moscow, 4 china→beijing,
5 france→paris, 6 sweden→stockholm, 7 japan→tokyo, 8 peru→lima, 9 kenya→nairobi;
male_female: 1 king→queen, 2 man→woman, 3 he→she, 4 brother→sister, 5 father→mother,
6 son→daughter, 7 uncle→aunt, 8 boy→girl, 9 husband→wife;
adj_comparative: 1 big→bigger, 2 small→smaller, 3 fast→faster, 4 dark→darker,
5 cold→colder, 6 old→older, 7 strong→stronger, 8 long→longer, 9 tall→taller.

Corpus input preparation verified: 24,867 train / 2,763 validation stories,
~2.05 characters per token, exactly 3 validation roundtrip exceptions.

Part 4 (`evidence/part4-results.json`, `evidence/part4-recall.png`, showcase
seed 531504 predeclared, 20-draw medians): outer R(1024) 3.89 (gate: within
15 % of 4); ReLU R(1024) 2.82 (gate: within 15 % of 2.83); ReLU² R(1024) 0.614
and R(4096) 1.189 (gate > 1; sheet references 0.60/1.2); softmax R < 0.056 at
every M (gate < 0.1, sheet reference 0.057 at 4096). Producing command:
`uv run python src/run_part4.py --attempt formal-01`.

Part 2/3/6 code is ready under exclusive-attempt drivers; formal Part 2 is
dispatched by the root (CONTROL_PART2_READY.json); the full memory sweep and
the 243-start fit + 400 bootstrap refits are likewise reserved for root
dispatch. Part 6 smoke (3 grid starts, real 240 runs): own mean log residual
−0.00066, printed parameters +0.0167, D*/N* at 5.76e23 ≈ 17.9 — smoke only,
not cited as the formal fit.
