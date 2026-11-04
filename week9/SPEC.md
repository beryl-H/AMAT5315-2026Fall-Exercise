# Week 9: Probabilistic inference by tensor contraction

Source: Jin-Guo Liu and Xi-Wei Pan, `week9-learning-sheet.pdf`, 15 pages, provided 2026-11-04. This specification covers the hands-on builds only.

Use Rust 1.96.0, f64 and CPU-only tenferro tensor/cpu/einsum crates, with omeco for greedy and seeded time-only TreeSA orders. Python makes figures and independently checks the saved evidence. UAI factors are unchanged and use ordered scopes with the last axis varying fastest. ASIA IDs 0..7 are A,T,S,L,B,E,X,D; state 0 is present/yes/positive.

## Required builds

1. Exact ASIA: enumerate and contract for prior, AD and ADX, all eight node marginals, T/L joint and evidence mass. Save full precision to `artifacts/{prior,AD,ADX}/run.json`. Check AB, tr(AB), and the three-spin Ising ring with the same Rust tensor library. Draw the AD T-query network in `evidence/graphs.svg`, with ordered factor axes, consistent smoking index, A/D evidence, and only T open. Plot `posteriors.png` and `joint.png`; label joint-minus-product as dependence.
2. Contraction order: save seeded random and omeco GreedyMethod plans and per-step work/largest-tensor costs; replay saved scalar-query plans; analyze structure without evaluating tensors. Keep labels needed outside each subtree or by output. Draw actual saved trees in `evidence/orders.png`. Order, plan and analyze are mutually exclusive.
3. BP: native WSL Pi-assisted implementation with uniform initial messages, synchronous updates, normalized proposals, excluded recipient message, evidence sliced into tables, damping 0.5, undamped residual tolerance 1e-10, limit 1000 sweeps. Save final beliefs and each sweep's beliefs/residual/update work at the same state, including returned state. Run nonuniform-tree exactness tests and plot ASIA residuals/errors. Preserve approximate errors even after convergence.
4. Promedus_14: download pinned UAI/evidence/PR/MAR, greedy and seeded time-only TreeSA plans, log10 evidence partition sum and state-ordered queries 59,102,139; retain scalar factors and intermediate log scales. BP uses the same settings and must report actual convergence/error, runtime and work. Compare both exact plans and published answers; plot costs and traces. BP computes all beliefs while exact queries only three variables, so compare workloads with this qualification.

## Acceptance

- ASIA masses: prior=1, AD=0.004501375, ADX=0.00098822675. All node/joint marginals normalized; maximum absolute marginal error and relative mass error against enumeration <1e-10. Independently inspect T/A scope and entries so two methods cannot share a transposition bug unnoticed.
- AB=[[4,4],[10,8]], trace=12, Ising Z=12.60256209895193. Absolute error <1e-12 against NumPy direct sums/enumeration.
- Promedus exact-plan agreement absolute tolerance 1e-10; PR comparison 1e-4; MAR comparison 1e-5. Reference log10Z=-7.806746304582435; state-1 marginals for 59,102,139 are .1975882582,.1497534323,.6175220184. Published rounded answers remain the authoritative downloaded comparison files.
- Different random/tie-breaking plans may have different costs. Neither greedy nor TreeSA is claimed globally optimal. Retain actual plans, settings and independently recounted costs.
- README must include evidence, exact reproduction commands, environment/data sources/settings. Commit after verified builds; push implementation and evidence as required by the learning sheet and authorized coursework scope. Never include credentials, private agent logs or unrelated weeks.

## Ownership

Exact/parser/plans/CLI/tests: Codex `rust_implementation` subagent. BP module and tree tests: native WSL Pi, volcengine-plan/glm-5-3-flash. Independent NumPy audit and figures: Codex `requirements_review` subagent. Integration, source manifests, visual review, commits and final readback: coordinator.

No existing data may be deleted. All prior weeks, installation failures and run artifacts are preserved. Input source URLs and SHA-256 are in `sources.json`.
