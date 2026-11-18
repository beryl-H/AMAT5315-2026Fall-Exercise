# Week 11 · Hard problems, reductions, and tensor-network sampling

The Week 11 learning sheet is the specification. Work in the order below, with seed 2026 for the required runs. Save raw JSON under `runs/` and generate each figure with Python from those files. The repository carries source, tests, design, plots' scripts, and reproduction instructions. The generated data and figures stay local.

1. Use `problemreductions-cli` 0.6.0 to reduce factoring to CircuitSAT and then SpinGlass. Solve and decode 15 at widths 2 and 3, then 6 and 7 at widths 2 and 2. Check each decoded product independently.
2. Enumerate every row of AND, OR, NOT, COPY, SET0, SET1 and XOR in the spin basis. Minimize over the XOR helper. The valid rows reach the stated gate floors, and every other row is at least 2 above. Enumerate the 16 shared-wire NAND-with-output-pin states. Exactly three have energy −5. Generate `gates.png`.
3. Plan with OMECO greedy and TreeSA, contract a minimum-energy/count semiring and a numerically rescaled Boltzmann measure, and sample downward using stored tensors. Check the 10×10 ferromagnet (−180, 2), factor 15 (−299, 1, log Z ≈ 598.3179, β=2), all 2,000 sample energies, the decoded product and the binomial expectation. Plot both planner widths for k=2–8 and the energy histogram in `factoring.png`.
4. Count N-queens boards via nonzero carry-network entries for N=4–10 (2, 10, 4, 40, 92, 352, 724). Draw 200 boards each for N=6 and 8 using conditional completion weights from the stored contraction, verify every board, and report frequencies and coverage in `nqueens.png`.
