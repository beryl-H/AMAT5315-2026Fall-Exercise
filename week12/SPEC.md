# AMAT5315 Week 12 — MILP and neural branching

The Week 12 learning sheet defines the required interface and acceptance criteria. Work proceeds in four builds in one native Pi conversation. Keep provider credentials and local experiment data outside the public repository. Commit each verified build in order and push the completed source to the course exercise repository.

## 1. LP relaxation, exact integer optima, and rounding

Implement `branchopt --mode gap` in Rust with `highs`. Use one HiGHS thread and disable presolve. Require optimal solver status and both MIP gaps set to zero for the independent MILP oracle.

- Workshop: LP value `83/3` at `(7/3, 8/3)`, integer value `27` at `(3, 2)`, rounded `(2, 3)` requiring `29 > 28` hours, floor `(2, 2)` worth `22`.
- Diet: continuous minimum `323/14` at `(89/28, 8/7)`.
- Use an absolute tolerance of `1e-4` for the analytical optima. For all 16 seeded 32-column, 24-row covering instances, check that the LP is a lower bound, the integer oracle plan covers every row, and the nearest-rounding repair covers every row and costs at least the oracle value.
- Save the problem matrices and cost vectors, the two optima, the rounded and repaired plans, and the per-instance gap and overhead. Generate the figure from the saved run.

## 2. Exact branch-and-bound and teacher labels

- Visit the workshop root, the `x2 <= 2` incumbent, and the `x2 >= 3` bound-pruned child in that order. Their bounds must be `83/3`, `27`, and `26.75`.
- For binary covering, start with no incumbent. Use a depth-first stack and visit the child matching the rounded LP value first. Only the LP statuses, bounds and a verified integer incumbent govern pruning. Branch on legal fractional columns only.
- Run both most-fractional and Strong Branching on the same 48 size-32×24 instances. Verify both against the HiGHS MILP oracle at absolute error below `1e-9`. Check the number of processed nodes and all LP solves, including both child probes for every Strong Branching candidate. Record LP and overall timings separately.
- Export the 12 features and both bound gains for each candidate, including the product score. Use a gain of `1e6` for an infeasible or bound-pruned child. Log only nodes with at least two legal candidates and at most 80 nodes per instance. Assign instances `0..35` to training and `36..47` to validation and verify no overlap.

## 3. Shared JAX model

- Train the shared float64 `12 → 48 tanh → 1` MLP on the CPU with pinned JAX `0.12.1` and NumPy. Standardize features using only real training candidates. Fit a masked softmax with uniform targets over best-score ties.
- Use Adam, 120 epochs, batch size 64, learning rate `0.004`, and L2 penalty `0.0002` on weights only. Keep the checkpoint with the lowest validation cross-entropy.
- Keep each instance in one split. Report validation Top-1 and Top-3 plus the uniform random success rate for the same definition of a correct pick. Export the full history and six held-out raw-feature probes with float64 logits and probabilities. Generate learning curves from the saved model.

## 4. Rust inference and controlled CPU evaluation

- Compare most-frac, strong, neural and root-neural on identical held-out instances from both suites. Use 16 of size 32×24 with seed 3026 and 8 of size 48×34 with seed 4026. Use the same trained model for both sizes.
- Online neural recomputes features and scores at each scored node. Root-neural standardizes the root scores of all variables and subsequently adds `3.0 × fractionality`. Both policies only select a legal candidate. Time LP solves and feature extraction plus inference separately using elapsed time on the stated CPU environment.
- For all six probes, compare both logits and probabilities to JAX at absolute error below `1e-4`. For all four policies and both suites, require optimum errors below `1e-4`. Verify one LP solve per node for policies without child lookahead and account for the extra LPs in Strong Branching.
- Report the early splits on the instance showing the largest absolute node saving for online neural. Plot node counts and CPU timing components from the saved runs. Treat the handout’s runtime and accuracy numbers as reference measurements and report the measured outcomes for this generator and environment.
