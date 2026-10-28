## Verdict

**No concrete implementation defect explaining the slope is demonstrated by the supplied code and numbers.** The outer-product path appears consistent with the frozen protocol:

- `memory_sweep.py:39-43`: one seeded RNG generates the full key array, then the full value array; both are normalized, and smaller cases use prefixes.
- `memory_sweep.py:52-54`: computes \(F=K(K^\top V)\), including the mathematically unit self-coefficient, and the specified ISR.
- `memory_sweep.py:74-80,111-121`: takes the query-key median first and then the three-draw median.
- `memory_sweep.py:89-95`: selects the first \(R=1\) crossing with log-log interpolation.
- `audit_memory.py:16-26`: reconstructs the curves and fits the slope over all three dimensions, unweighted.

The aggregate capacity need not equal the median of the three per-draw capacities because “median before crossing” and “crossing before median” do not commute. Thus the \(d=32\) aggregate capacity \(34.93582294\), versus per-draw values \(31.21198613,36.60567467,34.46924278\), is **not evidence of a bug**.

## Measured interpretation

The capacities are:

- \(d=16:\ 20.00949307\)
- \(d=32:\ 34.93582294\)
- \(d=64:\ 66.95665554\)

The adjacent-dimension log exponents are approximately \(0.804\) and \(0.939\), while the prescribed all-three-point OLS slope is \(0.8712714249504045\). It therefore fails the required \([0.9,1.1]\); the \(d=64\) capacity passes its separate gate. Under the frozen estimator, the scientific failure remains.

The supplied direct-sum check covers all three seeds at the six aggregate-crossing endpoints and agrees within \(10^{-12}\). This strongly supports arithmetic correctness around the crossings, although it is not whole-sweep validation.

## Per-draw variability

Seed slopes are:

- 101: \(1.11902133\)
- 102: \(0.86004532\)
- 103: \(0.85236596\)

This establishes **substantial finite-draw variability**, especially at \(d=16\), whose capacities range from \(14.43363040\) to \(20.54084510\). It does **not** establish a code defect, prove the population slope differs from one, or justify replacing the prescribed aggregate with a favorable seed or alternative estimator. With only three draws, tail probabilities and uncertainty are not reliably characterized; sampling/interpolation sensitivity is plausible but not separately proven.

## Minimal existing-data diagnostics

1. Reconstruct each seed’s outer curve and crossing solely from `progress.jsonl`, reconciling the three supplied per-draw capacities.
2. Recompute the prescribed per-\(M\) medians, aggregate crossing, and three-point OLS slope from those saved records.

No new draws or full sweep are needed. Assumption: the quoted “zero diagonal” direct sum restores the own term \(V\); without that addition, it tests a different estimator. I did not execute code.
