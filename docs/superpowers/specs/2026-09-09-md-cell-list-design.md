# md Cell-List Force Design

Date: 2026-09-09
Status: approved by user in brainstorming, pending spec review

## Purpose

Speed up the Lennard-Jones pair force calculation in the `md` crate with a
cell (linked) list, keeping the original force loop behind `--force naive`
and making the cell list the default (`--force cells`). The same pair scan
serves both the integrator's accelerations (`md run`) and the per-frame
energy recomputation (`md check`).

## Context

`System::refresh_accelerations` and `System::pair_energy` each run a bare
`O(N^2)` pair loop over minimum-image distances with the shifted-force
cutoff `rc = 2.5`. Sampling showed ~2/3 of `md run` time spent in the force
loop; `md check` repeats the same O(N^2) energy loop for every recorded
frame.

## Interface

- `pub enum ForceStrategy { Naive, Cells }` in `system.rs`.
- `System::set_force(&mut self, s: ForceStrategy)`; the default is `Naive`.
- `--force naive|cells` flag on both `md run` and `md check`; default
  `cells`. Any other value is a usage error.
- The strategy lives on `System`; integrators keep calling
  `refresh_accelerations()` unchanged; `pair_energy` honors it as well.
- The dimer constructors (`System::new`, `dimer`) keep the naive default, so
  existing dimer behavior and tests are untouched.
- If the box is too small for a `2x2` cell grid, `Cells` falls back to the
  naive scan.

## Algorithm

One shared pair-scan primitive drives both force and energy:

```rust
impl System {
    // visits every pair (i, j, d, r) inside the cutoff once
    fn scan_pairs(&self, visit: impl FnMut(usize, usize, [f64; 2], f64));
}
```

- `Naive`: all pairs `i < j`, minimum-image displacement, skip `r >= rc`.
- `Cells`: square grid of cells of side `>= rc` over the periodic box,
  `nx = floor(L / rc)`. The position -> cell index is rebuilt on every scan
  (O(N)). Each cell pairs with itself (`i < j`) and with its 8 periodic
  neighbors whose cell id is higher, so every unordered pair is visited
  once; pairs with `r >= rc` are skipped.
- Out-of-cutoff pairs contribute exactly `0.0` under the shifted-force
  potential, so both strategies sum the same nonzero set. Results agree to
  round-off only (summation order differs).

## Verification

- Unit tests, all seeded/deterministic:
  - naive vs cells energies agree to `1e-12` relative on random periodic
    configurations, including pairs straddling the box boundary;
  - naive vs cells accelerations agree to `1e-12` relative on the same
    configurations;
  - a configuration with every pair beyond `rc` gives zero total force and
    energy under both strategies;
  - the open-boundary dimer still passes its existing conservation test with
    the naive default.
- Benchmark: same run (`-n 400` and `-n 1600`, same steps, same machine)
  under `--force naive` vs the cells default; report the wall-time ratio.
  `md check` on both outputs must PASS.

## Out of scope

- Neighbor-list rebuild amortization, Verlet lists, cutoff correction beyond
  the shifted-force scheme, parallelism, GPU.
- Changing the RDF accumulation in `md video`.
- Any change to the physics: cells produce the same forces as naive up to
  round-off.
