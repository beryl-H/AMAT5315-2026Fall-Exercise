# md CLI Design: LJ Fluid Run, Check, Video

Date: 2026-09-09
Status: approved by user in brainstorming, pending spec review

## Purpose

Turn the `md` crate in `week2/` into a command-line tool for molecular dynamics
of a Lennard-Jones fluid in 2D:

- `md run` — simulate and save a trajectory,
- `md check` — verify the physics of a saved trajectory,
- `md video` — render an MP4 of the motion with the radial distribution
  function building up beside it.

The crate stays std-only: hand-rolled argument parsing, PPM frame rendering,
and a tiny in-crate RNG. ffmpeg muxes frames into MP4.

## Physical system

- 100 atoms in 2D, reduced units (`m = sigma = epsilon = kB = 1`).
- 10 x 10 triangular lattice start: rows offset by half a spacing, nearest-
  neighbor distance chosen so the density is exactly `rho = 0.8`.
- Square periodic box, `L = sqrt(N/rho) = sqrt(125) ~ 11.18`; minimum-image
  convention in the pair loop.
- Shifted-force Lennard-Jones potential with cutoff `rc = 2.5`:
  - `V(r) = U(r) - U(rc) - U'(rc) (r - rc)` for `r < rc`, else `0`
  - `F(r) = F(r) - F(rc)` for `r < rc`, else `0`
  where `U` and `F` are the existing full-range Lennard-Jones functions.
  Energy and force are continuous at `rc`.
- The existing free `energy`/`force` functions and the dimer (open boundary)
  behavior stay untouched; the cutoff versions are additional tested functions.

## CLI

```text
md run   [--temp 1.0] [--dt 0.005] [--steps 10000] [--equil 1000]
         [--out trajectory.txt] [--seed 1]
md check trajectory.txt  [--temp-tol 0.05] [--drift-tol 1e-3] [--ks-tol 0.05]
md video trajectory.txt  [--out video.mp4] [--fps 30]
```

- `run`: initial velocities Maxwell-Boltzmann at `--temp` (2D, from the
  in-crate RNG seeded by `--seed`); `--equil` steps of velocity rescaling to
  the target temperature; then `--steps` recorded NVE steps written to
  `--out`.
- `check`: recomputes all physics from the raw frames of the file; prints one
  line per check and a final `verdict: PASS|FAIL`; exit code 0 on PASS, 1 on
  FAIL.
- `video`: subsamples the trajectory to `--fps`, renders PPM frames, muxes
  with ffmpeg into `--out`.

Defaults are starting points; all are overridable on the command line.

## Trajectory file format

```text
# md trajectory v1
# N 100
# box 11.1803
# temp 1.0
# dt 0.005
# frames 10000
x0 y0 vx0 vy0
x1 y1 vx1 vy1
...            (N lines per frame, wrapped coordinates)
```

Comment header lines start with `#`; the frame count is part of the header;
frames follow one after another, one line per atom with wrapped coordinates.

## Physics checks (`md check`)

All recomputed from the saved frames (nothing is taken on trust from `run`):

- Temperature: per frame `T = sum_i |v_i|^2 / (2N)` (2D); the mean over frames
  must be within `--temp-tol` (fraction, default 5%) of the target temperature.
- Energy drift: total energy per frame from saved positions (shifted-force
  pair sum) and velocities; the least-squares slope of `E(t)` times the run
  duration, relative to `|E0|`, must be below `--drift-tol` (default `1e-3`).
  This measures trend, not the bounded velocity-Verlet oscillation.
- Speed distribution: empirical speeds vs the 2D Maxwell-Boltzmann
  distribution `f(v) = (v/T) exp(-v^2/(2T))` at the run's mean temperature;
  binned maximum deviation must be below `--ks-tol`.

`md run`'s own output must pass `md check`; that round trip is the acceptance
test.

## Video (`md video`)

- Subsample factor `ceil(frames / (--fps * 10))` for a clip of about ten
  seconds (e.g. 10000 frames at 30 fps samples every 34th frame).
- Left panel: the periodic box with atoms as filled circles (images not drawn).
- Right panel: `g(r)` recomputed from all frames up to the current one, on
  fixed axes `0..L/2` with a round y ceiling, so the curve visibly builds up.
- RDF definition (2D): `g(r) = <pairs in [r, r+dr)> / (N rho (pi((r+dr)^2 - r^2))/2)`,
  bin width `dr = 0.05`, out to `L/2`, averaged over frames seen so far.
- Rendering: PPM pixels drawn in Rust (circles and polylines, no fonts),
  muxed to MP4 by ffmpeg.

## Testing

- Unit tests: lattice geometry (count, spacing, density), minimum-image wrap,
  shifted-force continuity at `rc` and zero beyond, RNG determinism (same seed
  same velocities), temperature/energy recomputation from a written file
  round trip, trajectory parser, RDF normalization on a known lattice,
  CLI parsing.
- Integration: `run` then `check` passes; `check` rejects a doctored file
  (negative control); `video` produces a nonempty MP4 from a short trajectory.
- The existing dimer and pair-potential tests stay untouched and green.

## Out of scope

- Thermostatted production runs (rescaling is equilibration-only).
- 3D, mixtures, long-range corrections, pressure coupling.
- Restart files, multiple trajectories, parallelism.
