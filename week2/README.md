# Week 2 Part 5: Performance measurement

## Timing

Timed on this machine (median and min–max over repeated runs) for the same
default-equivalent workload.

| Program | Median (s) | Range: min–max (s) |
| --- | ---: | ---: |
| NumPy `week2-sim.py` | 7.018 | 6.928–7.090 |
| Rust debug | 3.757 | 3.732–3.787 |
| Rust release | 0.619 | 0.613–0.621 |

Observations (medians):

- Release Rust is about **6.07× faster** than debug Rust (3.757 / 0.619).
- The release median (0.619 s) is below one third of the debug median
  (3.757 / 3 ≈ 1.25 s).
- On this machine release Rust is about **11.34× faster** than the NumPy
  baseline (7.018 / 0.619).

## Profile

| Version | Force share (%) | Elapsed time (s) |
| --- | ---: | ---: |
| Naive | 97.0 | 0.635 |
| Cell list | … | … |

The naive profile was recorded with:

```
samply record md run --n 400 --eq-steps 200 --steps 1000 --out /tmp/md-prof
```

Hotspot: `md::fluid::fluid_accelerations`

Inclusive sample share: **97%**

![Naive profile flamegraph](profile-naive.png)
