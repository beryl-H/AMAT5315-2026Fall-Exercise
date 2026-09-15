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
| Cell list | 92.0 | 0.208 |

Both profiles used the same course-form case:

```
samply record md run --n 400 --eq-steps 200 --steps 1000 --out /tmp/md-prof
```

- The profiled cell-list run is about **3.05× faster** than the naive
  profiled run (0.635 / 0.208).
- Force evaluation remains the dominant workload (cell-list force share
  92%).
- The cell-list elapsed time (0.208 s) is below the naive elapsed time
  (0.635 s).

Hotspot (cell list): `md::fluid::accelerations_cells`

![Naive profile flamegraph](profile-naive.png)
![Cell-list profile flamegraph](profile-cells.png)

## Benchmark (release, wall-clock; --eq-steps 100 --steps 500 = 600 steps/run)

| N | Naive median (range) | Cells median (range) | Speedup |
| ---: | ---: | ---: | ---: |
| 100 | 0.030 (0.028–0.040) | 0.028 (0.026–0.035) | 1.07 |
| 400 | 0.383 (0.370–0.404) | 0.113 (0.099–0.137) | 3.40 |
| 1600 | 4.823 (4.804–5.006) | 0.449 (0.443–0.472) | 10.75 |

Seconds per integration step (median / 600) are plotted against N on a
log–log scale:

![Cell-list scaling](scaling.png)

The slopes reflect the pair-search algorithms: naive considers O(N²)
candidate pairs, while at fixed density and cutoff the cell-list candidate
work approaches O(N).
