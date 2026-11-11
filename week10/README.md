# Week 10: Approximate Ising tensor contractions

Rust f64 SVD, open-boundary MPS and periodic TRG, following the 15-page learning sheet. The required scans and independent checks have been executed locally. [中文学习与汇报指南](week10-学习指南.md) explains the physical result and one update of each algorithm.

The two geometries approach the same bulk transition from opposite sides. At χ=32, extrapolation of the specific-heat peak gives:

| Geometry and sizes | Tc, all sizes | Error vs Onsager | Tc, largest two | Error vs Onsager |
|---|---:|---:|---:|---:|
| Open MPS, L=16,32,64 | 2.267506090 | 0.0740012% | 2.268764599 | 0.0185403% |
| Periodic TRG, L=64,256,1024 | 2.269204722 | 0.0008553% | 2.269263142 | 0.0034298% |

Onsager Tc=2/log(1+sqrt(2))=2.269185314. All four fits meet the sheet's 1% target. These are finite-range extrapolations, not statistical confidence intervals. The difference between fit windows, and rank/step changes below, quantify the tested numerical sensitivity.

## Environment and reproduction

Executed on Ubuntu24.04/WSL2, AMD Ryzen7 8745H (16 logical CPUs), rustc/cargo1.96.0, Python3.11.17, NumPy2.2.6, SciPy1.17.1, Matplotlib3.10.3. Native WSL pi-agent0.87.1 with MiniMax-M3.1-Flash assisted the TRG prototype; the coordinator corrected its old/retained dimension mapping and round count, then independently tested and replayed the results. No credentials or agent session transcripts are part of this public directory.

The executed backend is SciPy's LP64 OpenBLAS (`scipy_dgesdd_` and `scipy_dgemm_`), one BLAS thread per process. Two independent CPU processes are used for grids. All tensor entries and scales use f64. Release opt-level3. Cargo.lock fixes Rust dependencies. BLAS implementations may change gauge/sign choices and last-bit results while preserving the tolerances.

Default Linux builds link system `liblapack.so.3` and `libblas.so.3` (Ubuntu packages `liblapack3 libblas3`). An optional SciPy-wheel backend is selected at build time using THERMO_OPENBLAS:

```bash
python -m pip install -r requirements.txt
# Optional fast backend, if SciPy is installed in this Python environment:
export THERMO_OPENBLAS="$(python -c 'import pathlib,scipy; print(next((pathlib.Path(scipy.__file__).parent.parent/"scipy.libs").glob("libscipy_openblas*.so")))')"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cargo test --locked
cargo fmt --check
cargo install --locked --path .
thermo --method boundary --l 64 --k 0.45 --chi 16 --out runs/boundary64
cargo install --locked --path . --force
thermo --method trg --l 1024 --k 0.44 --chi 16 --out runs/trg1024
python study.py --bin "$(command -v thermo)" --phase all --workers 2
python verify_and_plot.py --bin "$(command -v thermo)" --phase all
```

Run commands from the crate directory. Install SciPy separately to use that optional backend; the executed environment used SciPy1.17.1. The system backend avoids a SciPy dependency and is slower. The tool refuses to overwrite any existing output directory. The study script resumes complete runs and stops on incomplete runs. To recompute from scratch, copy source into a **new directory** and run `bash reproduce.sh`. No cleanup/deletion command is required. Existing complete datasets can be rechecked using `python verify_and_plot.py --bin thermo --phase all`.

`solver_seconds` measures the solver call including full-precision evidence serialization; it excludes compilation and process startup. The two numerical example commands were also executed from the installed binary in a fresh local `reproductions/installed-cli/` namespace and agree with the scan executable.

## Part 1: What Gaussian SVD compression saves

The matrix uses `rand_chacha0.9 ChaCha8Rng::seed_from_u64(2026)` and paired Box–Muller transforms of independent uniforms. The Rust run saves the 512×512 matrix, full thin U/VT, singular spectrum and five full reconstructed matrices as little-endian f64 in C order. NumPy reads the saved arrays and independently evaluates the norms.

| Rank | Measured relative Frobenius error | Two-factor/original storage |
|---:|---:|---:|
|16|0.9419574062|0.0625|
|64|0.7892022510|0.25|
|128|0.6135825839|0.5|
|256|0.3251653241|1.0|
|512|2.94e-15|2.0|

All five measured errors agree with the independently recomputed spectral-tail prediction within1e-8. First rank at ≤10% error: **395**, requiring **1.54296875×** the original number of stored entries in A and B. This Gaussian matrix therefore has no storage-saving two-factor approximation at that accuracy. The quoted count excludes the optional full reconstructions retained solely for verification. `artifacts/svd.png` marks both thresholds. SVD itself took0.079s in this run.

## Parts 2–3: Heat-peak refinement

The broad grid is K=0.40,…,0.50 inclusive, h=.0025, χ16 (41 points per size). Nine points centered at the nearest coarse peak-grid point use h=.00125 and χ16/32. Curvature is computed in **K**, then each three-point parabolic vertex is converted to T=1/K. The wider-step χ32 result uses every other refined sample, with no interpolation or extra solve. Each geometry is fitted separately to Tpeak=Tc+a/L.

| Geometry | L | Tpeak χ16, refined | Tpeak χ32, refined | Tpeak χ32, wider h | Rank shift | Step shift | cV peak χ32 |
|---|---:|---:|---:|---:|---:|---:|---:|
|Open|16|2.093104911|2.093104911|2.093074661|2.10982e-09%|0.00144523%|1.115448|
|Open|32|2.179676209|2.179676246|2.179655128|1.70288e-06%|0.000968872%|1.387532|
|Open|64|2.224219480|2.224220423|2.224194294|4.23708e-05%|0.00117473%|1.683394|
|Periodic|64|2.283843868|2.281601855|2.281598316|0.0982649%|0.00015512%|2.251392|
|Periodic|256|2.272536962|2.272260190|2.271899889|0.0121805%|0.0158565%|2.903819|
|Periodic|1024|2.270812696|2.270012404|2.270708648|0.0352549%|0.0306714%|3.162089|

The exact machine-readable numbers are in `artifacts/boundary-peaks.csv` and `artifacts/trg-peaks.csv`; use them for arithmetic. All rank and step peak shifts are below1%. Peak heights grow in both geometries. This study tests peak positions; the optional precise logarithmic peak-height extension was not performed.

Each geometry has162 unique scan/refinement points (coarse/refined χ16 overlaps are reused). Summed measured solver time, including raw-evidence I/O, is94.45s for boundary MPS and608.06s for TRG. Parallel elapsed time is smaller. Every g(K), setting, timing and relative output path is retained in `artifacts/{boundary,trg}-g.csv`. Local figures: `artifacts/boundary-critical.png` and `artifacts/trg-critical.png`.

## Small-system and saved-evidence verification

| Check | Absolute error | Required |
|---|---:|---:|
| Open4×4 g, K=0,.44,.7, χ8 | max2.22e-16 | <1e-8 |
| Periodic2×2 g, K=.44, χ16 | 0 | <1e-8 |
| Periodic4×4 g, K=.44, χ8 | 1.05639e-4 | <2e-3 |
| Open4 cV: finite difference vs variance | 3.83959e-6 | <1e-3 |
| Periodic2 cV: finite difference vs variance | 3.70469e-6 | <1e-3 |

The variance baselines are cV=.4393969122 (open4) and .4042782328 (periodic2). Python independently enumerates all configurations and counts one positive-direction bond per site, retaining distinct parallel bonds on the L2 torus. Rust has six independent specification tests, including zero K and the analytic L2 parallel-bond partition sum. All six pass in debug and release. CLI checks reject bad L, non-power-of-two periodic L, NaN K, χ0, omitted χ, and existing output.

For all18 largest-size refined runs per geometry, independent Python closes saved final tensors and restores scales. Boundary checks also validate each left-isometric core, last-core unit norm and all discarded-tail records. For one representative largest-size run at each rank, Python replays every TRG round from the saved S1–S4 factors, checks both split errors against the discarded spectra and compares the final tensor. All comparisons pass. Separate closed-form parabolic vertices and linear regressions reproduce every peak and both intercepts within1e-9 and1e-12 respectively.

## Files and evidence conventions

`src/linalg.rs`: explicit C/Fortran layout conversion around thin LAPACK SVD and row-major GEMM. `src/boundary.rs`: exact right sweep, truncated left sweep, one last-core scale per row. `src/trg.rs`: two ordered diagonal splits, GEMM ring contraction, 2log2L rounds, Nlogc scale accounting and independent opposite-leg closure. `src/model.rs`: site tensor and direct spin enumeration. `src/main.rs`: explicit CLI and Gaussian experiment.

Each run's final `run.json` is the completion marker. Boundary `boundary-evidence.json` includes all final MPS cores, edge vector, scales and each truncation's complete spectrum. TRG `step-XXXX.json` includes scales, spectra and four factors with C-order shapes; `trg-evidence.json` includes final tensor and scale summaries. `artifacts/verification-all.json` records independent outcomes; `artifacts/manifest.json` binds local source/raw evidence by SHA256. Raw data, plots, tensors and archives are ignored by Git; only source/design/scripts/docs are published. Reproduction recreates them locally.

Sources: supplied Week10 learning sheet; [specified TRG source revision](https://github.com/GiggleLiu/BackwardsLinalg.jl/blob/b16273a2d695fef31334085e40f9e0341a1ec0f0/examples/trg.jl). The port replaces the sample's all-four-equal closure with sum_ij A[i,j,i,j] and uses max-absolute-entry normalization as required by the sheet.
