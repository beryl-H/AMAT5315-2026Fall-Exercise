# Week 5 seismic design

Authority: week5-learning-sheet.pdf, Parts 2-4, and seismic.design.toml.
Work in the user-selected existing week5 directory; preserve prior Part 1 history.

## Numerical contract
Complete state is [previous pressure, current pressure], two float64 grid fields.
Outer grid values are zero. For each interior cell, centered differences give
u_next = (2*u - (1-sigma*dt)*u_prev + dt^2*(c^2*laplacian(u)+q))/(1+sigma*dt).
Five-point Laplacian uses dx^2; sponge sigma=strength*max(0,1-edge_distance/width)^2.
At time n*dt, q is source_amplitude times a Ricker pulse times a unit-peak
Gaussian of one-cell standard deviation. Sample receivers AFTER each step.
Fields use [z,x], positions [x,z]. Float32 recordings are for visualization only.

## Architecture
Rust CLI, experiment validation, NPY serialization and storage scheduler stay in
ordinary std Rust. Only the physics kernel is a no_std static library built by
build.rs with -Zautodiff=Enable, using nightly-2026-09-05 with enzyme. Safe wrappers
validate slice lengths and keep disjoint owned buffers. The Born loop calls one
Enzyme timestep JVP per step with velocity perturbation active; source, damping
and initial state are constants. The reverse loop calls one Enzyme timestep VJP,
adds receiver weights to current-pressure output adjoints, and accumulates the
velocity shadow. Do not substitute handwritten derivatives in production.

Full history saves N+1 complete states, including state 0 and N. Treeverse uses
the exact binomial capacity and ceil split rule on lecture page 15. State 0 has a
permanent saved slot; requested budget counts EXTRA slots. grad takes its primal
from a saved slot. Replay changes only its workspace, never image or adjoints.
Each shot writes restore/call/store/grad/fetch actions with live saved-state count.
Marmousi is never run with full history. Saved-state bytes exclude workspace,
model, data, adjoints and Enzyme scratch; report that scope explicitly.

## Independent correctness
Forward gather L2: 11.574770, relative tolerance 1e-4. Absolute shot peaks:
0.60809514, 0.59271397, 0.60809514 at trace index 83, relative tolerance 1e-4.
Positive diagonal peaks at steps 108 and 144 must shift 1.296 km within 0.141422 km.
Born/adjoint dot-product relative mismatch <1e-9, with left side about 0.0348479.
Independent random-weight transpose check and centered velocity finite differences
prevent a shared JVP/VJP scaling error. Image row-norm peak in x=7..33,z=10..33
must be within one cell of z=21. Treeverse errors versus full image <1e-9.
Budgets 1,3,5,10: forward calls per shot 28680,1695,990,642 and peaks 2,4,6,11.
N=6,budget=2 schedule has 8 forward calls, 6 descending reverse calls, peak 3.
Marmousi image L2: 6.7037741e-4, relative tolerance 1e-4; peak 6 states,
20,788,320 saved-state bytes. Input SHA-256 is computed from exact file bytes.

## Evidence and reproducibility
Python scripts under scripts/ verify, audit and plot with matplotlib. JSON and PNG
files stay under artifacts/; each tracked evidence file is below 5 MB. inputs/,
all *.npy, target/ and the project .venv stay local and ignored. README maps every
evidence file to its producing command, identifies quantities and limitations,
and gives fixed dependency versions, download and rebuild commands. Preserve
spec -> failing test -> implementation order for new scientific components.
