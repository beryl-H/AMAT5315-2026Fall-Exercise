# Week 6 cloud training record

**Status: formal training COMPLETE and verified; copy-back validated. Instance
release and the actual bill remain pending human actions. No cloud resource
was created, stopped or deleted by pi.**

Actual bill: UNKNOWN / pending — pi has not queried any billing API. The
human-provided screenshot-corrected quote (instance 18.065 CNY/h + disk
0.100 CNY/h + traffic 0.53 CNY/GB) is a quote, not a bill.

## Instance

| field | value |
|---|---|
| instance ID | `i-t4n7f2zqpa44cb3hh7uc` (confirmed by read-only instance metadata on the machine, not just the screenshot) |
| region / zone | ap-southeast-1 / `ap-southeast-1a` (metadata) |
| instance type | ecs.gn7i-c8g1.2xlarge (8 vCPU, 30 GiB, 1x NVIDIA A10 23028 MiB) |
| OS | Ubuntu 22.04.5 LTS |
| GPU / driver | NVIDIA A10 23028 MiB, driver 580.126.09 (nvidia-smi; CUDA 13.0 shown is the driver compatibility maximum, not an installed toolkit) |
| creation time | not exposed by instance metadata; per human screenshot the purchase was 2026-10-14 ~14:04 CST — recorded as approximate, not verified |
| release time | PENDING — human exact-ID confirmation required; pi will not release |
| billed cost | pending (see above) |

## Deployment (verified provenance)

- Source: local `git archive` of HEAD `36c56ee7e27e7db4335ab16b196f3f57e3551515`,
  archive SHA256 `144fb0f9601d58efdd12e834342884a36ed77cbda6d821e49de556512401f1a2`,
  identical remote vs local; 32 tracked files byte-for-byte, no
  `.venv/`/`inputs/`/`runs/`/credentials in the archive.
- Dataset: official `mnist.npz` uploaded separately, SHA256
  `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`,
  identical remote vs local.
- Environment: uv 0.13.1 (official installer), `uv sync --extra cuda` exit 0
  (completed 14:15:30+08:00), JAX 0.11.2, GPU assertion
  (`jax.devices()` shows `cuda:0`) passed before training; `nvidia-smi` and
  the assertion output are preserved in `gpu-check.log`.

## Training run (the ONLY valid run)

| field | value |
|---|---|
| run directory | `/root/week6/runs/train-mnist-a10-20261014-1421` (root's tmux session `mnist-verified`, flock `/root/week6/.formal-training.lock`) |
| command | `uv run diffusion train --data mnist --input inputs/mnist.npz --updates 30000 --batch 64 --seed 2026 --out runs/train-mnist-a10-20261014-1421` |
| start / end | 2026-10-14T14:21:00+08:00 / 2026-10-14T14:27:09+08:00 |
| wall time | 368 s (~84 steps/s measured at mid-run) |
| exit code | 0 |
| XLA memory fraction | unset (not needed; no OOM during the valid run) |
| final loss | 0.013941 (min 0.007980 at step 27973) |
| verification | 30000 ordered finite losses; exact settings (updates 30000, batch 64, seed 2026); 126 finite EMA arrays (`scripts/verify_formal_train.py`, all passed) |
| checkpoint SHA256 | `8b9b1a4704cf4fd1f8f4258c78e370db85ac0eca228cbc0bc03f2543e960f11d` (identical remote/local) |

## Incident record (honest audit)

A coordination race during initial setup started a second training process
while root had already launched one; root stopped both processes (PIDs 5003,
6522). Their mixed output is preserved un-deleted in the remote
`/root/week6/runs/train-mnist`. That run is **INVALID**: it was never
promoted, never used as evidence, and must not be. The lock script
(`start-formal-mnist-locked.sh` with flock) and the single dedicated run
directory above are the controls that prevent recurrence.

## Copy-back and promotion

Retrieved into NEW `week6/retrievals/20261014-1427-formal-mnist/` (run dir +
train log + gpu/start/end/exit/wall-seconds markers + `gpu-check.log` +
`sync.log` + remote SHA256 list). Remote/local SHA256 verified for all 10
files. `checkpoints/mnist.npz` was absent locally and was promoted from the
verified copy (same SHA256 as remote). No existing local run data was
overwritten.

## Release checklist (human)

1. Confirm no further GPU work is needed (pi has none: preview/trajectory
   sampling ran locally on CPU).
2. Confirm the exact ID `i-t4n7f2zqpa44cb3hh7uc`, stop if required, then
   DeleteInstance (permanent disk loss) per `cloud/runbook.md` section 6.
3. Confirm via `DescribeInstances` that the ID no longer exists; check for
   leftover disks/EIPs created for this run.
4. Record release time and, when it appears, the actual billed cost
   ("bill pending" until then).