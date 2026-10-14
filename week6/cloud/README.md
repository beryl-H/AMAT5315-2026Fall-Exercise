# Week 6 cloud training record

**Status: this task created no cloud resources — nothing was provisioned,
started, stopped or deleted, and no cloud API was executed. No statement here
describes the account's global state; only this task's actions are recorded.**

The actual price is UNKNOWN: no live quote exists because no account access is
configured on the machine that prepared this packet. Obtain a quote via the
read-only `DescribePrice` command or the console confirmation page in
`cloud/runbook.md` (sections 2-3) before any approval. Bills can lag behind
usage; record "bill pending" rather than 0 until an amount is actually shown.

## Instance 1 — MNIST 30000-update training (planned, gated)

| field | value |
|---|---|
| status | NOT_STARTED (this task created no resources) |
| region / zone | ap-southeast-1 / (zone pending live availability check) |
| instance type | ecs.gn7i-c8g1.2xlarge (8 vCPU, 30 GiB, 1x NVIDIA A10 24 GB) |
| instance ID | — (assigned at creation, gated) |
| creation time | — |
| release time | — (release = StopInstance + DeleteInstance by exact ID, gated, after verified copy-back) |
| GPU / driver | — (read from `nvidia-smi` on the instance; CUDA 12 needs driver >= 525) |
| training command | `uv run diffusion train --data mnist --input inputs/mnist.npz --updates 30000 --batch 64 --seed 2026 --out runs/train-mnist` |
| training wall time | — |
| final loss | — |
| billed cost | — UNKNOWN; no quote obtained yet ("bill pending" until a real amount exists) |
| release confirmation | — (`DescribeInstances` for the exact ID must return nothing) |