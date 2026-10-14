# Week 6 cloud training record

**Status: NOT_STARTED — no cloud resource exists, nothing has been created,
and no charges have been or can be incurred yet.**

The runbook (`cloud/runbook.md`) contains the gated commands. Price and zone
availability must come from the live read-only queries in its section 2 before
any approval; no price is quoted here because no account access exists on the
machine that prepared this packet.

## Instance 1 — MNIST 30000-update training (planned)

| field | value |
|---|---|
| status | NOT_STARTED |
| region / zone | ap-southeast-1 / (zone pending live DescribeZones) |
| instance type | ecs.gn7i-c8g1.2xlarge (1x NVIDIA A10 24 GB) |
| instance ID | — (assigned at creation, gated) |
| creation time | — |
| release time | — |
| GPU / driver | — (to be read from `nvidia-smi` on the instance) |
| training command | `uv run diffusion train --data mnist --input inputs/mnist.npz --updates 30000 --batch 64 --seed 2026 --out runs/train-mnist` |
| training wall time | — |
| final loss | — |
| billed cost | — (from the billing console after release; expected 0 while nothing exists) |
| release confirmation | — (`DescribeInstances` for the ID must return empty) |