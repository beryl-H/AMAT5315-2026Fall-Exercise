# Week 6 cloud training record

**Status: this task created no cloud resources — nothing was provisioned,
started, stopped or deleted, and no cloud API was executed. No statement here
describes the account's global state; only this task's actions are recorded.**

A human-provided console quote was received on 2026-10-14 (details below).
No authenticated CLI query was performed by this task. This is a quoted
configuration estimate, not an incurred bill or evidence of an instance being
created. Recheck the console price before purchase; bills can lag behind usage.

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

## Human-provided console quote (2026-10-14, before creation)

The screenshot shows Singapore, `ecs.gn7i-c8g1.2xlarge`, Ubuntu 22.04 64-bit,
ESSD AutoPL 60 GiB, pay-by-traffic with a 10 Mbps cap.

| item | screenshot amount | basis |
|---|---:|---|
| instance | CNY 18.065 | one hour |
| system disk | CNY 0.100 | one hour |
| network | CNY 1.060 | estimated 2 GB at CNY 0.53/GB |
| Ubuntu image | CNY 0 | image charge |
| displayed total | CNY 19.23 | one-hour scenario including the estimated traffic |

Inferred base rate: CNY 18.165/hour. A two-hour scenario with **2 GB total**
traffic is CNY 37.39 (18.165*2 + 0.53*2). This does not guarantee training
completion within two hours. Actual traffic and resource retention determine
the bill. The earlier CNY 23.18 screenshot used Windows Server and fixed
bandwidth, so it is superseded for this task.

The human is handling purchase. No instance ID, actual deployment, formal
training, release or bill has been confirmed yet.
