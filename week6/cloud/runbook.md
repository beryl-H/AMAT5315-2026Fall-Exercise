# Week 6 cloud runbook — MNIST 30000-update training on Alibaba Cloud (A10)

**Status: approval packet only. No resource has been created, started, stopped
or deleted by this task, and no cloud API has been executed. Every command that
changes cloud state is marked `[GATED]`. The actual price is UNKNOWN until the
human obtains a live quote; nothing in this file is a price claim.**

Target: `ecs.gn7i-c8g1.2xlarge` (8 vCPU, **30 GiB** memory, 1x NVIDIA A10
24 GB), pay-as-you-go, region Singapore `ap-southeast-1`, Ubuntu public image,
SSH-key login only, security group open on TCP 22 to the human's current SSH
egress IPv4 `/32` only.

Local preconditions already satisfied: the 200-update / 4-sample smoke train
and sample finished on CPU with logs and independent verification
(`scripts/verify_smoke.py`, all checks passed). The formal run needs 30000
updates, batch 64, seed 2026; the smoke checkpoint is not a formal model.

There are two equivalent paths: the **manual console sequence** (section 2,
recommended — no CLI or account configuration is assumed) and the **CLI path**
(sections 3-5, all gated). The human performs every purchase/release action.

## 1. One-time local preparation (no cloud access needed)

```bash
# dedicated SSH key; NEVER overwrite an existing key file
test -f ~/.ssh/week6_ecs && echo "key already exists: reuse it, do not regenerate" \
  || ssh-keygen -t rsa -b 4096 -f ~/.ssh/week6_ecs -C "amat5315-week6-ecs"
# the public half (~/.ssh/week6_ecs.pub) is uploaded; the private half never
# leaves this machine and is never committed, printed or embedded anywhere.
```

Deployment uses the **current locally tested source, not GitHub HEAD** (this
work is not pushed). Build the source archive from local git HEAD; `git
archive` only packs tracked files, so `.venv/`, `runs/`, `artifacts/`,
`inputs/` and any untracked secrets are excluded automatically:

```bash
cd <repo-root>
git archive --format=tar.gz -o /tmp/week6-src.tar.gz HEAD week6
tar -tzf /tmp/week6-src.tar.gz | less   # verify: week6/ tracked files only,
                                        # no inputs/, runs/, artifacts/, .venv/
sha256sum /tmp/week6-src.tar.gz         # record for the upload check
sha256sum week6/inputs/mnist.npz
# expected mnist.npz SHA256 (official file downloaded 2026-10-14):
# 731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1
```

## 2. Manual ECS console sequence (recommended; assume no CLI configured)

1. Log in to the Alibaba Cloud console (the human's account; Week 5/6 signup
   and payment verification are the human's own steps).
2. ECS console -> region **Singapore (ap-southeast-1)** -> Create Instance.
3. Billing: **Pay-as-you-go** (not subscription, not spot/preemptible).
4. Instance type: `ecs.gn7i-c8g1.2xlarge` — confirm the page shows 8 vCPU,
   **30 GiB**, 1x NVIDIA A10 (24 GB).
5. Zone: pick a Singapore zone with stock for this type. If out of stock,
   change zone within Singapore only; do not change region or GPU family.
6. Image: free Ubuntu public image, Ubuntu 22.04 or 24.04 LTS; if the page
   offers "install GPU driver automatically" for GPU instances, select it
   (driver must satisfy JAX: CUDA 12 needs Linux driver >= 525; verify with
   `nvidia-smi` later). Do not buy a paid image.
7. System disk: **60 GiB** (category as offered, e.g. ESSD; the category and
   size enter the price). No data disk.
8. Network: the default VPC/VSwitch of the region is fine; public IP with
   **pay-by-traffic** and an explicit bandwidth cap (e.g. 10 Mbps peak). Do
   not add EIPs, SLBs, or snapshot plans that are not needed.
9. Login: **SSH key** — upload the PUBLIC key `~/.ssh/week6_ecs.pub` (paste or
   import). The private key stays local. Do not set a password.
10. Security group: allow inbound TCP 22 ONLY from the human's current SSH
    egress IPv4 as `/<YOUR_EGRESS_IP>/32`. If WSL/Windows goes through a
    proxy/VPN, confirm the real egress IP first. Never use 0.0.0.0/0. Update
    the rule if the IP changes.
11. Instance name suggestion: `amat5315-week6-mnist`.
12. On the confirmation page record: region/zone, instance type, image,
    system disk category + size, network billing, bandwidth cap, currency,
    hourly instance price, disk and network prices, and the query time. Give
    this summary to the human for approval. **Stop here — do not click
    purchase until the price is explicitly approved.**

## 3. Read-only CLI quotes (optional, after the human configures the CLI)

Configure interactively per the official guide
(https://github.com/aliyun/aliyun-cli/blob/master/docs/en/configuration.md):
run `aliyun configure` (OAuth preferred if set up; otherwise the AK flow via
interactive prompts). Keys live in `~/.aliyun`; never paste keys into shell
history, scripts, or this repository.

```bash
aliyun ecs DescribeZones --RegionId ap-southeast-1
aliyun ecs DescribeInstanceTypes --InstanceTypes '["ecs.gn7i-c8g1.2xlarge"]'
aliyun ecs DescribeAvailableResource --RegionId ap-southeast-1 \
  --DestinationResource InstanceType --InstanceType ecs.gn7i-c8g1.2xlarge \
  --InstanceChargeType PostPaid
# price: ECS DescribePrice (official reference:
# https://www.alibabacloud.com/help/en/ecs/developer-reference/api-ecs-2014-05-26-describeprice )
# Quote the exact configuration: instance type, ImageId, system disk
# category/size 60, region/zone, PriceUnit Hour / Period 1 / SpotStrategy
# NoSpot, InternetChargeType PayByTraffic, explicit bandwidth cap.
aliyun ecs DescribePrice --RegionId ap-southeast-1 --ZoneId <ZoneId> \
  --InstanceType ecs.gn7i-c8g1.2xlarge --ImageId <ImageId> \
  --PriceUnit Hour --Period 1 --SpotStrategy NoSpot \
  --SystemDiskCategory cloud_essd --SystemDiskSize 60 \
  --InternetChargeType PayByTraffic --InternetMaxBandwidthOut 10
# (confirm flag names with `aliyun ecs DescribePrice --help` before running)
```

**No price may be recorded anywhere unless it came from a live DescribePrice
call or the console confirmation page.**

## 4. Instance creation — `[GATED]` (only after price approval)

Console path: continue from section 2 step 12 after approval.

CLI equivalent (all gated):

```bash
# key pair: import the LOCAL public key (CreateKeyPair prints a private key
# to stdout, which we avoid); guard against name collisions first
aliyun ecs DescribeKeyPairs --RegionId ap-southeast-1 \
  --KeyPairNames '["week6-ecs"]'        # expect empty before importing
aliyun ecs ImportKeyPair --RegionId ap-southeast-1 --KeyPairName week6-ecs \
  --PublicKeyBody "$(cat ~/.ssh/week6_ecs.pub)"

# security group + SSH-only ingress from the human's egress IP
aliyun ecs CreateSecurityGroup --RegionId ap-southeast-1 \
  --SecurityGroupName week6-gn7i-sg
aliyun ecs AuthorizeSecurityGroup --RegionId ap-southeast-1 \
  --SecurityGroupId <SecurityGroupId> --IpProtocol tcp --PortRange 22/22 \
  --SourceCidrIp <YOUR_EGRESS_IP>/32

# creation: VSwitchId REQUIRED (same region/zone); system disk category/size
# must match the quoted configuration; SSH key only, no PasswordInherit
aliyun ecs RunInstances --RegionId ap-southeast-1 --ZoneId <ZoneId> \
  --InstanceType ecs.gn7i-c8g1.2xlarge --ImageId <ImageId> \
  --InstanceChargeType PostPaid --Amount 1 --InstanceName amat5315-week6-mnist \
  --VSwitchId <VSwitchId> --SecurityGroupId <SecurityGroupId> \
  --KeyPairName week6-ecs --SystemDiskCategory cloud_essd --SystemDiskSize 60 \
  --InternetChargeType PayByTraffic --InternetMaxBandwidthOut 10

# poll by the RETURNED InstanceIds (InstanceNames filtering is not supported)
aliyun ecs DescribeInstances --RegionId ap-southeast-1 \
  --InstanceIds '["<InstanceId>"]'
```

Record the exact instance ID, public IP, zone and creation time in
`cloud/README.md` immediately.

## 5. Setup, training, copy-back — `[GATED]` (instance running)

```bash
# connect; the login user (root or ubuntu) comes from the console connection
# instructions for the chosen image — do not assume one
export ECS_HOST=<PUBLIC_IP>          # from DescribeInstances
export ECS_USER=<login-user>         # per console instructions
export ECS_KEY=~/.ssh/week6_ecs
ssh -i "$ECS_KEY" "$ECS_USER@$ECS_HOST"   # verify host fingerprint

# --- on the instance: GPU first, environment second ---
nvidia-smi                            # must show A10 24GB and driver >= 525
tmux new -s week6                     # keep training observable
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"  # reload PATH after the installer
uv --version

# --- from this machine: upload local tested source and data separately ---
scp -i "$ECS_KEY" /tmp/week6-src.tar.gz "$ECS_USER@$ECS_HOST:~/"
ssh -i "$ECS_KEY" "$ECS_USER@$ECS_HOST" 'mkdir -p ~/week6/inputs && \
  tar -xzf ~/week6-src.tar.gz -C ~ --strip-components=1'
sha256sum /tmp/week6-src.tar.gz                       # local value
scp -i "$ECS_KEY" week6/inputs/mnist.npz "$ECS_USER@$ECS_HOST:~/week6/inputs/"
ssh -i "$ECS_KEY" "$ECS_USER@$ECS_HOST" 'sha256sum ~/week6/inputs/mnist.npz'
# both SHA256 values must match the section 1 value before continuing

# --- on the instance: environment and GPU assertion ---
cd ~/week6
uv sync --extra cuda
uv run python -c 'import jax; d=jax.devices(); print(d); \
  assert any(x.platform == "gpu" for x in d), "no GPU visible"'
# the assertion MUST pass; do not start the 30000-update run otherwise.
# save nvidia-smi output and the jax.devices() output to the log.

# --- exact formal training command (log to file inside tmux) ---
uv run diffusion train --data mnist --input inputs/mnist.npz \
  --updates 30000 --batch 64 --seed 2026 --out runs/train-mnist \
  2>&1 | tee runs/train-mnist-train.log
# record start/end time, GPU model, driver, final loss, wall time

# --- copy back into a NEW retrieval folder; never overwrite existing runs ---
# (from this machine)
RETRIEVE=retrievals/$(date +%Y%m%d-%H%M%S)-train-mnist   # new folder
mkdir -p "$RETRIEVE"
scp -i "$ECS_KEY" -r "$ECS_USER@$ECS_HOST:~/week6/runs/train-mnist" "$RETRIEVE/"
scp -i "$ECS_KEY" "$ECS_USER@$ECS_HOST:~/week6/runs/train-mnist-train.log" "$RETRIEVE/"

# verify identity and content BEFORE promoting anything
sha256sum "$RETRIEVE/train-mnist/checkpoint.npz" "$RETRIEVE/train-mnist/loss.jsonl"
ssh -i "$ECS_KEY" "$ECS_USER@$ECS_HOST" \
  'sha256sum ~/week6/runs/train-mnist/checkpoint.npz ~/week6/runs/train-mnist/loss.jsonl'
# the SHA256 pairs must match; loss.jsonl must hold exactly 30000 ordered,
# finite records (spot-check with scripts/verify_smoke.py-style checks or:
python - <<'PY'
import json, math
rows = [json.loads(l) for l in open("<RETRIEVE>/train-mnist/loss.jsonl")]
assert [r["step"] for r in rows] == list(range(1, 30001))
assert all(math.isfinite(r["loss"]) for r in rows)
print("30000 ordered finite loss records OK")
PY
# only after verification:
mkdir -p checkpoints && cp "$RETRIEVE/train-mnist/checkpoint.npz" checkpoints/mnist.npz
```

## 6. Release and confirmation — `[GATED]`, needs SEPARATE explicit human
confirmation of the EXACT instance ID and of permanent disk loss

Release permanently deletes the instance and its (release-with-instance)
disks; only do this after section 5's verified copy-back. Stopping is NOT
releasing: disks, EIPs or snapshots may still bill while stopped.

Console: ECS ->Instances -> select the exact instance ID -> Details ->
More -> Release Instance. Stop the instance first if the page requires it,
and disable release protection for that exact ID if enabled. Confirm the ID
character-by-character before releasing.

CLI equivalent (gated):

```bash
aliyun ecs StopInstance --RegionId ap-southeast-1 --InstanceId <InstanceId>
# wait until DescribeInstances shows the instance Stopped
aliyun ecs DescribeInstances --RegionId ap-southeast-1 \
  --InstanceIds '["<InstanceId>"]'
aliyun ecs DeleteInstance --RegionId ap-southeast-1 \
  --InstanceId <InstanceId> --Force false
# confirmation: the exact ID must no longer exist
aliyun ecs DescribeInstances --RegionId ap-southeast-1 \
  --InstanceIds '["<InstanceId>"]'      # expect no matching instance
# check for leftovers created for THIS run (disks/EIP/snapshots); record IDs
# and prices of anything found before deciding what to do with them
aliyun ecs DescribeDisks --RegionId ap-southeast-1
```

Reference: https://api.alibabacloud.com/document/Ecs/2014-05-26 (DeleteInstance
is the ECS release API; `ReleaseInstance` is not).

## 7. Sign-off

Update `cloud/README.md` only from live outputs: instance ID, creation and
release times, GPU/driver, training wall time, final loss, and the billed
cost. Bills can lag; write "bill pending" rather than 0 until the amount is
actually shown. Do not execute any `[GATED]` command without the human.

## Official references (checked 2026-10-14)

- A10/gn7i specs: https://www.alibabacloud.com/help/en/ecs/user-guide/gpu-accelerated-compute-optimized-and-vgpu-accelerated-instance-families-1
- GPU driver auto-install: https://help.aliyun.com/zh/egs/automatically-install-or-load-tesla-drivers-via-imgaes
- Pay-as-you-go and release billing: https://help.aliyun.com/zh/ecs/pay-as-you-go-1
- JAX installation and CUDA 12 driver requirement (>= 525): https://docs.jax.dev/en/latest/installation.html
- DescribePrice API: https://www.alibabacloud.com/help/en/ecs/developer-reference/api-ecs-2014-05-26-describeprice
- ECS API (release = DeleteInstance): https://api.alibabacloud.com/document/Ecs/2014-05-26
- aliyun CLI configuration: https://github.com/aliyun/aliyun-cli/blob/master/docs/en/configuration.md
- Chinese original of this operating guide (human-provided):
  `/mnt/d/Keria/Course/AMAT5315/week6-support/阿里云A10租用操作.md`