# Week 6 cloud runbook — MNIST 30000-update training on Alibaba Cloud

**Status: packet prepared only. No resource has been created, rented, started or
released. Every command below that creates or releases a resource is marked
`[GATED]` and requires the human's explicit approval of the displayed price
first. Nothing here has been executed against any account.**

Instance target: `ecs.gn7i-c8g1.2xlarge` (1x NVIDIA A10, 24 GB), pay-as-you-go,
Ubuntu with NVIDIA driver, region Singapore `ap-southeast-1`, SSH access
restricted to the human's current public IP.

## 0. What was checked locally (pi, this host)

- No `aliyun` CLI is installed on this host and no `~/.aliyun` credentials
  directory exists, so **no live price or availability query could be run**.
  No price or availability claim in this packet comes from a live source.
- Week 5's execution log (`week5/docs/execution-log.md`) records that Week 6
  account signup and payment verification are the human's; no cloud
  registration existed at the end of Week 5.
- Fresh public documentation facts (no account needed):
  - aliyun CLI official repo releases: latest `v3.5.1`
    (https://github.com/aliyun/aliyun-cli/releases).
  - jax `0.11.2` publishes the `cuda12` pip extra (runtime wheels bundled;
    only the NVIDIA kernel driver + libcuda are needed on the host), so the
    repo's `uv sync --extra cuda` (`jax[cuda12]==0.11.2`) is the GPU path.
    This extra is already resolved in `week6/uv.lock`.

## 1. Prerequisites (human)

1. An Alibaba Cloud account with payment verification completed (Week 5/6
   signup is the human's; nothing is configured here).
2. Install the CLI (official binary release):

   ```bash
   curl -fL https://github.com/aliyun/aliyun-cli/releases/latest/download/aliyun-cli-linux-3.5.1-amd64.tgz -o /tmp/aliyun-cli.tgz
   tar -xzf /tmp/aliyun-cli.tgz -C ~/.local/bin aliyun && chmod +x ~/.local/bin/aliyun
   aliyun version   # expect 3.5.1 or newer
   ```

   (If the asset name changed, take the current `aliyun-cli-linux-*-amd64.tgz`
   from the releases page.)
3. Configure credentials **interactively**; keys stay in `~/.aliyun` and are
   never printed, copied or committed:

   ```bash
   aliyun configure set --region ap-southeast-1 --mode AK
   ```

4. Learn the human's current public IP for the SSH ingress rule (the IP is
   pasted into the command at run time; it is not stored in this repository):

   ```bash
   curl -s https://ifconfig.me   # human runs this; substitute <YOUR_PUBLIC_IP>
   ```

## 2. Read-only checks (safe to run; no resources created)

Run these and paste the outputs into the approval packet. Flag names can be
confirmed with `aliyun <service> <command> --help` before running.

```bash
# zones available in Singapore (pick one where gn7i has stock)
aliyun ecs DescribeZones --RegionId ap-southeast-1

# instance spec (confirm 1x A10 24GB, 8 vCPU, 29 GiB memory class)
aliyun ecs DescribeInstanceTypes --InstanceTypes '["ecs.gn7i-c8g1.2xlarge"]'

# availability of this exact type (set ZoneId from DescribeZones)
aliyun ecs DescribeAvailableResource --RegionId ap-southeast-1 \
  --DestinationResource InstanceType --InstanceType ecs.gn7i-c8g1.2xlarge \
  --InstanceChargeType PostPaid

# Ubuntu 22.04 system images (pick an ImageId; prefer one with a bundled GPU
# driver if listed, otherwise the driver is installed over SSH in step 5)
aliyun ecs DescribeImages --RegionId ap-southeast-1 --OSType linux \
  --ImageOwnerAlias system --ImageName "Ubuntu_22_04" --PageSize 50

# pay-as-you-go price quote (BSS OpenAPI; verify flags via --help first)
aliyun bssopenapi GetPayAsYouGoPrice --ProductCode ecs --ProductType ecs \
  --SubscriptionType PayAsYouGo \
  --RegionList '[{"RegionId":"ap-southeast-1"}]' --PriceModuleList '[]'
```

**No price may be quoted in this repository unless it came from that live
call.** Cross-check the returned unit price x hours against the console price
estimator before approving.

## 3. Security group and SSH key — `[GATED]`

```bash
# create an isolated security group
aliyun ecs CreateSecurityGroup --RegionId ap-southeast-1 \
  --SecurityGroupName week6-gn7i-sg --Description "week6 MNIST training"

# SSH ingress restricted to the human's current public IP ONLY
aliyun ecs AuthorizeSecurityGroup --RegionId ap-southeast-1 \
  --SecurityGroupId <SecurityGroupId> --IpProtocol tcp --PortRange 22/22 \
  --SourceCidrIp <YOUR_PUBLIC_IP>/32

# dedicated key pair for this run; save the private half outside git, chmod 600
aliyun ecs CreateKeyPair --RegionId ap-southeast-1 --KeyPairName week6-key
```

## 4. Instance creation — `[GATED]` (approve price first)

```bash
aliyun ecs RunInstances --RegionId ap-southeast-1 --ZoneId <ZoneId> \
  --InstanceType ecs.gn7i-c8g1.2xlarge --InstanceChargeType PostPaid \
  --Amount 1 --InstanceName week6-gn7i --ImageId <ImageId from step 2> \
  --SecurityGroupId <SecurityGroupId> --KeyPairName week6-key \
  --InternetChargeType PayByTraffic --InternetMaxBandwidthOut 10 \
  --PasswordInherit True

# read-only: get instance ID and public IP (repeat until InstanceStatus is Running)
aliyun ecs DescribeInstances --RegionId ap-southeast-1 \
  --InstanceNames week6-gn7i
```

Record the instance ID in `cloud/README.md` at creation time.

## 5. On-instance setup and training — `[GATED]`

```bash
ssh -i <week6-key.pem> ubuntu@<PUBLIC_IP_OF_INSTANCE>

# --- on the instance ---
curl -LsSf https://astral.sh/uv/install.sh | sh      # uv, official installer
exit
# upload source + data from this machine (inputs/ is 11 MB)
rsync -avP --exclude '.venv' --exclude 'runs' --exclude 'artifacts' \
  week6/ ubuntu@<PUBLIC_IP_OF_INSTANCE>:~/week6/
scp -i <week6-key.pem> week6/inputs/mnist.npz \
  ubuntu@<PUBLIC_IP_OF_INSTANCE>:~/week6/inputs/mnist.npz

ssh -i <week6-key.pem> ubuntu@<PUBLIC_IP_OF_INSTANCE>
cd ~/week6
uv sync --extra cuda                                 # GPU JAX wheels
uv run python -c "import jax; print(jax.devices())"  # MUST list a cuda device
# if no GPU shows: install/verify the NVIDIA driver (nvidia-smi) per Alibaba's
# "Install GPU drivers" guide before training; the cuda12 wheels only need the
# kernel driver + libcuda, not a CUDA toolkit.

# exact formal training command (30000 updates, batch 64, seed 2026)
uv run diffusion train --data mnist --input inputs/mnist.npz \
  --updates 30000 --batch 64 --seed 2026 --out runs/train-mnist
```

Record GPU model, driver version, training wall time and the final loss in
`cloud/README.md`.

## 6. Copy results back — `[GATED]`

```bash
mkdir -p week6/runs/train-mnist week6/checkpoints
scp -i <week6-key.pem> \
  ubuntu@<PUBLIC_IP_OF_INSTANCE>:~/week6/runs/train-mnist/checkpoint.npz \
  week6/runs/train-mnist/checkpoint.npz
scp -i <week6-key.pem> \
  ubuntu@<PUBLIC_IP_OF_INSTANCE>:~/week6/runs/train-mnist/loss.jsonl \
  week6/runs/train-mnist/loss.jsonl
cp week6/runs/train-mnist/checkpoint.npz week6/checkpoints/mnist.npz
```

## 7. Release and confirm — `[GATED]`

A stopped pay-as-you-go instance still incurs charges: release immediately
after the copy-back, then confirm the ID is gone.

```bash
aliyun ecs ReleaseInstance --RegionId ap-southeast-1 --InstanceId <InstanceId>

# confirmation: this must return no instances for that ID
aliyun ecs DescribeInstances --RegionId ap-southeast-1 \
  --InstanceIds '["<InstanceId>"]'
```

## 8. Sign-off

Fill `cloud/README.md` (instance ID, creation/release times, GPU, training
time, billed cost from the billing console) only from live outputs. All
`[GATED]` sections above are to be run by or together with the human after
price approval; pi will not create or release resources on its own.