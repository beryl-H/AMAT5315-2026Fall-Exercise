"""Part 4 reference check: compare the fixed-weight JAX UNet port against
official diffusers v0.40.0 PyTorch CPU on fixed inputs.

Run ONLY after: (a) full verified weights (SHA256
1558dbe3fb093b0857d473e376b711eeda1b186c557be5e85b3a56300cd49f0f), and
(b) the root-owned artifacts/reference-venv with torch==2.8.0+cpu and
diffusers==0.40.0 imports PASS. Preflight checks both conditions and exits
with a clear status otherwise.

Usage: PYTHONPATH=. <jax .venv python> scripts/part4_reference_check.py
"""

import hashlib
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

WEIGHTS = "inputs/cifar-model/diffusion_pytorch_model.safetensors"
EXPECTED_SHA256 = "1558dbe3fb093b0857d473e376b711eeda1b186c557be5e85b3a56300cd49f0f"
REF_VENV = "artifacts/reference-venv/bin/python"
SEED = 7400
INDICES = [0, 499, 999]
TOL_MAXABS = 1e-3
TOL_RELRMS = 1e-4


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def preflight():
    sha = sha256_file(WEIGHTS)
    if sha != EXPECTED_SHA256:
        raise SystemExit(f"WEIGHT SHA MISMATCH: {sha}")
    print("weights SHA256 OK:", sha)
    r = subprocess_run([REF_VENV, "-c",
                        "import torch, diffusers, safetensors; "
                        "print(torch.__version__, diffusers.__version__)"])
    if r.returncode != 0:
        raise SystemExit("reference venv not ready:\n" + r.stderr.decode())
    print("reference env:", r.stdout.decode().strip())


def subprocess_run(cmd):
    import subprocess
    return subprocess.run(cmd, capture_output=True)


def torch_reference(inputs):
    """Run official diffusers UNet2DModel in the reference venv."""
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        np.savez(f, inputs=inputs)
        path = f.name
    script = f"""
import numpy as np, torch, json
from diffusers import UNet2DModel
d = np.load({path!r})
x = torch.tensor(d['inputs'], dtype=torch.float32)
m = UNet2DModel.from_pretrained("inputs/cifar-model")
m.eval()
outs = {{}}
with torch.no_grad():
    for i in {INDICES}:
        t = torch.tensor([i], dtype=torch.long)
        outs[str(i)] = m(x, t).sample.numpy()
print("TORCH_JSON" + json.dumps(outs))
"""
    r = subprocess_run([REF_VENV, "-c", script])
    if r.returncode != 0:
        raise SystemExit("torch reference failed:\n" + r.stderr.decode())
    line = [l for l in r.stdout.decode().splitlines()
            if l.startswith("TORCH_JSON")][0]
    return json.loads(line[len("TORCH_JSON"):])


def main():
    preflight()
    from src.cifar_unet import load_cifar_unet, cifar_unet_predict
    weights = load_cifar_unet(WEIGHTS, "inputs/cifar-model/config.json")
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(1, 3, 32, 32)).astype(np.float32)
    torch_out = torch_reference(x)
    report = {}
    for i in INDICES:
        y = np.asarray(cifar_unet_predict(weights, jnp.asarray(x), i))
        t = torch_out[str(i)]
        maxabs = float(np.abs(y - t).max())
        relrms = float(np.sqrt(((y - t) ** 2).sum() / (t ** 2).sum()))
        report[str(i)] = {"max_abs": maxabs, "relative_rms": relrms,
                          "shape": list(y.shape)}
        print(f"index {i}: max|diff| = {maxabs:.3e}  rel RMS = {relrms:.3e}")
        np.savez(f"artifacts/part4/refcheck_index_{i}.npz",
                 input=x, jax=y, torch=t)
    ok = all(v["max_abs"] < TOL_MAXABS and v["relative_rms"] < TOL_RELRMS
             for v in report.values())
    report["tolerances"] = {"max_abs": TOL_MAXABS, "rel_rms": TOL_RELRMS,
                            "pass": ok}
    with open("evidence/part4_reference_check.json", "w") as f:
        json.dump(report, f, indent=2)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
