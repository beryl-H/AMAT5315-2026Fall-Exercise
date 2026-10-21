"""Part 4 reference check: compare the fixed-weight JAX UNet port against
official diffusers v0.40.0 PyTorch CPU on fixed inputs.

Run ONLY after: (a) full verified weights (SHA256
1558dbe3fb093b0857d473e376b711eeda1b186c557be5e85b3a56300cd49f0f), and
(b) the root-owned artifacts/reference-venv with torch==2.8.0+cpu and
diffusers==0.40.0 imports PASS. Preflight checks both conditions and exits
with a clear status otherwise.

I/O: task-local NPZ files under artifacts/part4/reference/ (input, torch
outputs, JAX outputs); only small metadata on stdout.

Usage: PYTHONPATH=. .venv/bin/python scripts/part4_reference_check.py
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

BASE = Path(__file__).resolve().parent.parent
WEIGHTS = BASE / "inputs/cifar-model/diffusion_pytorch_model.safetensors"
CONFIG = BASE / "inputs/cifar-model/config.json"
EXPECTED_SHA256 = ("1558dbe3fb093b0857d473e376b711eeda1b186c557be5e85b3a"
                   "56300cd49f0f")
REF_VENV = BASE / "artifacts/reference-venv/bin/python"
OUT = BASE / "artifacts/part4/reference"
SEED = 7400
INDICES = [0, 499, 999]
TOL_MAXABS = 1e-3
TOL_RELRMS = 1e-4
KEY_MAP_OLD_NEW = (".query. -> .to_q.", ".key. -> .to_k.",
                   ".value. -> .to_v.", ".proj_attn. -> .to_out.0.")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd):
    return subprocess.run(cmd, capture_output=True)


def preflight():
    sha = sha256_file(WEIGHTS)
    if sha != EXPECTED_SHA256:
        raise SystemExit(f"WEIGHT SHA MISMATCH: {sha}")
    print("weights SHA256 OK:", sha)
    r = run([str(REF_VENV), "-c",
             "import torch, diffusers, safetensors, numpy; "
             "print(torch.__version__, diffusers.__version__, "
             "safetensors.__version__, numpy.__version__)"])
    if r.returncode != 0:
        raise SystemExit("reference venv not ready:\n" + r.stderr.decode())
    env = r.stdout.decode().strip()
    print("reference env:", env)
    tv, dv, sv, nv = env.split()
    assert tv == "2.8.0+cpu", f"pinned torch expected 2.8.0+cpu, got {tv}"
    assert dv == "0.40.0", f"pinned diffusers expected 0.40.0, got {dv}"
    assert "+cpu" in tv and not torch_cuda_probe(REF_VENV), "CPU only required"
    return env


def torch_cuda_probe(venv):
    r = run([str(venv), "-c", "import torch; print(torch.cuda.is_available())"])
    return r.stdout.decode().strip() == "True"


def torch_reference(x_np):
    """Official diffusers UNet2DModel on CPU; strict key/shape/finite
    accounting after the deprecated old->new attention key mapping."""
    OUT.mkdir(parents=True, exist_ok=True)
    in_path = OUT / "input_seed7400.npz"
    np.savez(in_path, x=x_np, indices=np.array(INDICES))
    script = f"""
import json, numpy as np, torch
from diffusers import UNet2DModel
torch.set_num_threads(4)
d = np.load({str(in_path)!r})
x = torch.tensor(d['x'], dtype=torch.float32)
m = UNet2DModel.from_pretrained(
    {str(BASE / 'inputs/cifar-model')!r}, local_files_only=True,
    use_safetensors=True, torch_dtype=torch.float32)
m.eval()
# strict accounting: map old attention names to new and compare every array
import safetensors.numpy as stn
raw = stn.load_file({str(WEIGHTS)!r})
mapped = {{}}
for k, v in raw.items():
    nk = k.replace('.query.', '.to_q.').replace('.key.', '.to_k.')
    nk = nk.replace('.value.', '.to_v.').replace('.proj_attn.', '.to_out.0.')
    mapped[nk] = v
sd = m.state_dict()
missing = sorted(set(sd) - set(mapped))
extra = sorted(set(mapped) - set(sd))
bad = [k for k in sd if k in mapped and
       (sd[k].shape != mapped[k].shape or
        not np.array_equal(sd[k].numpy(), mapped[k]))]
nonfinite = [k for k in sd if not torch.isfinite(sd[k]).all()]
print("ACCT" + json.dumps({{
    "n_state_dict": len(sd), "n_mapped": len(mapped),
    "missing": missing[:10], "n_missing": len(missing),
    "extra": extra[:10], "n_extra": len(extra),
    "mismatched": bad[:10], "n_mismatched": len(bad),
    "nonfinite": nonfinite[:10], "dtype_float32": all(
        sd[k].dtype == torch.float32 for k in sd)}}))
outs = {{}}
with torch.no_grad():
    for i in {INDICES}:
        t = torch.tensor([i], dtype=torch.long)
        outs[str(i)] = m(x, t).sample.numpy()
np.savez({str(OUT / 'torch_outputs.npz')!r}, **outs)
print("DONE")
"""
    r = run([str(REF_VENV), "-c", script])
    if r.returncode != 0:
        raise SystemExit("torch reference failed:\n" + r.stderr.decode())
    meta = None
    for line in r.stdout.decode().splitlines():
        if line.startswith("ACCT"):
            meta = json.loads(line[len("ACCT"):])
    if meta is None or meta["n_missing"] or meta["n_extra"] \
            or meta["n_mismatched"] or meta["nonfinite"] \
            or meta["n_state_dict"] != 328 or not meta["dtype_float32"]:
        raise SystemExit(f"OFFICIAL LOADER ACCOUNTING FAILED: {json.dumps(meta)}")
    print("official loader accounting: 328/328 arrays equal after key map")
    return OUT / "torch_outputs.npz", meta


def main():
    env = preflight()
    from src.cifar_unet import load_cifar_unet, cifar_unet_predict
    weights = load_cifar_unet(WEIGHTS, CONFIG)
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(1, 3, 32, 32)).astype(np.float32)
    torch_path, meta = torch_reference(x)
    torch_out = np.load(torch_path)

    # hash of the parameter arrays before/after sampling (unchanged check)
    def param_hash(w):
        h = hashlib.sha256()
        for k in sorted(w):
            h.update(k.encode())
            h.update(np.asarray(w[k]).tobytes())
        return h.hexdigest()

    h_before = param_hash(weights)
    predict_jit = jax.jit(cifar_unet_predict)
    t0 = time.perf_counter()
    y0 = np.asarray(predict_jit(weights, jnp.asarray(x), 0))  # compile+run
    compile_s = time.perf_counter() - t0
    report = {"env": env, "key_map": KEY_MAP_OLD_NEW,
              "official_loader_accounting": meta,
              "compile_and_first_eval_s": compile_s}
    for i in INDICES:
        y = y0 if i == 0 else np.asarray(predict_jit(weights,
                                                     jnp.asarray(x), i))
        t = torch_out[str(i)]
        maxabs = float(np.abs(y - t).max())
        relrms = float(np.sqrt(((y - t) ** 2).sum() / (t ** 2).sum()))
        report[str(i)] = {"max_abs": maxabs, "relative_rms": relrms,
                          "shape": list(y.shape)}
        print(f"index {i}: max|diff| = {maxabs:.3e}  rel RMS = {relrms:.3e}")
        np.savez(OUT / f"jax_output_index_{i}.npz", y=y)
    h_after = param_hash(weights)
    report["weights_hash_before"] = h_before
    report["weights_hash_after"] = h_after
    report["weights_unchanged"] = h_before == h_after
    ok = all(v["max_abs"] < TOL_MAXABS and v["relative_rms"] < TOL_RELRMS
             for v in report.values() if isinstance(v, dict)
             and "max_abs" in v) and h_before == h_after
    report["tolerances"] = {"max_abs": TOL_MAXABS, "rel_rms": TOL_RELRMS,
                            "pass": ok}
    (BASE / "evidence").mkdir(exist_ok=True)
    with open(BASE / "evidence/part4_reference_check.json", "w") as f:
        json.dump(report, f, indent=2)
    print("weights unchanged:", report["weights_unchanged"])
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    import time
    main()
