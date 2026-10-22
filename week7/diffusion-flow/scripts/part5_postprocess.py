"""Part 5 saved-results validation (postprocess): NO UNet, NO sampler calls.

Loads the original formal run raw arrays, recomputes finite/shape checks,
all 16 hidden MSEs (float64, /1200), observed max changes, seed mean and
population variance, and verifies hashes/levels/seeds. Writes a NEW small
receipt under evidence/ plus a metrics/provenance summary; never overwrites
the original run outputs or metadata.

The original formal driver exited 1 on a stale summary-key print AFTER all
raw arrays and run_meta.json were written; this receipt records that history
honestly (formal entrypoint EXIT 1, data complete and validated).
"""

import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "artifacts/part5"
EVID = BASE / "evidence"
SEEDS = [7201, 7202, 7203, 7204]

meta = json.load(open(RAW / "run_meta.json"))
d = np.load(RAW / "inputs_used.npz")
truth, obs, mask = d["truth"], d["observations"], d["mask"]
comps = np.load(RAW / "completions_4seeds.npy")
states = np.load(RAW / "event_states_4seeds.npy")
init = np.load(RAW / "initial_states_4seeds.npy")
keys = np.load(RAW / "keys_4seeds.npy")
ev = np.load(RAW / "events.npz")
metrics = np.load(RAW / "metrics.npz")

checks = {}
checks["completions_shape"] = list(comps.shape)
checks["completions_finite"] = bool(np.isfinite(comps).all())
checks["event_states_shape"] = list(states.shape)          # [4,99,4,3,32,32]
checks["event_states_finite"] = bool(np.isfinite(states).all())
checks["initial_states_shape"] = list(init.shape)
checks["initial_states_finite"] = bool(np.isfinite(init).all())
checks["keys_shape"] = list(keys.shape)                    # [4,2,2] uint32

hidden = mask == 0
mses = np.zeros((4, 4))
oc = np.zeros((4, 4))
c64, t64 = comps.astype(np.float64), truth.astype(np.float64)
for si in range(4):
    for ii in range(4):
        err = (c64[si, ii] - t64[ii])[hidden[ii]]
        mses[si, ii] = (err ** 2).sum() / (400 * 3)
        oc[si, ii] = np.abs((c64[si, ii] - t64[ii])[~hidden[ii]]).max()
checks["mse_matches_saved"] = bool(
    np.allclose(mses, np.asarray(meta["hidden_mse_sum_sq_over_400x3"]),
                atol=0, rtol=1e-12))
checks["observed_changes_match_saved"] = bool(
    np.allclose(oc, np.asarray(meta["observed_max_changes"]), rtol=1e-12))
checks["all_observed_changes_below_1e-5"] = bool((oc < 1e-5).all())
# observed values identical across seeds and zero variance outside hole
ident = all(np.array_equal(c64[s, ii][~hidden[ii]], c64[0, ii][~hidden[ii]])
            for s in range(4) for ii in range(4))
checks["observed_identical_across_seeds"] = ident
var = c64.var(axis=0, ddof=0)
checks["zero_variance_outside_hole"] = bool(
    np.all(var[..., ~hidden[0]] == 0.0))
checks["positive_variance_inside_hole"] = bool(
    np.all(var[..., hidden[0]] > 0.0))
checks["levels_1000_to_0"] = (int(ev["t"][0]) == 1000
                              and int(ev["s"][-1]) == 0)
checks["n_events_99"] = states.shape[1] == 99
checks["n_retries_49"] = int(ev["is_retry"].sum()) == 49
checks["seeds_recorded"] = [int(s) for s in ev["seeds"]] == SEEDS
checks["weights_hash_canonical"] = (
    meta["weights_sha256_after"] == meta["expected_weights_hash"] ==
    "c35165b672140a1afa6b293b46553e81fd8d56c9f4872fc8b9c1826afd1f6c04")
checks["weights_unchanged"] = meta["weights_unchanged"]
checks["file_hashes_present"] = sorted(meta["file_hashes"]) == [
    "config", "images_npz", "repaint_source", "weights"]

assert all(v is True or (isinstance(v, list) and v) for k, v in checks.items()
           if not isinstance(v, bool)), checks
receipt = {
    "original_formal_driver_exit": 1,
    "original_failure": ("stale summary key max_observed_change AFTER all "
                         "raw arrays and run_meta.json were written; no "
                         "computation lost; no regeneration performed"),
    "postprocess_exit": 0,
    "data_validated": True,
    "hidden_mse_sum_sq_over_400x3": mses.tolist(),
    "hidden_mse_average": float(mses.mean()),
    "observed_max_changes": oc.tolist(),
    "global_observed_max_change": float(oc.max()),
    "variance_convention": "population over seeds, divisor 4 (ddof=0)",
    "checks": checks,
    "provenance": {"files": meta["file_hashes"],
                   "versions": meta["versions"],
                   "wall_time_s": meta["wall_time_s"],
                   "seeds": SEEDS},
}
assert all(v is True for k, v in checks.items() if isinstance(v, bool)), checks
with open(EVID / "part5_postprocess_receipt.json", "w") as f:
    json.dump(receipt, f, indent=2)
print(json.dumps({"mse_average": receipt["hidden_mse_average"],
                  "global_observed_max_change":
                      receipt["global_observed_max_change"],
                  "all_checks_pass": True}, indent=1))
