"""Part 2 evidence: VERIFY items 1, 3, 4 (item 2 metrics come from the
training driver). Produces PNGs in evidence/ and JSON metrics in evidence/;
RAW arrays (starts, snapshot states, donor mapping, per-chain Dice, clamp
checks, seed keys) are saved under artifacts/part2/evaluation-<timestamp>/
(git-ignored) for independent verification. No generated file is ever
deleted by this script.

Chain/key management (review fix): each run is ONE evolving chain; segment
lengths [1, 9, 90, 900] reach snapshots at steps 1, 10, 100, 1000 with the
chain state carried forward. Segment keys are split sequentially (fresh
randomness each segment and each step, shared across models/conditions but
never reused between segments).
"""

import json
import time
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ebm.data import load_mnist_pair
from ebm.rbm import binarize, clamped_gibbs_chain, dice_ink, gibbs_chain
from ebm.train import sha256_array

ROOT = Path(__file__).resolve().parent.parent
EV = ROOT / "evidence"
ART = ROOT / "artifacts/part2"
EVAL = ART / f"evaluation-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
STEPS = [1, 10, 100, 1000]
SEGMENTS = [1, 9, 90, 900]  # cumulative snapshot steps match STEPS
UPPER = 14  # observed rows 0..13 (392 bits); missing rows 14..27 (392 bits)
SEED = 2026


def load_params(k, epoch=20):
    d = np.load(ART / f"cd{k}/params_epoch{epoch:02d}.npz")
    return {kk: jnp.asarray(d[kk]) for kk in ("a", "b", "W")}


def load_test_split():
    test_images, test_labels = load_mnist_pair(
        ROOT / "inputs/mnist/t10k-images-idx3-ubyte.gz",
        ROOT / "inputs/mnist/t10k-labels-idx1-ubyte.gz")
    return test_images, test_labels


def save_fig(fig, name):
    fig.savefig(EV / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}", flush=True)


def evolve(params, v, key_root):
    """Carry one chain forward through SEGMENTS; return snapshot states."""
    states = []
    for L in SEGMENTS:
        seg_key, key_root = jax.random.split(key_root)
        run = jax.jit(lambda vv, kk: gibbs_chain(vv, params, kk, L))
        v = run(v, seg_key)
        states.append(np.asarray(v))
    return states  # snapshots at steps 1, 10, 100, 1000


def verify1():
    train_images, _ = load_mnist_pair(
        ROOT / "inputs/mnist/train-images-idx3-ubyte.gz",
        ROOT / "inputs/mnist/train-labels-idx1-ubyte.gz")
    bits = np.asarray(binarize(train_images[:16]))

    fig, axes = plt.subplots(2, 8, figsize=(10, 2.8), constrained_layout=True)
    for ax, img in zip(axes.ravel(), bits):
        ax.imshow(img, interpolation="nearest", cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("VERIFY1: first 16 binarized training images (v in {0,1})")
    save_fig(fig, "part2_binarized_first16.png")

    s = 2.0 * bits[0] - 1.0
    fig, ax = plt.subplots(figsize=(3.8, 3.8), constrained_layout=True)
    im = ax.imshow(s, interpolation="nearest", cmap="RdBu", vmin=-1, vmax=1)
    ax.set_title("spin configuration s=2v-1\n(+1 ink / -1 background), enlarged")
    fig.colorbar(im, ax=ax, shrink=0.8)
    save_fig(fig, "part2_spin_configuration.png")

    fig, ax = plt.subplots(figsize=(8.5, 5))
    vis_x = [3, 5, 7, 9, 11]
    hid_x = [3, 5, 7, 9]
    for xv in vis_x:
        for xh in hid_x:
            ax.plot([xv, xh], [0, 1], color="0.78", lw=0.6, zorder=1)
    ax.scatter(vis_x, [0] * 5, s=180, zorder=2, color="tab:blue")
    ax.scatter(hid_x, [1] * 4, s=180, zorder=2, color="tab:orange")
    ax.text(7, -0.34, "visible layer: 784 bits (28x28; subset drawn), bias a",
            ha="center", fontsize=10)
    ax.text(7, 1.14, "hidden layer: 128 bits (subset drawn), bias b",
            ha="center", fontsize=10)
    ax.text(14.2, 0.72, "couplings W_ij:\ncross-layer only,\nno within-layer edges",
            ha="center", fontsize=10)
    ax.annotate("every learned coupling joins one\nvisible bit to one hidden bit",
                xy=(6, 0.5), xytext=(14.2, 0.22), fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->"))
    ax.set_xlim(1, 17); ax.set_ylim(-0.45, 1.3); ax.axis("off")
    ax.set_title("RBM diagram: 784 visible / 128 hidden, restricted couplings")
    save_fig(fig, "part2_rbm_diagram.png")
    np.savez(EVAL / "verify1_binarized_first16.npz", bits=bits)


def verify3():
    p1, p10 = load_params(1), load_params(10)
    test_images, _ = load_test_split()
    test_bits = np.asarray(binarize(test_images[:16].reshape(-1, 784)))

    key = jax.random.PRNGKey(SEED)
    kstart, kchain = jax.random.split(key)
    start = jax.random.bernoulli(kstart, 0.5, (16, 784)).astype(jnp.float32)
    grids = {}
    for name, p in (("cd1", p1), ("cd10", p10)):
        grids[name] = evolve(p, start, kchain)  # [s1, s10, s100, s1000]

    # held-out originals and one-step sampled reconstructions (both models)
    recs = {}
    krec, _ = jax.random.split(kchain)
    for kk, p in ((1, p1), (10, p10)):
        recs[kk] = np.asarray(jax.jit(gibbs_chain)(
            jnp.asarray(test_bits), p, krec, 1))
    top = np.concatenate([test_bits, recs[1], recs[10]]).reshape(-1, 28, 28)
    fig, axes = plt.subplots(3, 16, figsize=(16, 3.4), constrained_layout=True)
    for row, label in enumerate(["held-out originals", "one-step recon CD-1",
                                 "one-step recon CD-10"]):
        for col in range(16):
            ax = axes[row, col]
            ax.imshow(top[row * 16 + col], interpolation="nearest",
                      cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label, fontsize=8)
    fig.suptitle("VERIFY3: first 16 held-out digits and one-step sampled reconstructions")
    save_fig(fig, "part2_reconstructions_first16.png")

    # generation grids: initial random row + steps 1,10,100,1000, per model
    for name in ("cd1", "cd10"):
        rows = [np.asarray(start)] + grids[name]
        fig, axes = plt.subplots(len(rows), 16, figsize=(16, 1.2 * len(rows)),
                                 constrained_layout=True)
        labels = ["random start"] + [f"step {s}" for s in STEPS]
        for r, imgrow in enumerate(rows):
            for c in range(16):
                ax = axes[r, c]
                ax.imshow(imgrow[c].reshape(28, 28), interpolation="nearest",
                          cmap="gray", vmin=0, vmax=1)
                ax.set_xticks([]); ax.set_yticks([])
                if c == 0:
                    ax.set_ylabel(labels[r], fontsize=8)
        fig.suptitle(f"VERIFY3: fresh generation, 16 chains, {name} — "
                     "same starts ACROSS MODELS (each chain has its own random start)")
        save_fig(fig, f"part2_generation_{name}.png")

    # first 3 chains (deterministic: stored order) across init + steps
    fig, axes = plt.subplots(3, 1 + len(STEPS) * 2, figsize=(14, 4.0),
                             constrained_layout=True)
    for chain in range(3):
        col = 0
        ax = axes[chain, col]
        ax.imshow(np.asarray(start)[chain].reshape(28, 28),
                  interpolation="nearest", cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if chain == 0:
            ax.set_title("shared random start", fontsize=7)
        col += 1
        for name in ("cd1", "cd10"):
            for si, step in enumerate(STEPS):
                ax = axes[chain, col]
                ax.imshow(grids[name][si][chain].reshape(28, 28),
                          interpolation="nearest", cmap="gray", vmin=0, vmax=1)
                ax.set_xticks([]); ax.set_yticks([])
                if chain == 0:
                    ax.set_title(f"{name} s{step}", fontsize=7)
                col += 1
    fig.suptitle("VERIFY3: first 3 generation chains (deterministic), init + 1/10/100/1000 steps")
    save_fig(fig, "part2_generation_trajectories.png")

    np.savez(EVAL / "verify3_generation_raw.npz",
             start=np.asarray(start),
             **{f"{n}_s{s}": grids[n][i] for n in ("cd1", "cd10")
                for i, s in enumerate(STEPS)},
             start_key_bits=sha256_array(np.asarray(kstart)),
             chain_key_bits=sha256_array(np.asarray(kchain)))


def donor_mapping(labels64):
    """First cyclic different-label held-out digit within the 64; fixed
    before any outcome. Full mapping returned."""
    mapping = []
    for i in range(64):
        for off in range(1, 64):
            j = (i + off) % 64
            if labels64[j] != labels64[i]:
                mapping.append(j)
                break
    return np.array(mapping)


def verify4():
    test_images, test_labels = load_test_split()
    imgs64 = test_images[:64]
    labs64 = test_labels[:64]
    bits64 = np.asarray(binarize(imgs64.reshape(-1, 784)))  # (64, 784)
    n_chains = 8
    assert bits64.shape[1] == 784 and UPPER * 28 == 392

    upper_mask = np.zeros((1, 784), dtype=bool)
    upper_mask[:, : UPPER * 28] = True
    assert int(upper_mask.sum()) == 392 and int((~upper_mask).sum()) == 392

    donor = donor_mapping(labs64)
    assert all(labs64[donor[i]] != labs64[i] for i in range(64))

    true_upper = np.repeat(bits64, n_chains, axis=0)      # (512, 784)
    wrong_upper = np.repeat(bits64[donor], n_chains, axis=0)
    truth_lower = bits64[:, UPPER * 28:].reshape(64, 1, -1)

    # one matched random-missing init for ALL conditions and models
    key = jax.random.PRNGKey(SEED)
    kinit, kstep = jax.random.split(key)
    init_missing = jax.random.bernoulli(
        kinit, 0.5, (64 * n_chains, 784)).astype(jnp.float32)
    init = jnp.where(jnp.asarray(upper_mask),
                     jnp.asarray(true_upper), init_missing)

    def run_condition(params, upper):
        clamped_start = jnp.where(jnp.asarray(upper_mask),
                                  jnp.asarray(upper), init)
        v = clamped_start
        states = []
        key_root = kstep
        for L in SEGMENTS:
            seg_key, key_root = jax.random.split(key_root)
            v = jax.jit(lambda vv, kk: clamped_gibbs_chain(
                vv, params, kk, L, jnp.asarray(upper),
                jnp.asarray(upper_mask)))(v, seg_key)
            states.append(np.asarray(v))
            # clamp invariant on the production function output
            assert np.array_equal(states[-1][:, :392], upper[:, :392]), \
                "clamped observed pixels changed"
        # missing-bit starts identical across conditions/models: same `init`
        assert np.array_equal(np.asarray(clamped_start)[:, 392:],
                              np.asarray(init)[:, 392:])
        return states

    stats = {}
    dice_full = {}
    cond_states = {}
    for k in (1, 10):
        params = load_params(k)
        for cond, upper in (("true", true_upper), ("wrong", wrong_upper)):
            t0 = time.perf_counter()
            states = run_condition(params, upper)
            cond_states[f"cd{k}_{cond}"] = states
            dice = np.zeros((len(STEPS), 64, n_chains))  # per-chain dice
            for si, step in enumerate(STEPS):
                vv = states[si].reshape(64, n_chains, 784)
                lower = vv[:, :, UPPER * 28:]
                for d in range(64):
                    for c in range(n_chains):
                        dice[si, d, c] = dice_ink(lower[d, c], truth_lower[d])
            dice_full[f"cd{k}_{cond}"] = dice
            stats[f"cd{k}_{cond}"] = {
                # average the 8 chains within each digit, then SEM over 64
                "mean": dice.mean(axis=2).mean(axis=1).tolist(),
                "sem_digits": (dice.mean(axis=2).std(axis=1, ddof=1)
                               / np.sqrt(64)).tolist(),
                "time_s": time.perf_counter() - t0,
            }
            print(f"cd{k} {cond}: mean@1000="
                  f"{stats[f'cd{k}_{cond}']['mean'][-1]:.4f}", flush=True)

    threshold_ok = {}
    for k in (1, 10):
        diff = (stats[f"cd{k}_true"]["mean"][-1]
                - stats[f"cd{k}_wrong"]["mean"][-1])
        threshold_ok[f"cd{k}"] = {
            "true_minus_wrong_at_1000": diff,
            "passes_ge_0.1": bool(diff >= 0.1),
        }
    print("VERIFY4 threshold:", json.dumps(threshold_ok, indent=2), flush=True)

    raw = {f"dice_{n}": dice_full[n] for n in dice_full}
    raw.update(donor=donor, labels=labs64,
               init_missing=np.asarray(init_missing),
               init_key_bits=sha256_array(np.asarray(kinit)),
               step_key_bits=sha256_array(np.asarray(kstep)))
    for n, sts in cond_states.items():
        for i, s in enumerate(STEPS):
            raw[f"states_{n}_s{s}"] = sts[i]
    np.savez_compressed(EVAL / "verify4_completion_raw.npz", **raw)

    (EV / "part2_completion_metrics.json").write_text(json.dumps(
        {"steps": STEPS, "n_digits": 64, "chains_per_digit": n_chains,
         "missing_rows": "14..27 (lower 14; 392 bits observed / 392 missing)",
         "donor_mapping": donor.tolist(),
         "donor_labels": labs64[donor].tolist(),
         "target_labels": labs64.tolist(),
         "stats": stats, "threshold": threshold_ok,
         "eval_dir": str(EVAL.relative_to(ROOT))}, indent=2))

    # plots: mean ± SEM vs steps
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)
    for ax, k in zip(axes, (1, 10)):
        for cond, color in (("true", "tab:green"), ("wrong", "tab:red")):
            m = np.array(stats[f"cd{k}_{cond}"]["mean"])
            e = np.array(stats[f"cd{k}_{cond}"]["sem_digits"])
            ax.errorbar(STEPS, m, yerr=e, marker="o", capsize=3,
                        color=color, label=f"{cond} upper half")
        ax.set_xscale("log"); ax.set_xticks(STEPS, [str(s) for s in STEPS])
        ax.set_xlabel("Gibbs steps"); ax.set_ylabel("ink Dice (missing half)")
        ax.set_title(f"CD-{k}"); ax.legend(fontsize=8)
    fig.suptitle("VERIFY4: completion ink overlap, 64 digits × 8 chains (mean ± SEM)")
    save_fig(fig, "part2_completion_curves.png")

    # per-model completion galleries: first 4 digits, predeclared chains 0-2,
    # steps 1,10,100,1000, plus clamped-upper and truth columns
    for k in (1, 10):
        params = load_params(k)
        states = run_condition(params, true_upper)  # reused, not rerun later
        fig, axes = plt.subplots(4, 2 + 3 * len(STEPS),
                                 figsize=(2.0 * (2 + 3 * len(STEPS)), 5.2),
                                 constrained_layout=True)
        for d in range(4):
            masked = bits64[d].copy(); masked[UPPER * 28:] = 0.0
            axes[d, 0].imshow(masked.reshape(28, 28), interpolation="nearest",
                              cmap="gray", vmin=0, vmax=1)
            axes[d, 0].set_ylabel(f"digit {int(labs64[d])}", fontsize=9)
            axes[d, 1].imshow(bits64[d].reshape(28, 28),
                              interpolation="nearest", cmap="gray",
                              vmin=0, vmax=1)
            col = 2
            for si, step in enumerate(STEPS):
                for c in range(3):  # predeclared chains 0,1,2
                    axes[d, col].imshow(
                        states[si].reshape(64, n_chains, 784)[d, c]
                        .reshape(28, 28), interpolation="nearest",
                        cmap="gray", vmin=0, vmax=1)
                    col += 1
            for ax in axes[d]:
                ax.set_xticks([]); ax.set_yticks([])
        for j, lab in enumerate(["observed", "truth"] +
                                [f"s{s} c{c}" for s in STEPS for c in (0, 1, 2)]):
            axes[0, j].set_title(lab, fontsize=7)
        fig.suptitle(f"VERIFY4: completions of first 4 held-out digits (chains 0-2), CD-{k}")
        save_fig(fig, f"part2_completion_{'cd1' if k==1 else 'cd10'}.png")


if __name__ == "__main__":
    EVAL.mkdir(parents=True, exist_ok=True)
    verify1()
    verify3()
    verify4()
    print("PART2_EVIDENCE_DONE", flush=True)
