"""Part 4 evidence builder — READ-ONLY postprocessing. Never trains.

Producing command (only after ALL 12 formal-decoder-02 runs are complete):

    uv run python make_part4_evidence.py

Reads ONLY saved logs/checkpoints/receipts under
artifacts/part4/formal-decoder-02/ (namespace of the corrected decoder
protocol, commit 473efb3, grok.py SHA d0501c3b...). The invalidated
symmetric-variant attempt artifacts/part4/formal-01 is NEVER read or mixed
in.

Default (strict) execution:
  1. asserts manager-status.json status == "COMPLETE";
  2. asserts all 12 run metrics exist and their SHA256 match the
     manager receipt;
  3. only then writes evidence/part4-*.png, evidence/part4-summary.json
     and evidence/part4-summary.md.
If training is still running it prints a clear message and exits NONZERO
without writing any final evidence figure or success summary.

`--preview` is a nonformal rendering aid: it skips the completion gate,
tolerates missing runs, writes draft PNGs ONLY into evidence-preview/
(never a summary, never evidence/).

Figures produced (strict mode):
  part4-grok-curves.png  — main seeds 0..4: train/test accuracy, loss and
                           all-parameter norm vs step (log-x). Step 0 is
                           preserved in the raw data but excluded from the
                           log axis; the initialization is annotated
                           separately. Every seed stays visible, including
                           the no-grok seed 0.
  part4-controls.png     — test accuracy vs step for main, wd0, 50%-fraction
                           and random-label controls with the chance 1/97
                           line (small multiples).
  part4-fourier.png      — Fourier power k=1..48 at initialization,
                           memorization and final per main seed, final key
                           frequencies marked; own-snapshot top-5 shares
                           printed. If no grok the snapshot is labelled
                           FINAL, not "after grokking"; a missing
                           memorization checkpoint is shown explicitly.
  part4-ablations.png    — final test accuracy for trained / keep-only
                           key5 / remove key5 / remove 5 random other /
                           keep-only 5 random other (all 10 draws shown),
                           with the 0.95 / 0.05 gates.

Gates are reported exactly as measured; failed gates stay failed. Controls
are observed results, not guaranteed successes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import grok as gk

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts" / "part4" / "formal-decoder-02"
EVID = ROOT / "evidence"
PREVIEW = ROOT / "evidence-preview"

MAIN = [f"main-s{i}" for i in range(5)]
WD0 = [f"wd0-s{i}" for i in range(3)]
FRAC50 = [f"frac50-s{i}" for i in range(3)]
RANDOM = ["random-labels-s0"]
RUN_NAMES = MAIN + WD0 + FRAC50 + RANDOM          # 12 fixed jobs
CHANCE = 1 / 97
LN97 = float(np.log(97))


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics_path(name: str) -> Path:
    return ART / name / "metrics.json"


def load_available_metrics() -> dict[str, dict]:
    out = {}
    for name in RUN_NAMES:
        p = metrics_path(name)
        if p.exists():
            out[name] = json.loads(p.read_text())
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

SEED_COLORS = {0: "#1f77b4", 1: "#d62728", 2: "#2ca02c", 3: "#9467bd",
               4: "#ff7f0e"}


def fig_grok_curves(runs: dict, outdir: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    init_notes = []
    for name in MAIN:
        m = runs[name]
        seed = m["seed"]
        c = SEED_COLORS[seed]
        log = m["log"]
        steps = [e["step"] for e in log]
        s_pos = steps[1:]                       # step 0 excluded from log-x
        lab = f"seed {seed}" + ("" if m["grok_step"] is not None
                                else " (no grok, FINAL)")
        axes[0].plot(s_pos, [e["train_acc"] for e in log[1:]], ls="--",
                     color=c, alpha=0.45, lw=1.2)
        axes[0].plot(s_pos, [e["test_acc"] for e in log[1:]], ls="-",
                     color=c, alpha=0.9, lw=1.4, label=lab)
        if m["grok_step"] is not None:
            axes[0].scatter([m["grok_step"]], [1.0], marker="*", s=130,
                            color=c, zorder=6, edgecolor="k", linewidths=0.4)
        axes[1].plot(s_pos, [e["train_loss"] for e in log[1:]], ls="--",
                     color=c, alpha=0.45, lw=1.2)
        axes[1].plot(s_pos, [e["test_loss"] for e in log[1:]], ls="-",
                     color=c, alpha=0.9, lw=1.4)
        axes[2].plot(s_pos, [e["param_norm"] for e in log[1:]], ls="-",
                     color=c, alpha=0.9, lw=1.4)
        e0 = log[0]
        init_notes.append(f"{seed}:{e0['param_norm']:.1f}")
    axes[0].axhline(CHANCE, color="gray", ls=":", lw=1)
    axes[0].axhline(0.99, color="crimson", ls=":", lw=1)
    axes[0].text(6000, 0.06, "chance 1/97", color="gray", fontsize=8)
    axes[0].text(200, 1.005, "0.99", color="crimson", fontsize=8)
    axes[0].set_ylabel("accuracy")
    axes[0].set_title("Main runs (30% split, weight decay 1)\n"
                      "Solid: test; dashed: train; star: first test accuracy >= 0.99\n"
                      "Step 0 omitted on log x-axis; initial norms (seed:norm): "
                      + ", ".join(init_notes), fontsize=9)
    axes[1].axhline(LN97, color="gray", ls=":", lw=1)
    axes[1].text(6000, LN97 * 1.15, "uniform ln 97 = 4.57", color="gray",
                 fontsize=8)
    axes[1].set_ylabel("loss (nats)")
    axes[1].set_yscale("log")
    axes[2].set_ylabel("all-parameter norm ‖θ‖₂")
    axes[2].set_xlabel("step (log scale; step 0 excluded, see annotation)")
    for ax in axes:
        ax.set_xscale("log")
    axes[0].legend(fontsize=8, loc="lower left", bbox_to_anchor=(0.02, 0.25))
    fig.suptitle("Grokking of (a+b) mod 97 — seeds 0..4, one-layer decoder, "
                 "no LayerNorm", y=0.995, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(outdir / "part4-grok-curves.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def fig_controls(runs: dict, outdir: Path) -> None:
    groups = [("main seeds 0-4 (30%, wd 1)", MAIN),
              ("wd 0 controls, seeds 0-2", WD0),
              ("50% training fraction, seeds 0-2", FRAC50),
              ("random labels, seed 0 (3000 steps)", RANDOM)]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True,
                             sharey=True)
    for ax, (title, names) in zip(axes.ravel(), groups):
        for name in names:
            if name not in runs:
                continue
            m = runs[name]
            log = m["log"]
            c = SEED_COLORS.get(m["seed"], "gray")
            lbl = (f"seed {m['seed']}" +
                   ("" if m["grok_step"] is not None else " (no grok)"))
            ax.plot([e["step"] for e in log[1:]],
                    [e["test_acc"] for e in log[1:]], color=c, lw=1.3,
                    label=lbl)
            fin = log[-1]["test_acc"]
            ax.scatter([log[-1]["step"]], [fin], color=c, s=18, zorder=5)
            # Exact final values are in the summary table; omit overlapping endpoint labels.
        ax.axhline(CHANCE, color="gray", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_title(title, fontsize=9)
        ax.set_ylim(-0.03, 1.05)
        ax.label_outer()
    for ax in axes[1]:
        ax.set_xlabel("step (log)")
    for ax in axes[:, 0]:
        ax.set_ylabel("test accuracy")
    axes[0, 0].text(60, CHANCE + 0.03, "chance 1/97 ≈ 0.010", fontsize=7,
                    color="gray")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Part 4 controls (observed results): test accuracy vs step\n"
                 "main vs no weight decay vs 50% data vs random labels",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outdir / "part4-controls.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _power(E: np.ndarray) -> np.ndarray:
    return gk.fourier_power(np.asarray(E))


def fig_fourier(runs: dict, outdir: Path) -> dict:
    """Per main seed: power k=1..48 at init/memorization/final."""
    shares = {}
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    axes = axes.ravel()
    for i, name in enumerate(MAIN):
        m = runs[name]
        seed = m["seed"]
        ax = axes[i]
        ckpt = ART / name
        E_init = gk.load_params(ckpt / "checkpoint-init.npz")["emb"]
        E_fin = gk.load_params(ckpt / "checkpoint-final.npz")["emb"]
        p_init, p_fin = _power(E_init), _power(E_fin)
        have_mem = (ckpt / "checkpoint-memorization.npz").exists()
        if have_mem:
            E_mem = gk.load_params(ckpt / "checkpoint-memorization.npz")["emb"]
            p_mem = _power(E_mem)
        ks = np.arange(1, 49)
        ax.plot(ks, p_init, color="#888888", lw=1.1, label="initialization")
        if have_mem:
            ax.plot(ks, p_mem, color="#3a6ea5", lw=1.1,
                    label="memorization")
        else:
            ax.text(0.5, 0.85, "memorization checkpoint:\nunavailable",
                    transform=ax.transAxes, ha="center", fontsize=8,
                    color="#3a6ea5")
        grokked = m["grok_step"] is not None
        final_lbl = "final (grokked)" if grokked else "FINAL (no grok)"
        ax.plot(ks, p_fin, color="#c46a1f", lw=1.4, label=final_lbl)
        key5 = m["fourier"]["final"]["top5"]
        ax.scatter(key5, p_fin[np.array(key5) - 1], marker="*", s=110,
                   color="crimson", zorder=6, label="final key frequencies")
        ax.set_yscale("log")
        ax.set_title(f"seed {seed} — " +
                     (f"grok step {m['grok_step']}" if grokked
                      else "NO GROK (FINAL)"), fontsize=9)
        ax.set_xlabel("frequency k")
        if i % 3 == 0:
            ax.set_ylabel("power (sum |X_k|² over dims)")
        mem_share = (m["fourier"]["memorization"] or {}).get("share")
        mem_text = "n/a" if mem_share is None else f"{mem_share:.3f}"
        ax.set_title(ax.get_title() + "\nTop-5 share: init "
                     f"{m['fourier']['init']['share']:.3f}; mem {mem_text}; "
                     f"final {m['fourier']['final']['share']:.3f}", fontsize=8)
        shares[seed] = {
            "init_top5": m["fourier"]["init"]["top5"],
            "init_share": m["fourier"]["init"]["share"],
            "memorization_top5": (m["fourier"]["memorization"] or {}).get("top5")
                                 if m["fourier"]["memorization"] else None,
            "memorization_share": (m["fourier"]["memorization"] or {}).get("share")
                                  if m["fourier"]["memorization"] else None,
            "final_top5": key5,
            "final_share": m["fourier"]["final"]["share"],
            "memorization_checkpoint": have_mem,
        }
        print(f"seed {seed} own-snapshot top-5 shares: init "
              f"{shares[seed]['init_share']:.4f} {shares[seed]['init_top5']}, "
              + (f"memorization {shares[seed]['memorization_share']:.4f} "
                 f"{shares[seed]['memorization_top5']}, "
                 if have_mem else "memorization UNAVAILABLE, ")
              + f"{final_lbl} {shares[seed]['final_share']:.4f} {key5}")
    axes[5].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    labels = ["final checkpoint" if x in ("FINAL (no grok)", "final (grokked)") else x for x in labels]
    axes[5].legend(handles, labels, loc="upper left", frameon=False, fontsize=9)
    axes[5].text(0.0, 0.52,
                 "Residue embedding: rfft over a=0..96.\n"
                 "Frequencies k=1..48; DC excluded.\n"
                 "Power is summed over 64 dimensions.\n\n"
                 "Power is diffuse at memorization.\n"
                 "For grokked seeds 1-4, final power\n"
                 "concentrates at a few frequencies.\n"
                 "Seed 0 did not grok within budget.",
                 fontsize=9, va="top")
    fig.suptitle("Part 4: Fourier power of the residue embedding — "
                 "initialization, memorization, final per main seed",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(outdir / "part4-fourier.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return shares


def fig_ablations(runs: dict, outdir: Path) -> None:
    cats = ["Trained", "Keep\nkey 5", "Remove\nkey 5",
            "Remove\nother 5", "Keep\nother 5"]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.8), sharey=True)
    for i, name in enumerate(MAIN):
        m = runs[name]
        ax = axes[i]
        ab = m["ablations"]
        trained = m["final_log_entry"]["test_acc"]
        ro = ab["remove_random_other5"]
        ko = ab["keep_only_random_other5"]
        ax.scatter([0], [trained], color="#1a3f66", s=40, zorder=5)
        ax.scatter([1], [ab["keep_only_key5"]], color="#1a3f66", s=40,
                   zorder=5)
        ax.scatter([2], [ab["remove_key5"]], color="#1a3f66", s=40, zorder=5)
        ax.scatter(3 + np.linspace(-0.12, 0.12, len(ro)), ro, color="#3a6ea5", s=22, alpha=0.85,
                   zorder=5, label="10 draws")
        ax.scatter([3], [float(np.mean(ro))], marker="_", s=260,
                   color="crimson", zorder=6, label="mean")
        ax.scatter(4 + np.linspace(-0.12, 0.12, len(ko)), ko, color="#3a6ea5", s=22, alpha=0.85,
                   zorder=5)
        ax.scatter([4], [float(np.mean(ko))], marker="_", s=260,
                   color="crimson", zorder=6)
        ax.axhline(0.95, color="green", ls=":", lw=1.2)
        ax.axhline(0.05, color="crimson", ls=":", lw=1.2)
        ax.axhline(CHANCE, color="gray", ls=":", lw=1)
        ax.set_xticks(range(5), cats, fontsize=7)
        ax.set_title(f"seed {m['seed']}" +
                     (f" (grok {m['grok_step']})" if m["grok_step"]
                      else " (NO GROK)"), fontsize=9)
        ax.set_ylim(-0.03, 1.08)
        if i == 0:
            ax.set_ylabel("test accuracy")
            ax.text(-0.35, 0.965, "0.95 gate", color="green", fontsize=7)
            ax.text(-0.35, 0.065, "0.05 gate", color="crimson", fontsize=7)
            ax.text(-0.35, CHANCE + 0.012, "chance 1/97", color="gray",
                    fontsize=6.5)
        if i == 4:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.suptitle("Part 4: frequency ablations on final checkpoints\n"
                 "All seeds shown; seed 0 did not grok; other-5 controls use 10 draws (x-jittered)", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.90))
    fig.savefig(outdir / "part4-ablations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# gates, table, summary
# ---------------------------------------------------------------------------

def main_gates(m: dict) -> dict:
    ab = m["ablations"]
    ro = ab["remove_random_other5"]
    ko = ab["keep_only_random_other5"]
    return {
        "grok_within_10000": m["grok_step"] is not None and m["grok_step"] <= 10000,
        "delay_gt_10": m["delay_ratio"] is not None and m["delay_ratio"] > 10,
        "final_top5_share_gt_0.8": m["fourier"]["final"]["share"] > 0.8,
        "keep_only_key5_gt_0.95": ab["keep_only_key5"] > 0.95,
        "remove_key5_lt_0.05": ab["remove_key5"] < 0.05,
        "remove_random_other5_gt_0.95_all10": len(ro) == 10 and all(v > 0.95 for v in ro),
        "keep_only_random_other5_lt_0.05_all10": len(ko) == 10 and all(v < 0.05 for v in ko),
    }


def md_table(runs: dict, gates: dict) -> str:
    lines = [
        "| seed | memorize step | grok step | delay | final train acc | "
        "final test acc | final top-5 share | key5 | keep key | remove key | "
        "remove other (10 draws) | keep other (10 draws) | all gates |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in MAIN:
        m = runs[name]
        ab = m["ablations"]
        ro, ko = ab["remove_random_other5"], ab["keep_only_random_other5"]
        ok = all(gates[m["seed"]].values())
        key5 = ",".join(str(k) for k in m["fourier"]["final"]["top5"])
        delay_text = "null" if m["delay_ratio"] is None else f"{m['delay_ratio']:.2f}"
        lines.append(
            f"| {m['seed']} | {m['memorization_step']} | "
            f"{m['grok_step'] if m['grok_step'] is not None else 'null'} | "
            f"{delay_text} | {m['final_log_entry']['train_acc']:.4f} | "
            f"{m['final_log_entry']['test_acc']:.4f} | "
            f"{m['fourier']['final']['share']:.4f} | {key5} | "
            f"{ab['keep_only_key5']:.4f} | {ab['remove_key5']:.4f} | "
            f"min {min(ro):.4f} / max {max(ro):.4f} | "
            f"max {max(ko):.4f} | {'PASS' if ok else 'FAIL'} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preview", action="store_true",
                    help="nonformal rendering check into evidence-preview/ "
                         "(skips the completion gate, writes no summary)")
    args = ap.parse_args(argv)

    if args.preview:
        runs = load_available_metrics()
        done = sorted(runs)
        print(f"PREVIEW (nonformal): {len(done)}/{len(RUN_NAMES)} runs have "
              f"metrics: {done}")
        if not runs:
            print("nothing to preview yet", file=sys.stderr)
            return 2
        outdir = PREVIEW
        outdir.mkdir(exist_ok=True)
        fig_grok_curves(runs, outdir)
        fig_controls(runs, outdir)
        fig_fourier(runs, outdir)
        fig_ablations(runs, outdir)
        print(f"preview PNGs written to {outdir} — NOT final evidence")
        return 0

    # ---- strict gate: complete manager + all 12 metrics -------------------
    status_path = ART / "manager-status.json"
    if not status_path.exists():
        print(f"STRICT GATE FAILED: {status_path} does not exist. "
              "Refusing to write final evidence.", file=sys.stderr)
        return 2
    mgr = json.loads(status_path.read_text())
    if mgr.get("status") != "COMPLETE":
        print(f"STRICT GATE FAILED: manager status is {mgr.get('status')!r} "
              f"(updated {mgr.get('updated_utc')}); formal-decoder-02 is not "
              "complete. Run this script only after all 12 jobs finish. "
              "No final evidence written.", file=sys.stderr)
        return 2
    completed = {c["name"]: c for c in mgr.get("completed", [])}
    missing = [n for n in RUN_NAMES if n not in completed
               or not metrics_path(n).exists()]
    if missing:
        print(f"STRICT GATE FAILED: missing completed metrics for {missing}. "
              "No final evidence written.", file=sys.stderr)
        return 2
    # metrics hashes must match the manager receipt
    for n, c in completed.items():
        got = sha(metrics_path(n))
        if got != c["metrics_sha256"]:
            print(f"STRICT GATE FAILED: metrics hash mismatch for {n}: "
                  f"{got} != receipt {c['metrics_sha256']}. "
                  "No final evidence written.", file=sys.stderr)
            return 2
    print(f"strict gate passed: manager COMPLETE, all 12 metrics present "
          f"and hash-matched (finished {mgr.get('finished_utc')})")

    runs = {n: json.loads(metrics_path(n).read_text()) for n in RUN_NAMES}

    # independent fourier cross-check against the checkpoints
    for name in MAIN:
        ck = ART / name
        for snap, fname in (("init", "checkpoint-init.npz"),
                            ("final", "checkpoint-final.npz")):
            ks, share = gk.top5(np.asarray(gk.load_params(ck / fname)["emb"]))
            rec = runs[name]["fourier"][snap]
            assert ks == rec["top5"] and abs(share - rec["share"]) < 1e-9, \
                f"fourier mismatch for {name}/{snap}"
        if runs[name]["fourier"]["memorization"]:
            ks, share = gk.top5(np.asarray(gk.load_params(
                ck / "checkpoint-memorization.npz")["emb"]))
            rec = runs[name]["fourier"]["memorization"]
            assert ks == rec["top5"] and abs(share - rec["share"]) < 1e-9, \
                f"fourier mismatch for {name}/memorization"
    print("fourier.json cross-checked against all main checkpoints: OK")

    EVID.mkdir(exist_ok=True)
    fig_grok_curves(runs, EVID)
    fig_controls(runs, EVID)
    fourier_shares = fig_fourier(runs, EVID)
    fig_ablations(runs, EVID)

    gates = {m["seed"]: main_gates(runs[n]) for n, m in runs.items() if n in MAIN}
    for s, g in sorted(gates.items()):
        for k, v in g.items():
            print(f"gate seed {s} {k}: {'PASS' if v else 'FAIL'}")
    all_pass = all(all(g.values()) for g in gates.values())
    print("all main gates pass:", all_pass)

    src = {n: sha(ROOT / n) for n in ("grok.py", "SPEC.md", "run_part4.py",
                                      "PART4_DECODER_PROTOCOL.md")}
    hash_match = all(src[n] == mgr["source_hashes"][n] for n in src)

    per_run = {}
    for n in RUN_NAMES:
        m = runs[n]
        c = completed[n]
        cmd = " ".join(c["command"]) if "command" in c else None
        per_run[n] = {
            "seed": m["seed"], "fraction": m["fraction"], "wd": m["wd"],
            "steps_run": m["steps_run"],
            "memorization_step": m["memorization_step"],
            "grok_step": m["grok_step"], "delay_ratio": m["delay_ratio"],
            "final_train_acc": m["final_log_entry"]["train_acc"],
            "final_test_acc": m["final_log_entry"]["test_acc"],
            "final_param_norm": m["final_log_entry"]["param_norm"],
            "fourier": m["fourier"], "ablations": m["ablations"],
            "metrics_sha256": c["metrics_sha256"],
            "settings_sha256": c["settings_sha256"],
            "actual_command": cmd,
            "finished_utc": m["utc_finished"],
        }

    summary = {
        "read_only": True,
        "namespace": "artifacts/part4/formal-decoder-02",
        "protocol": "PART4_DECODER_PROTOCOL.md (frozen; commit 473efb3 line)",
        "source_hashes_recomputed": src,
        "source_hashes_match_manager_receipt": hash_match,
        "manager": {"status": mgr["status"], "started_utc": mgr["started_utc"],
                    "finished_utc": mgr.get("finished_utc"),
                    "actor": mgr["actor"]},
        "environment": {"jax_platform": "cpu (JAX_PLATFORMS=cpu, taskset "
                                         "pinned cores; no GPU/cloud)",
                        "jax": "0.6.2", "numpy": "2.2.6",
                        "python": "3.11", "record": "evidence/environment.json"},
        "invalidated_predecessor": {
            "excluded": "artifacts/part4/formal-01 (symmetric noPos/noMask "
                        "variant) is INVALIDATED for the worksheet controls "
                        "(393 conflicting unordered pairs -> max train acc "
                        "0.860786 < 0.99) and is NOT read by this script nor "
                        "mixed into these results",
            "record": "artifacts/part4/formal-01/INVALIDATED-PROTOCOL.json"},
        "per_run": per_run,
        "fourier_own_snapshot_shares": fourier_shares,
        "main_gates": {str(s): g for s, g in sorted(gates.items())},
        "all_main_gates_pass": all_pass,
        "controls_note": "wd0, 50%-fraction and random-label controls are "
                         "observed results of the frozen protocol, not "
                         "guaranteed successes; no gate is applied to them.",
        "artifact_paths": {n: str(metrics_path(n)) for n in RUN_NAMES},
        "figures": ["evidence/part4-grok-curves.png",
                    "evidence/part4-controls.png",
                    "evidence/part4-fourier.png",
                    "evidence/part4-ablations.png"],
        "protocol_hash": src["PART4_DECODER_PROTOCOL.md"],
        "grok_py_hash": src["grok.py"],
    }
    (EVID / "part4-summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    table = md_table(runs, gates)
    (EVID / "part4-summary.md").write_text(
        "# Part 4 formal-decoder-02 — per-main-seed evidence\n\n"
        "Producing command: `uv run python make_part4_evidence.py` — "
        "strict-gated on manager COMPLETE + 12 hash-matched metrics.\n\n"
        + table + "\n\nGates: grok within 10000, delay > 10, final top-5 "
        "share > 0.8, keep key5 > 0.95, remove key5 < 0.05, remove random "
        "other5 > 0.95 in all 10 draws, keep only random other5 < 0.05 in "
        "all 10 draws. Failed gates stay failed; controls are observed "
        "results.\n")
    print("wrote evidence/part4-{grok-curves,controls,fourier,ablations}.png "
          "and part4-summary.{json,md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
