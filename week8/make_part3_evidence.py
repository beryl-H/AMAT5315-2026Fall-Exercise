"""Part 3 evidence builder — READ-ONLY postprocessing. Never trains.

Producing command:
    uv run python make_part3_evidence.py

Reads: the 18 fit-run metrics/manifests, the sealed prediction192 receipt,
and the two holdout-192 run metrics, all under artifacts/part3/. Rebuilds
the fit from the raw 18 losses with the frozen conventions (loss-space WLS,
SEM |L1-L2|/2 floor 1e-3, Birge max(1, sqrt(chi2/6))), regenerates the
10000-draw seed-20260 parametric bootstrap from the SAVED fit only, verifies
the 192 quantiles reproduce the sealed interval, and plots the requested
log-log evidence including the honest blind MISS of width 192.
No frozen module is modified; no training happens here.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import scaling as sc

ROOT = Path(__file__).resolve().parent
FIT = ROOT / "artifacts" / "part3" / "formal-fit-01"
HOLD = ROOT / "artifacts" / "part3" / "formal-holdout-192-01"
EVID = ROOT / "evidence"
EVID.mkdir(exist_ok=True)

WIDTHS = [8, 12, 16, 24, 32, 48, 64, 96, 128]
BIGRAM = 2.481889


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> None:
    # ---- read raw run results --------------------------------------------
    losses = {}
    for w in WIDTHS:
        for s in (1, 2):
            m = json.loads((FIT / f"fit-manifest-w{w:03d}-s{s}.json").read_text())
            assert m["width"] == w and m["seed"] == s
            losses[(w, s)] = m["loss"]
    pred = json.loads((FIT / "prediction" / "prediction192.json").read_text())
    hold = {s: json.loads((HOLD / f"w192-s{s}" / "metrics.json").read_text())
            for s in (1, 2)}
    mgr = json.loads((HOLD / "manager-status.json").read_text())
    starts = {s: json.loads((HOLD / f"w192-s{s}" / "settings.json").read_text())["started_unix_ts"] for s in (1, 2)}
    first_start_unix = min(starts.values())

    # ---- rebuild the fit from the raw 18 losses (width 8 excluded) --------
    fit_widths = WIDTHS[1:]
    Ns = np.array([sc.nonembedding_params(w) for w in fit_widths], float)
    seed_means = np.array([np.mean([losses[(w, 1)], losses[(w, 2)]]) for w in fit_widths])
    sigmas = np.array([sc.sigma_points([losses[(w, 1)], losses[(w, 2)]])
                       for w in fit_widths])
    fit = sc.fit_wls(Ns, seed_means, sigmas)

    # cross-check against the sealed fit (same data, same conventions)
    sf = pred["fit"]
    fit_matches_sealed = (np.allclose(fit["Ns"], sf["Ns"])
                          and np.allclose(fit["Ls"], sf["Ls"], atol=1e-9)
                          and abs(fit["alpha"] - sf["alpha"]) < 1e-6
                          and abs(fit["a"] - sf["a"]) < 1e-6
                          and abs(fit["birge"] - sf["birge"]) < 1e-9)
    print(f"rebuilt fit matches sealed receipt: {fit_matches_sealed} "
          f"(alpha {fit['alpha']:.8f} vs sealed {sf['alpha']:.8f}, "
          f"birge {fit['birge']:.4f})")

    # ---- regenerate the bootstrap from the SAVED fit only -----------------
    boot = sc.bootstrap(fit, target_width=pred["target_width"],
                        target_N=pred["target_N"],
                        n_draws=pred["prediction"]["n_draws"],
                        seed=pred["prediction"]["seed"])
    lo, hi = boot["pred_interval"]
    slo, shi = pred["prediction"]["pred_interval"]
    quantiles_match = abs(lo - slo) < 1e-6 and abs(hi - shi) < 1e-6
    print(f"bootstrap 192 quantiles match sealed interval: {quantiles_match} "
          f"[{lo:.9f}, {hi:.9f}]")

    # ---- holdout result and the honest MISS -------------------------------
    hold_losses = [hold[s]["full_heldout_loss"] for s in (1, 2)]
    hold_mean = float(np.mean(hold_losses))
    inside = slo <= hold_mean <= shi
    miss_nats = hold_mean - shi if hold_mean > shi else (slo - hold_mean if hold_mean < slo else 0.0)
    print(f"holdout 192 losses: {hold_losses[0]:.6f}, {hold_losses[1]:.6f}; "
          f"mean {hold_mean:.6f}; sealed interval [{slo:.6f}, {shi:.6f}]")
    print(f"BLIND PREDICTION " + ("INSIDE" if inside else
          f"MISS (outside by {miss_nats:.4f} nats)"))

    # gates (reported honestly; a MISS is a legitimate result)
    gates = {
        "fit_matches_sealed": bool(fit_matches_sealed),
        "bootstrap_quantiles_match_sealed": bool(quantiles_match),
        "alpha_within_0.05_0.10": 0.05 <= fit["alpha"] <= 0.10,
        "interval_width_below_0.2": (shi - slo) < 0.2,
        "target_N_gt_2x_max_fit_N": pred["target_N"] > 2 * max(Ns),
        "prediction_older_than_first_holdout_start":
            sc.assert_prediction_before_run(pred["utc_timestamp"],
                                            first_start_unix) is None,
        "trained_mean_inside_interval": bool(inside),
    }
    for k, v in gates.items():
        print(f"gate {k}: {'PASS' if v else 'FAIL'}")

    # ---- figure ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    # excluded width 8 (bigram plateau)
    w8 = [losses[(8, 1)], losses[(8, 2)]]
    ax.scatter([sc.nonembedding_params(8)] * 2, w8, marker="x", s=60,
               color="#888888", label="width 8 (excluded, bigram plateau)")
    # seed points and seed means
    for i, w in enumerate(fit_widths):
        for s in (1, 2):
            ax.scatter(Ns[i], losses[(w, s)], s=18, color="#3a6ea5", alpha=0.6)
    ax.scatter(Ns, seed_means, s=55, color="#1a3f66", zorder=5,
               label="seed means (widths 12-128, fit set)")
    # fit curve and 95% bootstrap band from the regenerated draws
    rng = np.random.default_rng(pred["prediction"]["seed"])
    sig = fit["sigmas"] * fit["birge"]
    mu = fit["a"] * fit["Ns"] ** (-fit["alpha"])
    draws = np.empty((pred["prediction"]["n_draws"], len(Ns)))
    grid_N = np.logspace(np.log10(Ns.min() * 0.9), np.log10(pred["target_N"] * 1.1), 200)
    band = np.empty((pred["prediction"]["n_draws"], len(grid_N)))
    for i in range(pred["prediction"]["n_draws"]):
        Ls = rng.normal(mu, sig)
        f = sc.fit_wls(fit["Ns"], Ls, fit["sigmas"])
        band[i] = f["a"] * grid_N ** (-f["alpha"])
    blo, bhi = np.percentile(band, [2.5, 97.5], axis=0)
    ax.fill_between(grid_N, blo, bhi, alpha=0.18, color="#3a6ea5",
                    label="95% bootstrap band (regenerated, seed 20260)")
    ax.plot(grid_N, fit["a"] * grid_N ** (-fit["alpha"]), color="#c46a1f",
            label=f"pure power-law fit: alpha={fit['alpha']:.4f}, Birge={fit['birge']:.2f}")
    ax.axhline(BIGRAM, color="crimson", ls=":", label="bigram 2.4819")
    # the blind prediction and the actual holdout point
    n192 = pred["target_N"]
    ax.scatter([n192], [boot["pred_point"]], marker="D", s=70, color="green",
               zorder=6, label=f"sealed blind prediction {boot['pred_point']:.4f} "
                               f"[{slo:.4f}, {shi:.4f}] (saved {pred['utc_timestamp']})")
    ax.errorbar([n192], [hold_mean],
                yerr=[[hold_mean - min(hold_losses)], [max(hold_losses) - hold_mean]],
                fmt="o", color="black", capsize=4, zorder=7,
                label=f"trained width 192 mean {hold_mean:.4f} — OUTSIDE interval (MISS)")
    ax.axvspan(n192 * 0.98, n192 * 1.02, color="black", alpha=0.06)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("non-embedding parameters N")
    ax.set_ylabel("held-out loss (nats/char), 1500-step budget")
    ax.set_title("Part 3: scaling law and blind prediction at width 192\n"
                 "Trained loss falls outside the pre-recorded 95% interval", fontsize=12)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(EVID / "part3-scaling-blind-miss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- compact summary ---------------------------------------------------
    # Worker start is distinct from the earlier manager dispatch timestamp.
    summary = {
        "read_only": True,
        "fit_widths": fit_widths,
        "per_run_losses": {f"w{w}-s{s}": losses[(w, s)] for w in WIDTHS for s in (1, 2)},
        "seed_means": dict(zip(map(str, fit_widths), seed_means.tolist())),
        "sigmas_sem_floor1e-3": sigmas.tolist(),
        "alpha": fit["alpha"], "a": fit["a"],
        "chi2": fit["chi2"], "dof": fit["dof"], "birge": fit["birge"],
        "alpha_interval": boot["alpha_interval"],
        "sealed_prediction": {"utc_timestamp": pred["utc_timestamp"],
                              "point": boot["pred_point"], "interval": [slo, shi],
                              "interval_width": shi - slo,
                              "sha256": pred["sha256"]},
        "holdout": {"losses": hold_losses, "seed_mean": hold_mean,
                    "inside_interval": inside, "miss_nats": miss_nats,
                    "first_run_started_utc": datetime.fromtimestamp(first_start_unix, timezone.utc).isoformat(),
                    "dispatch_started_utc": mgr["started_utc"],
                    "first_run_started_unix": mgr["started_unix"],
                    "finished_unix": {s: hold[s]["finished_unix_ts"] for s in (1, 2)},
                    "finished_utc": {s: hold[s]["utc_finished"] for s in (1, 2)},
                    "metrics_sha256": {s: sha(HOLD / f"w192-s{s}" / "metrics.json")
                                       for s in (1, 2)}},
        "gates": gates,
        "protocol_hashes": {"prediction_file": sha(FIT / "prediction" / "prediction192.json"),
                            "fit_manifest_shas": [sha(FIT / f"fit-manifest-w{w:03d}-s{s}.json")
                                                  for w in WIDTHS for s in (1, 2)]},
        "interpretation_note": ("The pure power law was frozen before the run; width 192 was "
                                "never fitted. The trained losses fall ABOVE the interval — a "
                                "legitimate failed blind extrapolation. Plausible explanations "
                                "(compute-limited flattening at a fixed 1500-step budget) are "
                                "interpretation, not proven mechanism. The sheet's floor-term "
                                "Protocol A failure is COURSE REFERENCE HISTORY, not an "
                                "experiment performed here."),
    }
    (EVID / "part3-scaling-summary.json").write_text(json.dumps(
        summary, indent=2,
        default=lambda o: bool(o) if isinstance(o, np.bool_) else str(o)) + "\n")
    print("wrote evidence/part3-scaling-blind-miss.png and part3-scaling-summary.json")


if __name__ == "__main__":
    main()
