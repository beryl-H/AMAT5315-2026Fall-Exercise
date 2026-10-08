"""Check actual numerical arrays and independently audit every recorded replay."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
sys.path.insert(0, str(ROOT))
import lj_ad
from audit_actions import audit


def read(path):
    return json.loads(Path(path).read_text())


def close(actual, expected, tolerance):
    relative = abs(actual - expected) / abs(expected)
    assert relative < tolerance, (actual, expected, relative, tolerance)
    return relative


def audit_run(folder, steps, budget, expected_calls):
    stats = read(folder / "result.json")["statistics"]
    reports = []
    for i, shot in enumerate(stats["per_shot"]):
        path = folder / f"actions-{i}.json"
        actual = audit(read(path), steps, budget)
        assert not any(actual["violations"].values()), (str(path), actual)
        assert actual["scheduler_forward_calls"] == expected_calls
        assert actual["reverse_calls"] == steps
        assert actual["peak_saved_states"] == budget + 1
        for key in ["scheduler_forward_calls", "reverse_calls", "peak_saved_states"]:
            assert actual[key] == shot[key], (str(path), key)
        actual["file"] = str(path.relative_to(ROOT))
        reports.append(actual)
        print(f"AUDIT {actual['file']}: {json.dumps(actual['violations'])}")
    assert stats["scheduler_forward_calls"] == expected_calls * len(reports)
    assert stats["reverse_calls"] == steps * len(reports)
    assert stats["peak_saved_states"] == budget + 1
    return reports


def main():
    report = {"checks_passed": False, "inner_product": "unweighted Euclidean sums",
              "parameter": "velocity", "saved_memory_scope": "two float64 wavefields per saved state"}
    deriv = read(ART / "ad/derivatives.json")
    analytic = -48 * deriv["r"]**-13 + 24 * deriv["r"]**-7
    single_errors = {name: abs(value-analytic) for name, value in {
        "forward": deriv["tangents"]["U"], "reverse": deriv["adjoints"]["r"],
        "jax": deriv["jax_grad"]}.items()}
    assert max(single_errors.values()) < 1e-12
    r = np.linspace(.95, 2.5, 601)
    reference = -48*r**-13 + 24*r**-7
    forward = np.array([lj_ad.forward_tangents(float(x))[1]["U"] for x in r])
    reverse = np.array([lj_ad.reverse_adjoints(float(x))[1]["r"] for x in r])
    h = 1e-6
    energy = lambda x: 4*(x**-12 - x**-6)
    fd = (energy(r+h)-energy(r-h))/(2*h)
    errors = {name: float(np.max(np.abs(value-reference))) for name, value in
              {"forward": forward, "reverse": reverse, "finite_difference": fd}.items()}
    assert max(errors["forward"], errors["reverse"]) < 1e-12
    assert errors["finite_difference"] > max(errors["forward"], errors["reverse"])
    report["part1"] = dict(r=deriv["r"], single_point_absolute_errors=single_errors,
                           grid_points=601, finite_difference_h=h, grid_max_absolute_errors=errors)
    print(f"AD grid maximum errors: {errors}")

    model = read(ROOT / "inputs/reflector.json")
    traces = np.load(ART / "forward/traces.npy")
    assert traces.dtype == np.float64 and traces.shape == (3, 240, 14)
    norm = float(np.linalg.norm(traces))
    norm_error = close(norm, 11.574770, 1e-4)
    peaks = []
    for i, expected in enumerate([.60809514, .59271397, .60809514]):
        time, receiver = np.unravel_index(np.abs(traces[i]).argmax(), traces[i].shape)
        amplitude = float(abs(traces[i, time, receiver]))
        close(amplitude, expected, 1e-4)
        assert time == 83
        assert abs(model["receivers"][receiver][0]-model["shots"][i][0]) == 1
        peaks.append(dict(shot=i, trace_index=int(time), receiver=int(receiver), absolute_peak=amplitude))
    frames = np.load(ART / "forward/wavefield.npy")
    metadata = read(ART / "forward/run.json")["recording"]
    assert frames.dtype == np.float32 and metadata["steps"] == list(range(3, 241, 3))
    assert not np.any(frames[:, [0, -1], :]) and not np.any(frames[:, :, [0, -1]])
    diagonal = np.arange(24)
    peak_distances = []
    for step in [108, 144]:
        frame = frames[metadata["steps"].index(step)]
        peak = int(np.argmax(frame[8+diagonal, 10+diagonal]))
        peak_distances.append(peak*np.sqrt(2)*.1)
    displacement = peak_distances[1]-peak_distances[0]
    assert abs(displacement-1.296) <= .1*np.sqrt(2)
    report["forward"] = dict(trace_l2=norm, reference_relative_error=norm_error, peaks=peaks,
                              wavefront_peak_distances_km=peak_distances,
                              measured_displacement_km=displacement, expected_displacement_km=1.296)
    print(f"Forward L2={norm:.12f}; diagonal displacement={displacement:.12f} km")

    d = np.load(ART / "born/born_data.npy")
    image = np.load(ART / "adjoint/image.npy")
    assert d.dtype == image.dtype == np.float64 and d.shape == traces.shape and image.shape == (41, 41)
    left = float(np.vdot(d, d))
    right = float(np.vdot(np.array(model["perturbation"]), image))
    dot_error = abs(left-right)/max(abs(left), abs(right))
    assert dot_error < 1e-9
    close(left, .0348479, 1e-4)
    profile = np.linalg.norm(image[10:34, 7:34], axis=1)
    peak_depth = 10 + int(profile.argmax())
    assert abs(peak_depth-21) <= 1
    adj_steps = read(ART / "adjoint/run.json")["recording"]["steps"]
    assert adj_steps == list(range(237, -1, -3)) and 132 in adj_steps
    full = read(ART / "adjoint/result.json")["statistics"]
    assert full["state_wavefields"] == 2 and full["peak_saved_states"] == 241
    assert full["saved_state_bytes"] == 26896 and full["peak_saved_bytes"] == 6481936
    assert all(s["scheduler_forward_calls"] == s["reverse_calls"] == 240 for s in full["per_shot"])
    report["derivatives"] = dict(born_squared_l2=left, perturbation_dot_image=right,
                                  dot_relative_error=dot_error, peak_depth_cell=peak_depth,
                                  full_history_statistics=full)
    print(f"Transpose: d·d={left:.15e}, m·image={right:.15e}, relative error={dot_error:.3e}; depth cell={peak_depth}")

    checkpoint_reports = []
    for budget, calls in [(1, 28680), (3, 1695), (5, 990), (10, 642)]:
        folder = ART / f"checkpoint-{budget}"
        candidate = np.load(folder / "image.npy")
        relative = float(np.linalg.norm(candidate-image)/np.linalg.norm(image))
        assert relative < 1e-9 and np.array_equal(candidate, image)
        statistics = read(folder / "result.json")["statistics"]
        assert statistics["state_wavefields"] == 2
        assert statistics["saved_state_bytes"] == 26896
        assert statistics["peak_saved_bytes"] == (budget+1)*26896
        audits = audit_run(folder, 240, budget, calls)
        checkpoint_reports.append(dict(budget=budget, image_relative_l2_error=relative,
                                       bitwise_identical=True, statistics=statistics, audits=audits))
        print(f"CHECKPOINT budget={budget}, relative image L2={relative}, peak={budget+1}, calls/shot={calls}")
    report["checkpointing"] = checkpoint_reports

    marm = read(ROOT / "inputs/marmousi.json")
    born = np.load(ART / "marmousi-born/born_data.npy")
    migrated = np.load(ART / "marmousi-image/image.npy")
    assert born.shape == (9, 1200, 91) and migrated.shape == (269, 805)
    assert born.dtype == migrated.dtype == np.float64
    marm_norm = float(np.linalg.norm(migrated))
    marm_error = close(marm_norm, 6.7037741e-4, 1e-4)
    stats = read(ART / "marmousi-image/result.json")["statistics"]
    assert stats["state_wavefields"] == 2 and stats["saved_state_bytes"] == 3464720
    assert stats["peak_saved_states"] == 6 and stats["peak_saved_bytes"] == 20788320
    assert stats["storage"] == "treeverse" and stats["checkpoints"] == 5
    audits = audit_run(ART / "marmousi-image", 1200, 5, 7884)
    report["marmousi"] = dict(image_l2=marm_norm, reference_relative_error=marm_error,
                              statistics=stats, theoretical_full_history_bytes=1201*2*805*269*8,
                              audits=audits, image_depth_gain_applied=False)
    print(f"Marmousi image L2={marm_norm:.12e}; peak={stats['peak_saved_states']} states, {stats['peak_saved_bytes']:,} bytes")

    fingerprints = {}
    for name, expected in [("reflector", "22b84948e28e8e13e958f47b2d4bf9a70a3d0c8ee6ffcaab3cdeb8003b9902a6"),
                            ("marmousi", "67beabdee96d8e3f6433c2f897b3b24b0c744e7808784148bb06310665d888b0")]:
        digest = hashlib.sha256((ROOT / f"inputs/{name}.json").read_bytes()).hexdigest()
        assert digest == expected
        fingerprints[name] = digest
    for path in ART.glob("*/run.json"):
        metadata = read(path)
        assert metadata["experiment_sha256"] == fingerprints[metadata["experiment"]["name"]]
        assert "background" not in metadata["experiment"] and "perturbation" not in metadata["experiment"]
    report["input_sha256"] = fingerprints

    expected_pngs = ["inputs.png", "ad/modes.png", "ad/graph.png", "ad/grad-graph.png",
                     "forward/gathers.png", "forward/wavefield.png", "forward/wave-speed.png",
                     "adjoint/wavefield.png", "adjoint/image.png", "checkpoint-actions.png",
                     "checkpoint-work.png", "marmousi.png"]
    sizes = {str(path.relative_to(ROOT)): path.stat().st_size for path in ART.rglob("*")
             if path.is_file() and path.suffix in [".png", ".json"] and path.name != "verification.json"}
    assert all((ART/path).is_file() for path in expected_pngs)
    assert all(size < 5_000_000 for size in sizes.values()), sizes
    report["evidence_file_sizes_bytes"] = sizes
    report["checks_passed"] = True
    (ART / "verification.json").write_text(json.dumps(report, indent=2)+"\n")
    print("All numerical, fingerprint, replay and evidence-size checks passed.")


if __name__ == "__main__":
    main()
