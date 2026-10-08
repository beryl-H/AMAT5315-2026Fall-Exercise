"""Derivative checks independent of the generated timestep derivative code."""
import copy
import json
import numpy as np
import pytest
from test_seismic import ROOT, invoke, small_model, write_model


def run_model(tmp_path, model, mode, name, **flags):
    path = write_model(tmp_path, model, name + ".json")
    out = tmp_path / name
    args = ["--experiment", path, "--mode", mode, "--out", out]
    for key, value in flags.items():
        args.extend(["--" + key, value])
    invoke(*args)
    return out


def test_born_is_velocity_directional_derivative_not_squared_speed(tmp_path):
    rng = np.random.default_rng(5315)
    model = small_model(7, 8, 20)
    perturbation = rng.normal(size=(8, 7)) * .2
    model["perturbation"] = perturbation.tolist()
    born = run_model(tmp_path, model, "born", "born")
    data = np.load(born / "born_data.npy")
    assert data.dtype == np.float64 and data.shape == (1, 20, 3)
    h = 1e-5
    traces = []
    for sign in [-1, 1]:
        changed = copy.deepcopy(model)
        changed["background"] = (np.array(model["background"]) + sign * h * perturbation).tolist()
        out = run_model(tmp_path, changed, "forward", "minus" if sign < 0 else "plus")
        traces.append(np.load(out / "traces.npy"))
    fd = (traces[1] - traces[0]) / (2 * h)
    assert np.linalg.norm(data) > 1e-5
    assert np.linalg.norm(data - fd) / np.linalg.norm(fd) < 1e-7


def test_random_weight_transpose_and_duplicate_receiver_injection(tmp_path):
    rng = np.random.default_rng(8471)
    model = small_model(7, 8, 20)
    model["shots"] = [[2., 3.], [4., 2.]]
    model["receivers"] = [[2, 3], [2, 3], [4, 3], [3, 4]]
    m = rng.normal(size=(8, 7)) * .2
    model["perturbation"] = m.tolist()
    born = run_model(tmp_path, model, "born", "random-born")
    d = np.load(born / "born_data.npy")
    w = rng.normal(size=d.shape)
    weights = tmp_path / "weights.npy"
    np.save(weights, w)
    adj = run_model(tmp_path, model, "adjoint", "random-adjoint", data=weights, every=2)
    image = np.load(adj / "image.npy")
    left = np.vdot(d, w)
    right = np.vdot(m, image)
    assert abs(left - right) / max(abs(left), abs(right)) < 1e-10
    recorded = np.load(adj / "wavefield.npy")
    assert not np.any(recorded[:, 0, :]) and not np.any(recorded[:, :, 0])
    assert not np.any(recorded[:, -1, :]) and not np.any(recorded[:, :, -1])
    assert not np.any(image[0, :]) and not np.any(image[:, 0])
    meta = json.loads((adj / "run.json").read_text())
    assert meta["recording"]["steps"] == list(range(18, -1, -2))
    counters = json.loads((adj / "result.json").read_text())["statistics"]
    assert counters["state_wavefields"] == 2
    assert counters["peak_saved_states"] == 21
    assert counters["saved_state_bytes"] == 2 * 7 * 8 * 8
    assert counters["scheduler_forward_calls"] == 40 and counters["reverse_calls"] == 40


def test_adjoint_rejects_weights_of_wrong_shape(tmp_path):
    model = small_model()
    weights = tmp_path / "bad.npy"
    np.save(weights, np.zeros((1, 12, 2)))
    result = invoke("--experiment", write_model(tmp_path, model), "--mode", "adjoint",
                    "--data", weights, "--out", tmp_path / "bad-out", ok=False)
    assert "shape" in result.stderr


@pytest.fixture(scope="module")
def reflector_derivatives(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("reflector-derivatives")
    born, adj = tmp / "born", tmp / "adjoint"
    invoke("--experiment", ROOT / "inputs/reflector.json", "--mode", "born", "--out", born)
    invoke("--experiment", ROOT / "inputs/reflector.json", "--mode", "adjoint",
           "--data", born / "born_data.npy", "--every", 3, "--out", adj)
    return born, adj


def test_reflector_dot_product_and_localization(reflector_derivatives):
    born, adj = reflector_derivatives
    d = np.load(born / "born_data.npy")
    image = np.load(adj / "image.npy")
    model = json.loads((ROOT / "inputs/reflector.json").read_text())
    left = np.vdot(d, d)
    right = np.vdot(np.array(model["perturbation"]), image)
    assert left == pytest.approx(.0348479, rel=1e-4)
    assert abs(left - right) / max(abs(left), abs(right)) < 1e-9
    profile = np.linalg.norm(image[10:34, 7:34], axis=1)
    assert abs(10 + profile.argmax() - 21) <= 1
    counters = json.loads((adj / "result.json").read_text())["statistics"]
    assert counters["peak_saved_states"] == 241
    assert counters["peak_saved_bytes"] == 6481936
