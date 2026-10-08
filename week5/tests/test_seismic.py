"""Black-box checks derived from lecture equations and printed references."""
import json
import math
import os
import subprocess
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("SEISMIC_BIN", ROOT / "target/release/seismic"))


def invoke(*args, ok=True):
    assert BIN.is_file(), "seismic executable is not implemented yet"
    result = subprocess.run([str(BIN), *map(str, args)], text=True, capture_output=True)
    if ok:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
    return result


def small_model(nx=7, nz=8, steps=12):
    return dict(name="small", nx=nx, nz=nz, dx=1., dt=.15, steps=steps,
                source_frequency=.7, source_peak_time=.5, source_amplitude=1.,
                shots=[[3., 3.]], receivers=[[2, 3], [4, 3], [3, 4]],
                sponge_width=2, sponge_strength=.4, length_unit_m=100., time_unit_s=.1,
                background=np.full((nz, nx), 1.4).tolist(),
                perturbation=np.zeros((nz, nx)).tolist())


def write_model(tmp_path, model, name="model.json"):
    path = tmp_path / name
    path.write_text(json.dumps(model))
    return path


def test_enzyme_smoke_is_cube_value_and_both_derivatives():
    values = json.loads(invoke("--enzyme-smoke").stdout)
    assert values == [8., 12., 12.]


def test_one_cell_centered_update_and_zero_boundaries(tmp_path):
    model = small_model(3, 3, 2)
    model.update(dt=.2, source_frequency=1., source_peak_time=0.,
                 shots=[[1., 1.]], receivers=[[1, 1], [0, 0]], sponge_width=0)
    model["background"] = np.ones((3, 3)).tolist()
    path = write_model(tmp_path, model)
    out = tmp_path / "forward"
    invoke("--experiment", path, "--mode", "forward", "--every", 1, "--out", out)
    a = np.load(out / "traces.npy")
    theta = math.pi * .2
    pulse = (1 - 2 * theta * theta) * math.exp(-theta * theta)
    np.testing.assert_allclose(a[0, :, 0], [.04, .0736 + .04 * pulse], rtol=0, atol=1e-14)
    assert np.all(a[0, :, 1] == 0)
    assert a.dtype == np.float64
    wave = np.load(out / "wavefield.npy")
    assert wave.dtype == np.float32 and wave.shape == (2, 3, 3)
    assert not np.any(wave[:, :, 0]) and not np.any(wave[:, 0, :])
    meta = json.loads((out / "run.json").read_text())
    assert meta["recording"]["steps"] == [1, 2]
    assert meta["recording"]["times"] == [.2, .4]
    assert "background" not in meta["experiment"]
    assert "perturbation" not in meta["experiment"]


def test_sponge_divides_the_update(tmp_path):
    model = small_model(3, 3, 1)
    model.update(dt=.2, source_frequency=1., source_peak_time=0.,
                 shots=[[1., 1.]], receivers=[[1, 1]], sponge_width=2, sponge_strength=.4)
    path = write_model(tmp_path, model)
    out = tmp_path / "damped"
    invoke("--experiment", path, "--mode", "forward", "--out", out)
    assert np.load(out / "traces.npy")[0, 0, 0] == pytest.approx(.04 / 1.02, abs=1e-14)


def test_unstable_cfl_is_rejected(tmp_path):
    model = small_model()
    model["dt"] = 2.
    r = invoke("--experiment", write_model(tmp_path, model), "--mode", "forward",
               "--out", tmp_path / "invalid", ok=False)
    assert "CFL" in r.stderr


def test_far_ricker_tail_stays_finite(tmp_path):
    model = small_model()
    model["source_peak_time"] = 1e200
    out = tmp_path / "far-tail"
    invoke("--experiment", write_model(tmp_path, model), "--mode", "forward", "--out", out)
    data = np.load(out / "traces.npy")
    assert np.all(np.isfinite(data)) and not np.any(data)


def test_unrecorded_rerun_cannot_leave_a_stale_wavefield(tmp_path):
    path = write_model(tmp_path, small_model())
    out = tmp_path / "recorded"
    invoke("--experiment", path, "--mode", "forward", "--every", 1, "--out", out)
    before = (out / "run.json").read_bytes()
    result = invoke("--experiment", path, "--mode", "forward", "--out", out, ok=False)
    assert "wavefield" in result.stderr
    assert (out / "run.json").read_bytes() == before


@pytest.fixture(scope="module")
def reflector_forward(tmp_path_factory):
    out = tmp_path_factory.mktemp("reflector-forward")
    invoke("--experiment", ROOT / "inputs/reflector.json", "--mode", "forward",
           "--every", 3, "--out", out)
    return out


def test_reflector_gathers_match_lecture(reflector_forward):
    data = np.load(reflector_forward / "traces.npy")
    assert data.shape == (3, 240, 14)
    assert np.linalg.norm(data) == pytest.approx(11.574770, rel=1e-4)
    for shot, expected in enumerate([.60809514, .59271397, .60809514]):
        step, receiver = np.unravel_index(np.abs(data[shot]).argmax(), data[shot].shape)
        assert step == 83
        assert abs(data[shot, step, receiver]) == pytest.approx(expected, rel=1e-4)
        rx = json.loads((ROOT / "inputs/reflector.json").read_text())["receivers"][receiver][0]
        sx = [10, 20, 30][shot]
        assert abs(rx - sx) == 1


def test_diagonal_wave_speed_matches_lecture(reflector_forward):
    data = np.load(reflector_forward / "wavefield.npy")
    meta = json.loads((reflector_forward / "run.json").read_text())["recording"]
    peaks = []
    for step in [108, 144]:
        frame = data[meta["steps"].index(step)]
        line = np.array([frame[8 + k, 10 + k] for k in range(24)])
        peaks.append(line.argmax() * .1 * math.sqrt(2))
    assert (144 - 108) * .2 * .1 == pytest.approx(.72)
    assert peaks[1] - peaks[0] == pytest.approx(1.296, abs=.1 * math.sqrt(2))
