"""Exact lecture schedule counts and end-to-end replay invariants."""
import json
import numpy as np
import pytest
from test_seismic import ROOT, invoke, small_model, write_model
from test_derivatives import run_model


def inspect_actions(actions, steps, budget):
    saved = {0}
    workspace = None
    grads = []
    peak = 1
    calls = 0
    for item in actions:
        action, step = item["action"], item["step"]
        if action == "restore":
            assert step in saved
            workspace = step
        elif action == "call":
            assert workspace == step
            workspace += 1
            calls += 1
        elif action == "store":
            assert workspace == step and step not in saved
            saved.add(step)
        elif action == "grad":
            assert step in saved  # a workspace by itself does not authorize a VJP
            grads.append(step)
        elif action == "fetch":
            assert step != 0 and step in saved
            saved.remove(step)
        else:
            pytest.fail("unknown schedule action")
        assert item["saved_states"] == len(saved)
        peak = max(peak, len(saved))
        assert len(saved) <= budget + 1
    assert grads == list(range(steps - 1, -1, -1)) and saved == {0}
    return calls, peak


def test_six_steps_two_slots_have_lecture_schedule(tmp_path):
    model = small_model(7, 8, 6)
    model["perturbation"] = np.full((8, 7), .1).tolist()
    born = run_model(tmp_path, model, "born", "six-born")
    out = run_model(tmp_path, model, "adjoint", "six-tree", data=born / "born_data.npy",
                    storage="treeverse", checkpoints=2)
    actions = json.loads((out / "actions-0.json").read_text())
    assert [a["step"] for a in actions if a["action"] == "store"][0] == 3
    assert inspect_actions(actions, 6, 2) == (8, 3)


def test_zero_budget_and_misplaced_checkpoint_flags_are_rejected(tmp_path):
    model = small_model()
    weights = tmp_path / "weights.npy"
    np.save(weights, np.zeros((1, 12, 3)))
    path = write_model(tmp_path, model)
    bad = invoke("--experiment", path, "--mode", "adjoint", "--data", weights,
                 "--storage", "treeverse", "--checkpoints", 0, "--out", tmp_path / "zero", ok=False)
    assert "positive" in bad.stderr
    invoke("--experiment", path, "--mode", "forward", "--checkpoints", 2,
           "--out", tmp_path / "ignored-flag", ok=False)


@pytest.mark.parametrize("budget,calls,peak", [(1, 28680, 2), (3, 1695, 4), (5, 990, 6), (10, 642, 11)])
def test_reflector_treeverse_is_identical_to_full_and_respects_budget(tmp_path, budget, calls, peak):
    out = tmp_path / f"checkpoint-{budget}"
    invoke("--experiment", ROOT / "inputs/reflector.json", "--mode", "adjoint",
           "--data", ROOT / "artifacts/born/born_data.npy", "--storage", "treeverse",
           "--checkpoints", budget, "--out", out)
    np.testing.assert_array_equal(np.load(out / "image.npy"), np.load(ROOT / "artifacts/adjoint/image.npy"))
    result = json.loads((out / "result.json").read_text())["statistics"]
    assert result["peak_saved_states"] == peak
    assert result["peak_saved_bytes"] == peak * 2 * 41 * 41 * 8
    assert result["scheduler_forward_calls"] == calls * 3
    assert result["reverse_calls"] == 240 * 3
    for shot in result["per_shot"]:
        actions = json.loads((out / shot["actions_file"]).read_text())
        assert inspect_actions(actions, 240, budget) == (calls, peak)
