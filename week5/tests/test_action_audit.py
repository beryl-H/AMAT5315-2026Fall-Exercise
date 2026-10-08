"""Mutations prove the action audit catches failures rather than just reporting zero."""
import copy
import importlib.util
from pathlib import Path


def auditor():
    path = Path(__file__).resolve().parents[1] / "scripts/audit_actions.py"
    assert path.exists(), "action auditor is not implemented yet"
    spec = importlib.util.spec_from_file_location("audit_actions", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.audit


def valid_actions():
    values = [("restore", 0, 1), ("call", 0, 1), ("call", 1, 1), ("store", 2, 2),
              ("grad", 2, 2), ("fetch", 2, 1), ("restore", 0, 1), ("call", 0, 1),
              ("store", 1, 2), ("grad", 1, 2), ("fetch", 1, 1), ("grad", 0, 1)]
    return [dict(action=a, step=s, saved_states=n) for a, s, n in values]


def test_valid_schedule_has_zero_audit_counts():
    result = auditor()(valid_actions(), 3, 1)
    assert result["scheduler_forward_calls"] == 3
    assert result["reverse_calls"] == 3 and result["peak_saved_states"] == 2
    assert all(v == 0 for v in result["violations"].values())


def test_missing_duplicate_and_out_of_order_reverse_steps_are_detected():
    audit = auditor()
    a = valid_actions()
    dropped = [x for x in a if not (x["action"] == "grad" and x["step"] == 1)]
    assert audit(dropped, 3, 1)["violations"]["missing_reverse_steps"] == 1
    duplicated = a[:5] + [copy.deepcopy(a[4])] + a[5:]
    assert audit(duplicated, 3, 1)["violations"]["duplicated_reverse_steps"] == 1
    a[4]["step"], a[9]["step"] = a[9]["step"], a[4]["step"]
    assert audit(a, 3, 1)["violations"]["reverse_steps_out_of_descending_order"] > 0


def test_invalid_restore_budget_and_workspace_are_detected():
    audit = auditor()
    a = valid_actions()
    a[0]["step"] = 99
    assert audit(a, 3, 1)["violations"]["invalid_restores"] == 1
    assert audit(valid_actions(), 3, 0)["violations"]["budget_overruns"] > 0
    b = valid_actions()
    b[1]["step"] = 1
    assert audit(b, 3, 1)["violations"]["invalid_calls"] > 0
    c = valid_actions()
    c[4]["step"] = 1
    assert audit(c, 3, 1)["violations"]["invalid_grad_inputs"] > 0
