"""Independently replay action metadata; never trust reported scheduler counters."""
import argparse
import collections
import json


def audit(actions, steps, budget):
    names = ["missing_reverse_steps", "duplicated_reverse_steps",
             "reverse_steps_out_of_descending_order", "invalid_restores", "budget_overruns",
             "invalid_grad_inputs", "invalid_calls", "invalid_stores", "invalid_fetches",
             "saved_state_count_mismatches", "invalid_actions", "unfinished_saved_slots"]
    violations = dict.fromkeys(names, 0)
    saved = {0}
    workspace = None
    reverse_steps = []
    calls = 0
    peak = 1
    for event in actions:
        action, step = event["action"], event["step"]
        if action == "restore":
            if step not in saved:
                violations["invalid_restores"] += 1
            workspace = step
        elif action == "call":
            if workspace != step or not 0 <= step < steps:
                violations["invalid_calls"] += 1
            workspace = step + 1
            calls += 1
        elif action == "store":
            if workspace != step or step in saved or not 0 < step <= steps:
                violations["invalid_stores"] += 1
            saved.add(step)
        elif action == "grad":
            if step not in saved or not 0 <= step < steps:
                violations["invalid_grad_inputs"] += 1
            reverse_steps.append(step)
        elif action == "fetch":
            if step == 0 or step not in saved:
                violations["invalid_fetches"] += 1
            elif step in saved:
                saved.remove(step)
        else:
            violations["invalid_actions"] += 1
        if event["saved_states"] != len(saved):
            violations["saved_state_count_mismatches"] += 1
        if len(saved) > budget + 1:
            violations["budget_overruns"] += 1
        peak = max(peak, len(saved))
    occurrences = collections.Counter(reverse_steps)
    violations["missing_reverse_steps"] = sum(i not in occurrences for i in range(steps))
    violations["duplicated_reverse_steps"] = sum(max(0, count - 1) for count in occurrences.values())
    violations["reverse_steps_out_of_descending_order"] = sum(
        value != steps - 1 - i for i, value in enumerate(reverse_steps))
    violations["unfinished_saved_slots"] = int(saved != {0})
    return dict(violations=violations, scheduler_forward_calls=calls,
                reverse_calls=len(reverse_steps), peak_saved_states=peak)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--budget", type=int, required=True)
    args = parser.parse_args()
    with open(args.path) as file:
        result = audit(json.load(file), args.steps, args.budget)
    print(json.dumps(result, indent=2))
    raise SystemExit(int(any(result["violations"].values())))
