"""Read a Part 1 run independently of the Rust solver."""

import json
import math
import os
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUN = Path(os.environ.get("AMAT5315_GAP_RUN", ROOT / "artifacts/gap/run.json"))


class GapEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_data = json.loads(RUN.read_text(encoding="utf-8"))

    def test_analytical_optima_and_rounding(self):
        workshop = self.run_data["workshop"]
        diet = self.run_data["diet"]["lp"]
        for solution, vector, value in (
            (workshop["lp"], [7 / 3, 8 / 3], 83 / 3),
            (workshop["mip"], [3, 2], 27),
            (diet, [89 / 28, 8 / 7], 323 / 14),
        ):
            self.assertAlmostEqual(solution["value"], value, delta=1e-4)
            for actual, expected in zip(solution["x"], vector, strict=True):
                self.assertAlmostEqual(actual, expected, delta=1e-4)
        self.assertEqual(workshop["rounded"], [2, 3])
        self.assertFalse(workshop["rounded_feasible"])
        self.assertEqual(workshop["rounded_work_hours"], 29)
        self.assertEqual(workshop["floor"], [2, 2])
        self.assertEqual(workshop["floor_value"], 22)

    def test_covering_instances_and_repair(self):
        batch = self.run_data["instances"]
        self.assertEqual([instance["id"] for instance in batch], list(range(16)))
        gaps, failures = [], 0
        for instance in batch:
            with self.subTest(id=instance["id"]):
                problem = instance["problem"]
                rows = [[j for j, coefficient in row if coefficient == 1] for row in problem["rows"]]
                costs = problem["costs"]
                self.assertEqual((len(costs), len(rows)), (32, 24))
                degree = [0] * len(costs)
                for row, raw in zip(rows, problem["rows"], strict=True):
                    self.assertGreaterEqual(len(row), 3)
                    self.assertEqual(len(row), len(set(row)))
                    self.assertEqual(len(row), len(raw))
                    for j in row:
                        self.assertIn(j, range(len(costs)))
                        degree[j] += 1
                for cost, d in zip(costs, degree, strict=True):
                    self.assertGreaterEqual(d, 2)
                    self.assertIn(cost - 2 * d - 1, range(4))

                lp, mip = instance["lp"], instance["mip"]
                for solution in (lp, mip):
                    vector = solution["x"]
                    self.assertEqual(len(vector), len(costs))
                    self.assertTrue(all(-1e-6 <= x <= 1 + 1e-6 for x in vector))
                    self.assertTrue(all(sum(vector[j] for j in row) >= 1 - 1e-6 for row in rows))
                    self.assertAlmostEqual(
                        solution["value"], math.fsum(c * x for c, x in zip(costs, vector)), delta=1e-5
                    )
                self.assertTrue(all(abs(x - round(x)) <= 1e-6 for x in mip["x"]))
                self.assertLessEqual(lp["value"], mip["value"] + 1e-6)
                gap = (mip["value"] - lp["value"]) / mip["value"]
                self.assertAlmostEqual(instance["relative_gap"], gap, delta=1e-6)
                gaps.append(gap)

                rounded = [int(x >= 0.5) for x in lp["x"]]
                self.assertEqual(instance["rounded"], rounded)
                uncovered = [i for i, row in enumerate(rows) if not any(rounded[j] for j in row)]
                self.assertEqual(instance["uncovered_rows"], uncovered)
                self.assertEqual(instance["rounded_feasible"], not uncovered)
                failures += bool(uncovered)
                repaired = rounded.copy()
                for row in rows:
                    if not any(repaired[j] for j in row):
                        j = min(row, key=lambda j: (costs[j], j))
                        repaired[j] = 1
                self.assertTrue(all(any(repaired[j] for j in row) for row in rows))
                self.assertEqual(instance["repaired"], repaired)
                actual_cost = sum(c * x for c, x in zip(costs, repaired, strict=True))
                self.assertAlmostEqual(instance["repaired_cost"], actual_cost, delta=1e-6)
                self.assertGreaterEqual(actual_cost, mip["value"] - 1e-6)
                self.assertAlmostEqual(
                    instance["rounding_overhead"], (actual_cost - mip["value"]) / mip["value"], delta=1e-6
                )
        summary = self.run_data["summary"]
        self.assertEqual(summary["rounding_infeasible"], failures)
        self.assertAlmostEqual(summary["mean_relative_gap"], math.fsum(gaps) / 16, delta=1e-6)
