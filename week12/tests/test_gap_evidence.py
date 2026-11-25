"""Check the revised sheet's gap benchmark independently of branchopt."""
import json
from pathlib import Path
from statistics import mean
import unittest

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


def energy(graph, bits):
    spins = [1 - 2 * bit for bit in bits]
    return sum(b["j"] * spins[b["u"]] * spins[b["v"]] for b in graph["bonds"]) + sum(
        h * spin for h, spin in zip(graph["fields"], spins, strict=True)
    )


class GapEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_data = json.loads((ARTIFACTS / "gap/run.json").read_text(encoding="utf-8"))

    def test_analytical_examples(self):
        square = self.run_data["square"]
        self.assertEqual(square["lp"]["x"][:4], [0.5] * 4)
        self.assertAlmostEqual(square["lp"]["value"], -4)
        self.assertAlmostEqual(square["mip"]["value"], -2)
        self.assertAlmostEqual(square["rounded_energy"], 2)
        self.assertAlmostEqual(self.run_data["diet"]["value"], 323 / 14)

    def test_seeded_instances(self):
        cases = self.run_data["instances"]
        self.assertEqual(len(cases), 16)
        for case in cases:
            with self.subTest(id=case["id"]):
                graph = case["graph"]
                bonds = [(b["u"], b["v"]) for b in graph["bonds"]]
                self.assertEqual(bonds, sorted(set(bonds)))
                self.assertEqual(len(bonds), 52)
                self.assertTrue(all(0 <= u < v < 22 for u, v in bonds))
                self.assertTrue(all(sum(i in edge for edge in bonds) >= 2 for i in range(22)))
                self.assertAlmostEqual(case["lp"]["value"], -52, places=3)
                self.assertEqual(case["fractional_spins"], 22)
                self.assertEqual(case["rounded_bits"], [1] * 22)
                self.assertAlmostEqual(case["mip"]["value"], energy(graph, case["ground_bits"]))
                self.assertAlmostEqual(case["rounded_energy"], energy(graph, case["rounded_bits"]))
                self.assertAlmostEqual(case["gap"], case["mip"]["value"] - case["lp"]["value"])
                self.assertAlmostEqual(case["rounding_excess"], case["rounded_energy"] - case["mip"]["value"])
                self.assertGreater(case["rounding_excess"], 0)
        self.assertAlmostEqual(mean(c["mip"]["value"] for c in cases), -38.25, places=3)
        self.assertAlmostEqual(mean(c["gap"] for c in cases), 13.75, places=3)
        self.assertAlmostEqual(mean(c["rounding_excess"] for c in cases), 34.25, places=3)
        self.assertAlmostEqual(cases[11]["rounding_excess"], 60)
