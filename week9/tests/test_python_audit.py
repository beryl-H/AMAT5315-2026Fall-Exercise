"""Synthetic structural regression checks; not experimental evidence."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('audit', ROOT / 'verify_and_plot.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def leaf(i):
    return {'Leaf': {'tensor_index': i}}


def node(a, b, inputs, output):
    return {'Node': {'args': [a, b], 'eins': {'ixs': inputs, 'iy': output}}}


class AuditChecks(unittest.TestCase):
    def test_hyperedge_kept_until_last_occurrence(self):
        p = {'dimensions': [2], 'input_labels': [[0], [0], [0]], 'output_labels': [],
             'tree': node(node(leaf(0), leaf(1), [[0], [0]], [0]), leaf(2), [[0], [0]], [])}
        c = audit.recount(p)
        self.assertEqual(c['work'], 4)
        self.assertEqual(c['largest_tensor'], 2)
        self.assertEqual(c['steps'][0]['retained_labels'], [0])
        p['tree']['Node']['args'][0]['Node']['eins']['iy'] = []
        with self.assertRaisesRegex(AssertionError, 'premature'):
            audit.recount(p)

    def test_duplicate_factor_is_rejected(self):
        p = {'dimensions': [2], 'input_labels': [[0], [0]], 'output_labels': [],
             'tree': node(leaf(0), leaf(0), [[0], [0]], [0])}
        with self.assertRaises(AssertionError):
            audit.recount(p)

    def test_asia_state_and_scope_order(self):
        b = audit.enumerate_asia(ROOT / 'asia.uai', {0: 0, 7: 0, 6: 0})
        self.assertAlmostEqual(b['evidence_mass'], .00098822675, places=15)
        self.assertAlmostEqual(b['joint_tl'][0, 0], .02221352538777158, places=14)
        self.assertAlmostEqual(float(b['marginals'][0, 0]), 1, places=14)

    def test_swapped_tb_scope_with_unchanged_values_is_rejected(self):
        """Negative control retained on disk: methods agreeing cannot prove input correctness."""
        import hashlib
        from datetime import datetime, timezone
        original = ROOT / 'asia.uai'
        before = original.read_bytes()
        words = before.decode().split()
        cursor = 1
        n = int(words[cursor]); cursor += 1 + n
        nf = int(words[cursor]); cursor += 1
        changed = False
        for i in range(nf):
            count = int(words[cursor]); cursor += 1
            if i == 1:
                self.assertEqual(words[cursor:cursor + count], ['1', '0'])
                words[cursor:cursor + count] = ['0', '1']
                changed = True
            cursor += count
        self.assertTrue(changed)
        folder = ROOT.parent / 'week9-support' / 'negative-controls'
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        altered = folder / f'asia-swapped-tb-scope-{stamp}.uai'
        altered.write_text('\n'.join(words) + '\n')
        # The deliberately incorrect copy is kept for audit; no cleanup/deletion.
        with self.assertRaises(AssertionError):
            audit.enumerate_asia(altered, {})
        self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(),
                         hashlib.sha256(before).hexdigest())


if __name__ == '__main__':
    unittest.main()
