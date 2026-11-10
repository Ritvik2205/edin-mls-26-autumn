import importlib.util
import unittest
from pathlib import Path

from extensions.analysis.generate_from_cache import build_candidate
from extensions.analysis.select_ids import select_ids

RAG = Path(__file__).resolve().parents[3]


def supplied_validator():
    spec = importlib.util.spec_from_file_location("evaluate_answers", RAG / "scripts" / "evaluate_answers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.validate_candidates


class CandidateTests(unittest.TestCase):
    def test_select_ids_differing_plus_controls(self):
        ids = select_ids({"a": "x", "b": "y", "c": "z"}, {"a": "x", "b": "Y!", "c": "z"}, n_controls=1)
        self.assertIn("b", ids)
        self.assertEqual(len(ids), 2)

    def test_select_ids_deterministic_and_sorted(self):
        v1 = {f"q{i}": "same" for i in range(20)}
        var = dict(v1, q3="changed")
        a, b = select_ids(v1, var, n_controls=5, seed=0), select_ids(v1, var, n_controls=5, seed=0)
        self.assertEqual(a, b)
        self.assertEqual(a, sorted(a))
        self.assertEqual(len(a), 6)

    def test_select_ids_ignores_ids_missing_from_either(self):
        self.assertEqual(select_ids({"a": "x", "b": "y"}, {"a": "X"}, n_controls=3), ["a"])

    def test_candidate_uses_v1_question(self):
        row = build_candidate("q1", "dev", "V1 text", "variant text", "ans", ["r1"], {})
        self.assertEqual(row["transcript"], "V1 text")
        self.assertEqual(row["asr_transcript"], "variant text")

    def test_validates_with_supplied_checker(self):
        row = build_candidate("q1", "dev", "V1 text", "variant text", "ans", ["r1"], {"generation_ms": 1.0})
        seen = supplied_validator()([row], {"q1": {}, "q2": {}}, "dev", allow_subset=True)
        self.assertEqual(seen, {"q1"})


if __name__ == "__main__":
    unittest.main()
