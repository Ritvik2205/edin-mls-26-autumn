import csv
import tempfile
import unittest
from pathlib import Path

from extensions.analysis.summarise import (load_runs, main_rows, paired_bootstrap_ci,
                                           parse_retrieval, stats, variant_table)
from extensions.fast_asr.bench_asr import COLUMNS


def write_csv(path, gpu, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            base = {c: "0" for c in COLUMNS}
            base.update({"gpu": gpu, "split": "dev", "variant": "v1", "setting": "full", "transcript": "t"})
            base.update(r)
            w.writerow(base)


RETRIEVAL_OUT = """[  1/2] q1 R@10=0.500
[  2/2] q2 R@10=1.000

Aggregate:
  Recall@5   0.600000
  Recall@10  0.750000
  Recall@20  0.800000
  Recall@100 0.900000
  MRR@10     0.550000
  NDCG@10    0.600000
"""


class SummariseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_refuses_mixed_gpus(self):
        a, b = self.d / "a.csv", self.d / "b.csv"
        write_csv(a, "NVIDIA GeForce RTX 2080 Ti", [{"query_id": "q1"}])
        write_csv(b, "NVIDIA RTX A6000", [{"query_id": "q1"}])
        with self.assertRaisesRegex(ValueError, "RTX 2080 Ti.*A6000"):
            load_runs([a, b])

    def test_stats_percentiles(self):
        s = stats([1, 2, 3, 4, 100])
        self.assertEqual(s["p50"], 3)
        self.assertEqual(s["n"], 5)
        self.assertAlmostEqual(s["mean"], 22.0)

    def test_bootstrap_deterministic_and_brackets(self):
        d, lo, hi = paired_bootstrap_ci([1, 1, 1, 1], [2, 2, 2, 2])
        self.assertEqual((d, lo, hi), (1.0, 1.0, 1.0))
        r1 = paired_bootstrap_ci([0, 1, 0, 1, 1], [1, 1, 0, 1, 0])
        self.assertEqual(r1, paired_bootstrap_ci([0, 1, 0, 1, 1], [1, 1, 0, 1, 0]))
        self.assertLessEqual(r1[1], r1[0])
        self.assertLessEqual(r1[0], r1[2])

    def test_repeat_rows_excluded_from_main_stats(self):
        p = self.d / "a.csv"
        write_csv(p, "G", [{"query_id": "q1", "pass": "0", "asr_ms": "10"},
                           {"query_id": "q1", "pass": "1", "asr_ms": "99"}])
        rows = main_rows(load_runs([p]))
        self.assertEqual([r["asr_ms"] for r in rows], [10.0])

    def test_parse_retrieval(self):
        r = parse_retrieval(RETRIEVAL_OUT)
        self.assertEqual(r["per_query"], {"q1": 0.5, "q2": 1.0})
        self.assertAlmostEqual(r["aggregate"]["MRR@10"], 0.55)

    def test_variant_table_match_rate_against_v1(self):
        p = self.d / "a.csv"
        rows = [{"query_id": "q1", "variant": "v1", "transcript": "a b"},
                {"query_id": "q2", "variant": "v1", "transcript": "c d"},
                {"query_id": "q1", "variant": "v3", "setting": "m0", "transcript": "a b"},
                {"query_id": "q2", "variant": "v3", "setting": "m0", "transcript": "c x"}]
        write_csv(p, "G", rows)
        table = {(t["variant"], t["setting"]): t for t in variant_table(main_rows(load_runs([p])))}
        self.assertEqual(table[("v1", "full")]["match_pct"], 100.0)
        self.assertEqual(table[("v3", "m0")]["match_pct"], 50.0)
        self.assertAlmostEqual(table[("v3", "m0")]["wer_vs_v1"], 0.25)


if __name__ == "__main__":
    unittest.main()
