import csv
import tempfile
import unittest
from pathlib import Path

from extensions.fast_asr.bench_asr import COLUMNS, run_bench
from extensions.fast_asr.cachefile import read_cache

META = {"split": "practice", "cuda": "all", "gpu": "FakeGPU", "node": "n1", "commit": "abc1234"}


class FakeTranscriber:
    def __init__(self, empty_ids=()):
        self.calls = []
        self.empty_ids = set(empty_ids)

    def after_warmup(self):
        pass

    def transcribe(self, item):
        self.calls.append(item["query_id"])
        text = "" if item["query_id"] in self.empty_ids else f"text {item['query_id']}"
        return {"duration_s": item["duration_seconds"], "real_frames": 500, "enc_frames": 3000,
                "prompt_len": 59, "n_tokens": 4, "feat_ms": 1.0, "encode_ms": 10.0, "decode_ms": 20.0,
                "peak_mem_bytes": 123, "kv_bytes": 0, "transcript": text}


def items(n):
    return [{"query_id": f"q{i}", "duration_seconds": 5.0, "audio_file": f"a{i}.wav"} for i in range(n)]


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class BenchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.csv, self.cache = d / "out.csv", d / "cache.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def run_it(self, fake, its, warmup=0, repeat_ids=()):
        return run_bench(fake, its, "v2", "full", self.csv, self.cache, warmup, list(repeat_ids), META)

    def test_writes_one_row_per_query_and_cache(self):
        n = self.run_it(FakeTranscriber(), items(3))
        self.assertEqual(n, 3)
        with open(self.csv, encoding="utf-8") as f:
            self.assertEqual(f.readline().strip().split(","), COLUMNS)
        rows = read_rows(self.csv)
        self.assertEqual([r["query_id"] for r in rows], ["q0", "q1", "q2"])
        self.assertEqual(rows[0]["asr_ms"], "31.0")
        self.assertEqual(rows[0]["ms_per_token"], "5.0")
        self.assertEqual(rows[0]["gpu"], "FakeGPU")
        self.assertEqual(len(read_cache(self.cache)), 3)

    def test_resume_skips_done(self):
        self.run_it(FakeTranscriber(), items(2))
        n = self.run_it(FakeTranscriber(), items(3))
        self.assertEqual(n, 1)
        self.assertEqual(len(read_rows(self.csv)), 3)
        self.assertEqual(len(self.cache.read_text().splitlines()), 3)

    def test_empty_transcript_row(self):
        self.run_it(FakeTranscriber(empty_ids={"q1"}), items(3))
        row = read_rows(self.csv)[1]
        self.assertEqual((row["empty"], row["transcript"]), ("1", ""))
        self.assertEqual(read_cache(self.cache)["q1"], "")

    def test_warmup_rows_not_written(self):
        fake = FakeTranscriber()
        self.run_it(fake, items(3), warmup=2)
        self.assertEqual(len(fake.calls), 5)
        self.assertEqual(len(read_rows(self.csv)), 3)

    def test_repeat_pass_rows(self):
        self.run_it(FakeTranscriber(), items(3), repeat_ids=["q0", "q2"])
        rows = read_rows(self.csv)
        self.assertEqual([(r["query_id"], r["pass"]) for r in rows],
                         [("q0", "0"), ("q1", "0"), ("q2", "0"), ("q0", "1"), ("q2", "1")])
        self.assertEqual(len(read_cache(self.cache)), 3)


if __name__ == "__main__":
    unittest.main()
