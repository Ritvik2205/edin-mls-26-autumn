import json
import tempfile
import unittest
from pathlib import Path

from extensions.fast_asr.cachefile import append_cache, read_cache


class CacheFileTests(unittest.TestCase):
    def test_cache_roundtrip_matches_evaluate_rag_format(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "sub" / "cache.jsonl"
            append_cache(path, "q1", "Concord returned")
            append_cache(path, "q2", "café ünïcode")
            self.assertEqual(read_cache(path), {"q1": "Concord returned", "q2": "café ünïcode"})
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[1], json.dumps({"query_id": "q2", "transcript": "café ünïcode"},
                                                  ensure_ascii=False))

    def test_missing_file_is_empty(self):
        self.assertEqual(read_cache(Path("/nonexistent/cache.jsonl")), {})


if __name__ == "__main__":
    unittest.main()
