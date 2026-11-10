import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from extensions.fast_asr.adapter import FastASRAdapter

FAKE_WORKER = """
import json, sys
print(json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
    json.loads(line)
    print(json.dumps({"transcript": " ".join(sys.argv[1:])}), flush=True)
"""


class FastAdapterTests(unittest.TestCase):
    def test_fast_adapter_spawns_variant_worker(self):
        with tempfile.TemporaryDirectory() as d:
            worker = Path(d) / "fake_worker.py"
            worker.write_text(FAKE_WORKER)
            audio = Path(d) / "q.wav"
            with wave.open(str(audio), "wb") as f:
                f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000)
                f.writeframes(np.zeros(10, dtype=np.int16).tobytes())
            with FastASRAdapter(variant="v3", margin_s=1, python=sys.executable, worker_path=worker) as asr:
                argv = asr.transcribe(audio)
        self.assertIn("--variant v3 --margin 1", argv)
        self.assertIn("--cuda all", argv)

    def test_rejects_bad_variant_flags(self):
        with self.assertRaises(ValueError):
            FastASRAdapter(variant="v3")
        with self.assertRaises(ValueError):
            FastASRAdapter(variant="v2", bucket_s=10)


if __name__ == "__main__":
    unittest.main()
