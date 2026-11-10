import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from extensions.fast_asr.audio import read_audio


def write_wav(path, rate=16000, channels=1):
    with wave.open(str(path), "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(np.array([0, 16384, -16384] * channels, dtype=np.int16).tobytes())


class AudioTests(unittest.TestCase):
    def test_pcm16_to_float(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.wav"
            write_wav(p)
            np.testing.assert_allclose(read_audio(p), [0, 0.5, -0.5])

    def test_rejects_wrong_rate_and_stereo(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.wav"
            write_wav(p, rate=8000)
            with self.assertRaises(ValueError):
                read_audio(p)
            write_wav(p, channels=2)
            with self.assertRaises(ValueError):
                read_audio(p)


if __name__ == "__main__":
    unittest.main()
