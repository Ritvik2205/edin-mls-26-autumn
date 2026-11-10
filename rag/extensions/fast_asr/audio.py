"""16 kHz mono PCM16 WAV -> float32 in [-1, 1), the same contract as rag/src/asr_worker.read_audio."""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def read_audio(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getcomptype()) != (16000, 1, 2, "NONE"):
            raise ValueError("Expected a 16 kHz mono PCM16 WAV file")
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16).astype(np.float32)
    if not len(samples):
        raise ValueError("Audio file is empty")
    return samples / 32768.0
