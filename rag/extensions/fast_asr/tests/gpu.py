"""Shared helpers for tests that need the mls-asr env and a CUDA GPU (skip otherwise)."""
import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None


def cuda_available() -> bool:
    if not HAS_TORCH:
        return False
    import torch
    return torch.cuda.is_available()


requires_gpu = unittest.skipUnless(cuda_available(), "needs torch + CUDA (run in mls-asr on a GPU node)")
