"""M = 1 linear layers: the two GEMV kernels against the tiled GEMM (GPU, student kernels).

launch_linear picks the kernel from ASR_GEMV at every launch, so one process can compare them.
The tiled reference for a single row x is row 0 of a two-row call, which always takes the
tiled kernel.
"""
import os
import unittest
from pathlib import Path

from extensions.fast_asr.tests.gpu import requires_gpu

REPO = Path(__file__).resolve().parents[4]
TEST_WAV = REPO / "hw1-asr" / "test_audio.wav"
GROUPS = os.environ.get("FAST_ASR_TEST_CUDA", "all")
# (N, K): decoder q_proj, gate_proj, down_proj, lm_head, and an odd shape for the scalar path.
SHAPES = [(2048, 2048), (6144, 2048), (2048, 6144), (59264, 2048), (37, 33)]


@requires_gpu
@unittest.skipUnless(GROUPS == "all", "compares student kernels with each other")
class GemvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from extensions.fast_asr import _paths  # noqa: F401
        from glm_asr_cuda import ops
        cls.torch, cls.ops = torch, ops
        ops.set_cuda_groups("all")
        cls.saved = os.environ.get("ASR_GEMV")

    def tearDown(self):
        if self.saved is None:
            os.environ.pop("ASR_GEMV", None)
        else:
            os.environ["ASR_GEMV"] = self.saved

    def _pair(self, mode, N, K, bias, act):
        """(tiled row, GEMV row) for one random x."""
        t = self.torch
        g = t.Generator(device="cuda").manual_seed(N * 7 + K)
        x = t.randn(1, K, device="cuda", generator=g)
        w = t.randn(N, K, device="cuda", generator=g) / K ** 0.5
        b = t.randn(N, device="cuda", generator=g) if bias else None
        os.environ["ASR_GEMV"] = "off"
        tiled = self.ops.linear(t.cat([x, x]), w, b, act)[0]
        os.environ["ASR_GEMV"] = mode
        return tiled, self.ops.linear(x, w, b, act)[0]

    def test_seq_is_bit_identical_to_tiled(self):
        for N, K in SHAPES:
            for bias, act in [(False, None), (True, None), (True, "gelu")]:
                tiled, gemv = self._pair("seq", N, K, bias, act)
                self.assertTrue(self.torch.equal(tiled, gemv), f"N={N} K={K} bias={bias} act={act}")

    def test_warp_is_close_to_tiled(self):
        worst = 0.0
        for N, K in SHAPES:
            tiled, gemv = self._pair("warp", N, K, True, None)
            worst = max(worst, float((tiled - gemv).abs().max()))
            self.assertTrue(self.torch.allclose(tiled, gemv, rtol=1e-4, atol=1e-5), f"N={N} K={K}")
        print(f"\nwarp GEMV vs tiled: max abs difference {worst:.3e}")

    def test_off_at_m1_equals_tiled_row(self):
        tiled, same = self._pair("off", 2048, 2048, True, None)
        self.assertTrue(self.torch.equal(tiled, same))

    def test_kv_decode_with_gemv(self):
        """seq: tokens and probabilities equal V1's bit for bit. warp: reported, tokens checked."""
        from extensions.fast_asr.audio import read_audio
        from extensions.fast_asr.kv_model import load_fast, prepare_inputs
        fast, processor = load_fast()
        inp = prepare_inputs(processor, read_audio(TEST_WAV))
        inp.pop("feat_ms")
        os.environ["ASR_GEMV"] = "off"
        ref = fast.generate_recompute(**inp)
        os.environ["ASR_GEMV"] = "seq"
        seq = fast.generate_kv(**inp)
        self.assertEqual(seq.tokens, ref.tokens)
        self.assertEqual(max(abs(a - b) for a, b in zip(ref.probs, seq.probs)), 0.0)
        os.environ["ASR_GEMV"] = "warp"
        warp = fast.generate_kv(**inp)
        diff = max(abs(a - b) for a, b in zip(ref.probs, warp.probs)) if warp.tokens == ref.tokens else float("nan")
        print(f"\nwarp GEMV decode: tokens equal V1: {warp.tokens == ref.tokens}, "
              f"max probability difference {diff:.3e}")
        self.assertEqual(warp.tokens, ref.tokens)


if __name__ == "__main__":
    unittest.main()
