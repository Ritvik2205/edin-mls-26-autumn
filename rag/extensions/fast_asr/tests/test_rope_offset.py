import unittest

from extensions.fast_asr.tests.gpu import requires_gpu


@requires_gpu
class RopeOffsetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from extensions.fast_asr import _paths  # noqa: F401  (puts hw1-asr on sys.path)
        from glm_asr_cuda import ops
        from glm_asr_cuda.model import inv_freq
        from extensions.fast_asr.rope_offset import rope_at_
        cls.torch, cls.ops, cls.rope_at_ = torch, ops, rope_at_
        cls.freq, cls.rdim = inv_freq(128, 1.0, 10000.0, "cuda")
        cls.afreq, cls.ardim = inv_freq(64, 0.5, 10000.0, "cuda")

    def rand(self, s, h, d=128):
        g = self.torch.Generator(device="cuda").manual_seed(s * 31 + h)
        return self.torch.randn(1, s, h, d, device="cuda", generator=g)

    def test_offset_zero_bitwise_equals_launch_rope(self):
        q, k = self.rand(59, 16), self.rand(59, 4)
        q2, k2 = q.clone(), k.clone()
        self.ops.rope_(q, k, self.freq, self.rdim)
        self.rope_at_(q2, k2, self.freq, self.rdim, 0)
        self.assertTrue(self.torch.equal(q, q2) and self.torch.equal(k, k2))

    def test_offset_matches_full_sequence_row(self):
        for P in (58, 144):
            qf, kf = self.rand(P + 1, 16), self.rand(P + 1, 4)
            q1, k1 = qf[:, P:P + 1].clone(), kf[:, P:P + 1].clone()
            self.ops.rope_(qf, kf, self.freq, self.rdim)
            self.rope_at_(q1, k1, self.freq, self.rdim, P)
            self.assertTrue(self.torch.equal(qf[:, P:P + 1], q1), P)
            self.assertTrue(self.torch.equal(kf[:, P:P + 1], k1), P)

    def test_audio_partial_rotary(self):
        q, k = self.rand(5, 20, 64), self.rand(5, 20, 64)
        q0, k0 = q.clone(), k.clone()
        self.rope_at_(q, k, self.afreq, self.ardim, 7)
        self.assertTrue(self.torch.equal(q[..., 32:], q0[..., 32:]))
        self.assertTrue(self.torch.equal(k[..., 32:], k0[..., 32:]))
        self.assertFalse(self.torch.equal(q[..., :32], q0[..., :32]))

    def test_kernel_and_scratch_paths_bitwise_equal(self):
        if "rope" not in self.ops.USE_CUDA:
            self.skipTest("launch_rope_at is a CUDA path; rope group is on PyTorch")
        for P in (1, 58, 144):
            q, k = self.rand(1, 16), self.rand(1, 4)
            q2, k2 = q.clone(), k.clone()
            self.rope_at_(q, k, self.freq, self.rdim, P, impl="kernel")
            self.rope_at_(q2, k2, self.freq, self.rdim, P, impl="scratch")
            self.assertTrue(self.torch.equal(q, q2) and self.torch.equal(k, k2), P)

    def test_unknown_impl_rejected(self):
        q, k = self.rand(1, 16), self.rand(1, 4)
        with self.assertRaises(ValueError):
            self.rope_at_(q, k, self.freq, self.rdim, 3, impl="fast")


class RopeImplChoiceTests(unittest.TestCase):
    """No GPU needed: which RoPE path is chosen."""

    def test_choice(self):
        from extensions.fast_asr.rope_offset import choose_impl
        self.assertEqual(choose_impl(None, rope_on_cuda=True, has_symbol=True, env=None), "kernel")
        self.assertEqual(choose_impl(None, rope_on_cuda=True, has_symbol=False, env=None), "scratch")
        self.assertEqual(choose_impl(None, rope_on_cuda=False, has_symbol=True, env=None), "scratch")
        self.assertEqual(choose_impl(None, rope_on_cuda=True, has_symbol=True, env="scratch"), "scratch")
        self.assertEqual(choose_impl("scratch", rope_on_cuda=True, has_symbol=True, env="kernel"), "scratch")
        with self.assertRaises(ValueError):
            choose_impl("kernel", rope_on_cuda=False, has_symbol=True, env=None)


if __name__ == "__main__":
    unittest.main()
