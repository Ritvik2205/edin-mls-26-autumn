"""V1 recompute vs supplied generate, V2 KV cache vs V1, V3 trimming invariants (GPU)."""
import json
import os
import unittest
from pathlib import Path

from extensions.fast_asr.tests.gpu import requires_gpu

REPO = Path(__file__).resolve().parents[4]
TEST_WAV = REPO / "hw1-asr" / "test_audio.wav"
CORPUS = REPO / "rag" / "data" / "rag_corpus"
# Kernel groups the tests run on. "all" = the student kernels (default). Set
# FAST_ASR_TEST_CUDA=none to debug the V2/V3 logic on the PyTorch path before the kernels
# are finished, or e.g. "task1,task2" while only some tasks pass.
GROUPS = os.environ.get("FAST_ASR_TEST_CUDA", "all")


def longest_practice_clip() -> Path:
    rows = [json.loads(l) for l in (CORPUS / "student_audio_manifest.jsonl").read_text().splitlines() if l.strip()]
    best = max((r for r in rows if r["split"] == "practice"), key=lambda r: r["duration_seconds"])
    return CORPUS / best["audio_file"]


@requires_gpu
class EquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from extensions.fast_asr.kv_model import load_fast, prepare_inputs, real_frames
        from extensions.fast_asr.audio import read_audio
        from glm_asr_cuda import ops
        cls.torch, cls.ops, cls.real_frames = torch, ops, staticmethod(real_frames)
        ops.set_cuda_groups(GROUPS)
        cls.fast, cls.processor = load_fast()
        cls.base = cls.fast.base
        cls.inp = prepare_inputs(cls.processor, read_audio(TEST_WAV))
        cls.inp.pop("feat_ms")

    def tearDown(self):
        self.ops.set_cuda_groups(GROUPS)

    # ---- Task 8: V1 instrumented recompute ----
    def test_recompute_equals_supplied_generate(self):
        tok_ref, p_ref = self.base.generate(**self.inp)
        r = self.fast.generate_recompute(**self.inp)
        self.assertEqual(r.tokens, tok_ref)
        self.assertEqual(r.probs, p_ref)  # exact floats
        self.assertEqual(r.prompt_len, 59)
        self.assertGreater(r.encode_ms, 0)
        self.assertGreater(r.decode_ms, 0)

    # ---- Task 9: V2 KV cache ----
    def test_kv_bit_identical_on_student_kernels(self):
        if GROUPS != "all":
            self.skipTest("bit-identity is a claim about the student kernels (GROUPS=all)")
        a = self.fast.generate_recompute(**self.inp)
        b = self.fast.generate_kv(**self.inp)
        self.assertEqual(b.tokens, a.tokens)
        self.assertEqual(max(abs(x - y) for x, y in zip(a.probs, b.probs)), 0.0)  # spec §5 hypothesis

    def test_kv_torch_path_close(self):
        self.ops.set_cuda_groups("none")
        a = self.fast.generate_recompute(**self.inp)
        b = self.fast.generate_kv(**self.inp)
        self.assertEqual(b.tokens, a.tokens)
        self.assertLess(max(abs(x - y) for x, y in zip(a.probs, b.probs)), 1e-4)

    def test_prefill_inside_decode(self):
        for r in (self.fast.generate_recompute(**self.inp), self.fast.generate_kv(**self.inp)):
            self.assertGreater(r.prefill_ms, 0)
            self.assertLess(r.prefill_ms, r.decode_ms)

    def test_kv_max_new_tokens_boundary(self):
        a = self.fast.generate_recompute(**self.inp, max_new_tokens=3)
        b = self.fast.generate_kv(**self.inp, max_new_tokens=3)
        self.assertEqual(b.tokens, a.tokens)
        self.assertEqual(len(b.tokens), 3)

    def test_kv_first_token_eos(self):
        first = self.fast.generate_kv(**self.inp, max_new_tokens=1).tokens[0]
        saved = self.base.c.eos_token_ids
        self.base.c.eos_token_ids = (first,)
        try:
            b = self.fast.generate_kv(**self.inp)
            self.assertEqual(b.tokens, [first])
        finally:
            self.base.c.eos_token_ids = saved

    def test_kv_bytes(self):
        b = self.fast.generate_kv(**self.inp)
        self.assertEqual(b.kv_bytes, 28 * 2 * (59 + 100) * 4 * 128 * 4)

    # ---- Task 10: V3 encoder trimming ----
    def test_full_length_equals_v2(self):
        self.assertEqual(self.fast.generate_kv(**self.inp, encoder_frames=3000).probs,
                         self.fast.generate_kv(**self.inp).probs)

    def test_prompt_len_unchanged_longest_clip(self):
        from extensions.fast_asr.audio import read_audio
        from extensions.fast_asr.kv_model import prepare_inputs
        from extensions.fast_asr.trim import trimmed_frames
        inp = prepare_inputs(self.processor, read_audio(longest_practice_clip()))
        inp.pop("feat_ms")
        v2 = self.fast.generate_kv(**inp, max_new_tokens=1)
        for kw in ({"margin_s": 0}, {"bucket_s": 10}):
            T = trimmed_frames(self.real_frames(inp["features_mask"]), **kw)
            r = self.fast.generate_kv(**inp, encoder_frames=T, max_new_tokens=1)
            self.assertEqual(r.prompt_len, v2.prompt_len, kw)
            self.assertEqual(r.enc_frames, T)

    def test_rejects_cutting_real_audio(self):
        real = self.real_frames(self.inp["features_mask"])
        with self.assertRaises(ValueError):
            self.fast.generate_kv(**self.inp, encoder_frames=(real // 8) * 8 - 8)
        with self.assertRaises(ValueError):
            self.fast.generate_kv(**self.inp, encoder_frames=(real // 8) * 8 + 9)  # not a multiple of 8


if __name__ == "__main__":
    unittest.main()
