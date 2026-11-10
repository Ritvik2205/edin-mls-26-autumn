import unittest

from extensions.fast_asr.trim import trimmed_frames


class TrimmedFramesTests(unittest.TestCase):
    def test_margin_rounds_up_to_8(self):
        self.assertEqual(trimmed_frames(350, margin_s=0), 352)
        self.assertEqual(trimmed_frames(350, margin_s=1), 456)  # 350 + 100 -> 456

    def test_caps_at_total(self):
        self.assertEqual(trimmed_frames(2990, margin_s=5), 3000)

    def test_bucket_never_truncates(self):
        self.assertEqual(trimmed_frames(1040, bucket_s=10), 1040)  # 10.4 s clip
        self.assertEqual(trimmed_frames(500, bucket_s=10), 1000)

    def test_exactly_one_mode(self):
        with self.assertRaises(ValueError):
            trimmed_frames(100)
        with self.assertRaises(ValueError):
            trimmed_frames(100, margin_s=1, bucket_s=10)


if __name__ == "__main__":
    unittest.main()
