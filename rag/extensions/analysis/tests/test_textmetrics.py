import unittest

from extensions.analysis.textmetrics import wer, word_errors


class TextMetricsTests(unittest.TestCase):
    def test_wer(self):
        self.assertEqual(wer("Concord returned to its place", "concord returned to its place."), 0.0)
        self.assertAlmostEqual(wer("a b c d", "a x c"), 0.5)  # 1 sub + 1 del
        self.assertEqual(wer("", ""), 0.0)

    def test_word_errors_counts_insertions(self):
        self.assertEqual(word_errors("a b", "a b c"), 1)

    def test_empty_ref_nonempty_hyp(self):
        self.assertEqual(wer("", "extra words"), 2.0)  # denominator max(1, len(ref))


if __name__ == "__main__":
    unittest.main()
