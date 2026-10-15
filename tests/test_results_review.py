"""Seed summaries use sample SD and paired differences, never pooled F1."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec=importlib.util.spec_from_file_location("review",Path(__file__).parents[1]/"scripts/review-e001-results.py")
review=importlib.util.module_from_spec(spec);spec.loader.exec_module(review)


class ResultsReviewTest(unittest.TestCase):
    def test_seed_means_sample_deviation_paired_differences_and_empty_slices(self):
        a=[.1,.2,.3,.4,.5];b=[.11,.22,.33,.44,.55]
        summary=review.moments(a)
        self.assertAlmostEqual(summary["mean"],30)
        self.assertAlmostEqual(summary["std"],np.sqrt(250))
        self.assertEqual(summary["mean"],100*np.mean(a))
        self.assertEqual(summary["std"],100*np.std(a,ddof=1))
        delta=review.paired(a,b)
        self.assertAlmostEqual(delta["mean"],3)
        self.assertAlmostEqual(delta["std"],np.sqrt(2.5))
        self.assertEqual(delta["positive_seeds"],5)
        self.assertEqual(review.fmt([None]*5),"—")
        self.assertIsNone(review.paired([None]*5,[None]*5)["positive_seeds"])
        with self.assertRaises(AssertionError):review.moments([None,.1,.1,.1,.1])
        with self.assertRaises(AssertionError):review.moments([.1]*4)


if __name__=="__main__":unittest.main()
