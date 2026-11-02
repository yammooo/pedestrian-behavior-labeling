import unittest

from pedestrian_behavior.experiments.e002_transfer import _source_normalization


class TransferNormalizationTests(unittest.TestCase):
    def test_requires_checkpoint_training_source_statistics(self):
        source = {"baseline": {"mean": [1.0]}, "bbox": {"mean": [2.0]}}
        checkpoint = {"normalization": source}
        self.assertIs(_source_normalization(checkpoint, source["baseline"], source["bbox"]), source)
        with self.assertRaisesRegex(ValueError, "training-source"):
            _source_normalization(checkpoint, {"mean": [9.0]}, source["bbox"])


if __name__ == "__main__":
    unittest.main()
