"""Input-only stress selection and explicit throughput forecast arithmetic."""
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from pedestrian_behavior.experiments.e002a_throughput import cases, forecast, OneUpdate, advisory_cold
from pedestrian_behavior.data.indexed_points import DTYPES


class ThroughputTest(unittest.TestCase):
    def test_page_eviction_can_retain_verification_without_resetting_flags(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in DTYPES: (root/(name+".bin")).write_bytes(b"x")
            points = Mock(root=root, record={"scenes":{"g":{"path":"."}}})
            dataset = SimpleNamespace(supply=SimpleNamespace(points=points), entries=[{"group":"g"}])
            with patch("os.posix_fadvise", create=True) as advise:
                advisory_cold(dataset, [0], reset_verification=False)
                points.close.assert_called_once()
                points.reset_verification.assert_not_called()
                self.assertEqual(advise.call_count, len(DTYPES))
                advisory_cold(dataset, [0])
                points.reset_verification.assert_called_once()

    def test_cases_preserve_unique_complete_tracks_and_use_input_costs_only(self):
        data = SimpleNamespace(entries=[{"slots": i+1} for i in range(70)], cost=lambda i: 70-i)
        class Dataset:
            entries = data.entries
            def __len__(self): return 70
            def cost(self,i): return data.cost(i)
        selected = cases(Dataset())
        self.assertEqual(selected["dense"], list(range(64)))
        self.assertEqual(selected["long"], list(range(69,5,-1)))
        self.assertEqual(cases(Dataset()),selected)
        self.assertTrue(all(len(set(v))==64 for v in selected.values()))

    def test_forecast_counts_train_validation_once_and_keeps_overhead_explicit(self):
        rows=[{"operation":"training","tracks":64,"seconds":32},
              {"operation":"training","tracks":64,"seconds":16},
              {"operation":"validation","tracks":64,"seconds":8}]
        r=forecast(rows,1000,200)
        self.assertEqual(r["stress_envelope_seconds"],585)
        self.assertTrue(r["under_30_minutes"])
        self.assertEqual(list(OneUpdate(None,[1,4])),[[1,4]])
        logical = SimpleNamespace(microbatches=lambda indices: iter([indices[:1], indices[1:]]))
        timed = OneUpdate(logical, [1,4])
        self.assertEqual(list(timed.microbatches([1,4])), [[1],[4]])
        self.assertGreaterEqual(timed.loading_seconds, 0.)
