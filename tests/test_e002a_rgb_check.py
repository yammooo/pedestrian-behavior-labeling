"""Visual-check regression: match extraction batches and preserve prior evidence."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from pedestrian_behavior.experiments.e002a_rgb_check import refresh_probe_gallery, run
from pedestrian_behavior.inspection.feature_probe import square_crop


class DinoVisualCheckTest(unittest.TestCase):
    def test_original_batch_composition_counts_and_missing_examples(self):
        self.exercise(compare=False)

    def test_multiscale_keeps_baseline_and_encodes_two_larger_crops(self):
        self.exercise(compare=True)

    def exercise(self, compare):
        with tempfile.TemporaryDirectory() as temporary:
            threads = torch.get_num_threads()
            self.addCleanup(torch.set_num_threads, threads)
            root = Path(temporary)/"outputs"
            probe = root/"experiments/E002a/native-probe"
            (probe/"fixtures").mkdir(parents=True)
            image = Image.new("RGB", (100, 100), "green")
            box = [.3, .2, .6, .8]
            tracks, boxes = [], {}
            for i in range(20):
                identity = {"scenario": "group", "track_id": str(i)}
                locator = json.dumps(identity)
                valid = i >= 3
                tracks.append({"locator": locator, "slot": 0, "crop": square_crop(image, box if valid else None)[1]})
                if valid:
                    boxes[str(i)] = box
            label = "loki-group-0000"
            # Deliberately batch-sensitive features catch accidental single-crop inference.
            original = np.concatenate([np.full((16, 768), 16), np.full((1, 768), 1)]).astype(np.float32)
            np.savez(probe/"fixtures"/(label+"-rgb.npz"), cls=original,
                     locators=np.array([t["locator"] for t in tracks[3:]]))
            weights = probe/"weights.pth"
            weights.write_bytes(b"fixture")
            source = {"status": "complete", "protected": {}, "config": {
                "dino_code": "fixture", "dino_weights": str(weights), "loki_root": None,
                "road_index": None, "waymo_root": "/fixture/waymo"}, "frames": [{"dataset": "loki", "group": "group",
                    "key": "0000", "split": "training", "rendered": True, "tracks": tracks}]}
            source_path = probe/"probe.json"
            source_path.write_text(json.dumps(source))
            before = source_path.read_bytes()
            for dataset in ("loki", "road-waymo"):
                collection = root/"experiments/E001/reader-preparation"/dataset
                collection.mkdir(parents=True)
                (collection/"manifest.json").write_text(json.dumps({"sources": {
                    "root": "fixture", "index": "fixture", "sha256": "fixture"}, "scenes": {"group": {}}}))
            batches = []
            class Encoder:
                def __init__(self, *args):
                    pass
                def descriptors(self, images, *, patches):
                    n = len(images)
                    batches.append(n)
                    return np.full((n, 768), n, np.float32), np.ones((n, 16, 16, 768), np.float32)
            module = "pedestrian_behavior.experiments.e002a_rgb_check."
            reader = SimpleNamespace(sources={"sha256": "fixture"})
            output = probe.parent/"visual-check"
            with patch(module+"ImageEncoder", Encoder), patch(module+"loki.TrackReader", return_value=reader), \
                 patch(module+"road_waymo.TrackReader", return_value=reader) as road_reader, \
                 patch(module+"native_camera", return_value=iter([("0000", image, boxes)])), \
                 patch(module+"crop_sheet"), patch(module+"dino_sheet"), patch(module+"crop_scale_sheet"), \
                 patch(module+"patch_maps", return_value=(np.zeros((16, 16, 3)), np.ones((16, 16)))):
                run(SimpleNamespace(probe=probe, output=output, device="cpu", compare_crop_scales=compare))
            self.assertIsInstance(road_reader.call_args.args[1], Path)
            result = json.loads((output/"visual-check.json").read_text())
            self.assertEqual(result["status"], "complete")
            self.assertEqual(batches, [16, 16, 1, 2, 2] if compare else [16, 16, 1])
            self.assertEqual(source_path.read_bytes(), before)
            self.assertEqual(result["changed_files"], [])
            frame = result["frames"][0]
            self.assertEqual((frame["valid_crops"], frame["tracks"]), (17, 20))
            self.assertEqual([e["crop"]["valid"] for e in frame["examples"]], [True, True, False])
            with np.load(output/"fixtures"/(frame["examples"][0]["name"]+".npz"), allow_pickle=False) as saved:
                self.assertEqual(saved["patches"].shape, (16, 16, 768))
                np.testing.assert_array_equal(saved["cls"], original[0])
            if compare:
                scales = frame["examples"][0]["crop_scales"]
                self.assertEqual([s["expansion"] for s in scales], [1.5, 2., 3.])
                self.assertEqual(result["crop_scales"], [1.5, 2., 3.])
                for scale in scales[1:]:
                    with np.load(output/"fixtures"/scale["fixture"], allow_pickle=False) as saved:
                        self.assertEqual(saved["crop"].shape, (256, 256, 3))
                        np.testing.assert_array_equal(saved["cls"], np.full(768, 2))
                self.assertIn("1.5× / 2× / 3×", (output/"index.html").read_text())
            self.assertIn("17/20 valid crops", (output/"index.html").read_text())
            # The original 3D gallery must show the BEV and corrected crops,
            # while preserving numerical evidence and the old presentation.
            (probe/"index.html").write_text("original gallery")
            (probe/"panels").mkdir()
            image.save(probe/"panels"/(label+"-bev.png"))
            for example in frame["examples"]:
                image.save(output/"panels"/(example["name"]+"-crop.png"))
            refresh_probe_gallery(probe, output)
            page = (probe/"index.html").read_text()
            self.assertIn("Native RGB + LiDAR checks", page)
            self.assertIn("17/20 valid crops", page)
            self.assertIn("Missing target crop — scene context only", page)
            self.assertNotIn("Existing LiDAR BEV</summary>", page)
            self.assertEqual((probe/"index-before-crop-fix.html").read_text(), "original gallery")
            self.assertEqual(source_path.read_bytes(), before)
            source["status"] = "partial-rgb-weights-unavailable"
            source_path.write_text(json.dumps(source))
            refresh_probe_gallery(probe, output)
            self.assertEqual((probe/"index-before-crop-fix.html").read_text(), "original gallery")
            result["frames"][0]["valid_crops"] = 0
            (output/"visual-check.json").write_text(json.dumps(result))
            with self.assertRaises(ValueError):
                refresh_probe_gallery(probe, output)
            self.assertEqual((probe/"index.html").read_text(), page)



if __name__ == "__main__":
    unittest.main()
