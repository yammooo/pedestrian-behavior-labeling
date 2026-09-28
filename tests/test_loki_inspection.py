import json
import math
import struct
import tempfile
import unittest
from pathlib import Path

from pedestrian_behavior.datasets import loki
from pedestrian_behavior.inspection.__main__ import select
from pedestrian_behavior.inspection.render import bev_pixel, focus_centers, footprint, read_points


class LokiInspectionTest(unittest.TestCase):
    def test_join_and_alignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            scenario = Path(temporary) / "scenario_001"
            scenario.mkdir()
            (scenario / "image_0000.png").touch()
            (scenario / "image_0002.png").touch()
            (scenario / "pc_0000.ply").touch()
            (scenario / "odom_0000.txt").touch()
            (scenario / "label2d_0000.json").write_text(json.dumps({"Pedestrian": {
                "both": {"box": {"left": 1, "top": 2, "width": 3, "height": 4}},
                "two_only": {"box": {"left": 5}},
            }}))
            (scenario / "label3d_0000.txt").write_text(
                "labels,track_id,intended_actions\n"
                "Pedestrian,both,Waiting to cross\n"
                "Pedestrian,three_only,Crossing the road\n"
                "Car,car,Stopped\n"
            )
            first, second = list(loki.frames(scenario))
            self.assertEqual([first.frame_id, second.frame_id], ["0000", "0002"])
            self.assertEqual(set(first.pedestrians), {"both", "two_only", "three_only"})
            self.assertEqual(first.pedestrians["both"].action, "Waiting to cross")
            self.assertEqual(first.pedestrians["both"].box["left"], 1)
            self.assertIsNone(first.pedestrians["two_only"].label3d)
            self.assertIsNone(first.pedestrians["three_only"].box)
            self.assertIsNone(second.paths["label2d"])
            self.assertIsNone(second.paths["label3d"])
            self.assertEqual(second.pedestrians, {})

    def test_selection_is_stable_and_paged(self):
        matches = {("scenario_001", "b"), ("scenario_000", "a"), ("scenario_002", "c")}
        all_matches = select(matches, 0, 0, 3)
        self.assertEqual(all_matches, select(set(reversed(sorted(matches))), 0, 0, 3))
        self.assertEqual(all_matches[1:], select(matches, 0, 1, 2))
        self.assertEqual(len(set(all_matches)), 3)

    def test_bev_geometry_and_missing_label_focus(self):
        with tempfile.TemporaryDirectory() as temporary:
            cloud = Path(temporary) / "pc_0000.ply"
            cloud.write_bytes(
                b"ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
                b"property float x\nproperty float y\nproperty float z\n"
                b"property float intensity\nend_header\n"
                + struct.pack("<ffffffff", 1, 2, 3, 0.5, -1, -2, -3, 0.25)
            )
            self.assertEqual(list(read_points(cloud)), [(1, 2, 3, 0.5), (-1, -2, -3, 0.25)])
            self.assertEqual(bev_pixel(1, 2, (1, 2)), (302, 302))
            self.assertEqual(bev_pixel(2, 3, (1, 2)), (317.1, 286.9))
            label = {"pos_x": "10", "pos_y": "20", "dim_x": "4", "dim_y": "2", "yaw": str(math.pi / 2)}
            for actual, expected in zip(footprint(label), [(11, 18), (9, 18), (9, 22), (11, 22)]):
                self.assertAlmostEqual(actual[0], expected[0])
                self.assertAlmostEqual(actual[1], expected[1])
            frames = [loki.Frame("s", str(i), {}, {}) for i in range(6)]
            frames[1].pedestrians["p"] = loki.Pedestrian("p", label3d={"pos_x": "0", "pos_y": "2"})
            frames[3].pedestrians["p"] = loki.Pedestrian("p", label3d={"pos_x": "4", "pos_y": "6"})
            self.assertEqual(focus_centers(frames, "p"), [(0, 2), (0, 2), (2, 4), (4, 6), (4, 6), (4, 6)])
            cloud.write_bytes(b"ply\nformat ascii 1.0\nend_header\n")
            with self.assertRaisesRegex(ValueError, "unsupported PLY"):
                list(read_points(cloud))


if __name__ == "__main__":
    unittest.main()
