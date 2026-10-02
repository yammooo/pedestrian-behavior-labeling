import csv
import gzip
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pedestrian_behavior.datasets.road_waymo import component_rows, focus_centers, range_points, read_tracks
from pedestrian_behavior.inspection.road_waymo import aligned_frames


class RoadWaymoInspectionTest(unittest.TestCase):
    def test_native_geometry_and_pixel_motion(self):
        c = "[LiDARCalibrationComponent]."
        calibration = {c + "extrinsic.transform": np.eye(4).ravel(),
                       c + "beam_inclination.values": None,
                       c + "beam_inclination.min": 0., c + "beam_inclination.max": 0.}
        values = [2., 1., 0., -1.]
        np.testing.assert_allclose(range_points(values, [1, 1, 4], calibration), [[2, 0, 0]])
        vehicle_pose = np.eye(4)
        vehicle_pose[0, 3] = 1
        np.testing.assert_allclose(range_points(values, [1, 1, 4], calibration,
                                               [0, 0, math.pi / 2, 0, 0, 0], vehicle_pose),
                                   [[-1, 2, 0]], atol=1e-12)
        np.testing.assert_allclose(range_points(values, [1, 1, 4], calibration,
                                               [0, math.pi / 2, 0, 0, 0, 0], np.eye(4)),
                                   [[0, 0, -2]], atol=1e-12)
        calibration[c + "beam_inclination.values"] = [-math.pi / 6, math.pi / 6]
        points = range_points(values * 2, [2, 1, 4], calibration)
        np.testing.assert_allclose(points[:, 2], [1, -1])
        calibration[c + "beam_inclination.values"] = [0.]
        calibration[c + "extrinsic.transform"] = [0, -1, 0, 3, 1, 0, 0, 4, 0, 0, 1, 0, 0, 0, 0, 1]
        # Extrinsic yaw is subtracted from azimuth before transforming to vehicle coordinates.
        np.testing.assert_allclose(range_points(values, [1, 1, 4], calibration), [[5, 4, 0]], atol=1e-12)
        self.assertEqual(range_points([0., 1., 0., 0.], [1, 1, 4], calibration).shape, (0, 3))
        with self.assertRaisesRegex(ValueError, "matching vehicle"):
            range_points(values, [1, 1, 4], calibration, [0.] * 6)

    def test_mask_and_repeated_annotation(self):
        with tempfile.TemporaryDirectory() as temporary:
            index = Path(temporary)
            row = dict(road_clip_id="clip", road_tube_uid="person", merged_agent_label="Ped",
                       camera_name="1", has_3d_box="False", frame_timestamp_micros="10",
                       road_frame_1based="1", road_annotation_id="a", road_box_normalized_json="[.1,.2,.3,.4]",
                       waymo_box3d_center_size_heading_json="", action_labels_json='["Wait2X"]',
                       loc_labels_json='["Pav"]', waymo_3d_label="", semantic_disagreement="False")
            row["road_box_normalized_json"] = json.dumps([.1, .2, .3, .4])
            with gzip.open(index / "pedestrians.csv.gz", "wt", newline="") as source:
                writer = csv.DictWriter(source, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
                writer.writerow(dict(row, road_annotation_id="b"))
                writer.writerow(dict(row, frame_timestamp_micros="20", road_frame_1based="2",
                                     has_3d_box="True", waymo_box3d_center_size_heading_json="[1,2,3,4,5,6,0]",
                                     waymo_3d_label="Cyclist", semantic_disagreement="True"))
            observations = read_tracks(index, [("clip", "person")])[("clip", "person")]
            self.assertIsNone(observations[10]["box3d"])
            self.assertEqual(observations[10]["annotation_ids"], ["a", "b"])
            self.assertEqual(observations[20]["original_type"], "Cyclist")
            self.assertTrue(observations[20]["disagreement"])
            self.assertEqual(observations[20]["box3d"][3:6], [4, 5, 6])
            with gzip.open(index / "pedestrians.csv.gz", "at", newline="") as source:
                csv.DictWriter(source, fieldnames=list(row)).writerow(dict(row, action_labels_json='["Stop"]'))
            with self.assertRaisesRegex(ValueError, "conflicting repeated"):
                read_tracks(index, [("clip", "person")])

    def test_timestamp_focus_and_missing_modalities(self):
        timestamps = [0, 1, 4, 6]
        poses = {ts: np.eye(4) for ts in timestamps}
        for ts in timestamps:
            poses[ts][0, 3] = 2 * ts
        observations = {0: {"box3d": [10, 0, 0, 1, 1, 2, 0]},
                        1: {"box3d": None}, 4: {"box3d": [12, 0, 0, 1, 1, 2, 0]}}
        np.testing.assert_allclose(focus_centers(timestamps, observations, poses),
                                   [(10, 0), (10.5, 0), (12, 0), (8, 0)])
        self.assertIsNone(observations[1]["box3d"])
        self.assertEqual(focus_centers(timestamps, {}, poses), [(0, 0)] * 4)
        self.assertEqual(list(aligned_frames(iter([(0, ["a"]), (4, ["b"])]), timestamps)),
                         [["a"], [], ["b"], []])
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lidar.parquet"
            data = {"key.segment_context_name": ["scene", "scene"],
                    "key.frame_timestamp_micros": [4, 0],
                    "range.values": [[2., 3.], None]}
            pq.write_table(pa.table(data), path)
            rows = component_rows(path, "scene")
            np.testing.assert_equal(next(rows)["range.values"], [2, 3])
            with self.assertRaisesRegex(ValueError, "unordered timestamps"):
                next(rows)


if __name__ == "__main__":
    unittest.main()
