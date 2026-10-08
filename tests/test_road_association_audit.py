"""Native-ID audit preserves extensions/missing pairs and rejects source changes."""

import csv
import gzip
import importlib.util
from pathlib import Path
import tempfile
import unittest

from pedestrian_behavior.data.preparation import prepare_collection
from pedestrian_behavior.datasets.road_waymo import TrackReader
from test_track_preparation import native_readers, pose

spec = importlib.util.spec_from_file_location("association_audit", Path(__file__).resolve().parents[1]/"scripts/audit-road-associations.py")
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


class RoadAssociationAuditTest(unittest.TestCase):
    def test_native_extensions_missing_pairs_and_source_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            _, reader=native_readers(root,[0,200000,400000],[pose()]*3,
                {"person":{0:[10,5,1],2:[11,5,1]}},camera={"person":{0,1,2}},road_times={"person":{1}})
            path=reader.index/"pedestrians.csv.gz"
            with gzip.open(path,"rt",newline="") as source:
                rows=list(csv.DictReader(source))
            for row in rows:
                row["road_box_normalized_json"]="[0.1,0.2,0.3,0.4]"
            with gzip.open(path,"wt",newline="") as destination:
                writer=csv.DictWriter(destination,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            reader=TrackReader(reader.index)
            collection=root/"collection"
            manifest,_=prepare_collection(reader,collection)
            report=audit_module.audit(reader,collection,[])
            counts=report["counts"]
            self.assertEqual([counts[k] for k in ("native_candidates","with_unique_lidar_id","native_union_observations",
                "native_3d_observations","export_pairs","additional_3d_timestamps","gain_3d_without_export_pair",
                "3d_before_road","3d_after_road","additional_front_timestamps")],[1,1,3,2,0,2,1,1,1,2])
            self.assertEqual(report["cases"][0]["snapshot_times"],[0,200000,400000])
            self.assertEqual(report["cases"][0]["native_road_times"],[200000])
            self.assertEqual(report["status"],"structural-pass")
            rows[0]["action_labels_json"]='["Stop"]'
            with gzip.open(path,"wt",newline="") as destination:
                writer=csv.DictWriter(destination,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            with self.assertRaisesRegex(ValueError,"source mismatch"):
                audit_module.audit(TrackReader(reader.index),collection,[])


if __name__=="__main__":
    unittest.main()
