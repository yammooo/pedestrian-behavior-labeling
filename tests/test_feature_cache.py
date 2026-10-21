"""Production storage acceptance: immutable identities, bytes, persistence and failures."""

import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pedestrian_behavior.data.feature_cache import FeatureCache, gb_bytes, identity_digest, isolated_extraction
from pedestrian_behavior.data.frozen_features import point_crop, point_union, validate_points, validate_rgb
from pedestrian_behavior.experiments.e002a_features import FeatureCatalog, FeatureSupply, FrozenEncoders, selected_tracks, validate_roots, run


def payload(value=1):
    return {"values":np.full((4,64),value,np.float32)}


class StorageTest(unittest.TestCase):
    def test_byte_lru_shared_namespaces_persistence_and_lower_capacity(self):
        with tempfile.TemporaryDirectory() as root:
            with FeatureCache(root,min_free_bytes=0) as cache:
                a,b,c = ({"dataset":x} for x in ("loki","road","test"))
                cache.get(a,payload); size = cache.snapshot()["resident_bytes"]
                cache.trim(size*2)
                cache.get(b,payload)
                cache.get(a,lambda:self.fail("hit recomputed"))
                cache.get(c,payload)
                self.assertIsNone(cache.db.execute("SELECT key FROM entries WHERE key=?",(identity_digest(b),)).fetchone())
                self.assertLessEqual(cache.snapshot()["resident_bytes"],size*2)
                self.assertEqual(cache.snapshot()["evictions"],1)
            with FeatureCache(root,capacity_bytes=size,min_free_bytes=0) as cache:
                self.assertEqual(cache.snapshot()["entries"],1)
                np.testing.assert_array_equal(cache.get(c,lambda:self.fail("lost persistent MRU"))["values"],payload()["values"])
                self.assertEqual(cache.snapshot()["capacity_bytes"],size)

    def test_capacity_is_not_identity_oversize_bypasses_and_zero_cache(self):
        with tempfile.TemporaryDirectory() as root:
            with FeatureCache(root,capacity_bytes=1,min_free_bytes=0) as cache:
                expected=cache.get({"frame":1},payload)
                self.assertEqual(cache.snapshot()["bypasses"],1)
                self.assertEqual(cache.snapshot()["entries"],0)
                cache.trim(100_000)
                cache.get({"frame":1},payload)
                cache.trim(None)
                np.testing.assert_array_equal(cache.get({"frame":1},lambda:self.fail("capacity expired key"))["values"],expected["values"])
                cache.trim(0)
                self.assertEqual(cache.snapshot()["entries"],0)
        self.assertEqual(gb_bytes(450),450_000_000_000)
        for bad in (-1,float("nan"),float("inf")):
            with self.assertRaises(ValueError):gb_bytes(bad)

    def test_identity_changes_do_not_reuse_old_features(self):
        with tempfile.TemporaryDirectory() as root, FeatureCache(root,min_free_bytes=0) as cache:
            base={"checkpoint":"one","source":"one","preprocessing":"one","population":"one"}
            cache.get(base,payload)
            for field in base:
                changed=dict(base);changed[field]="two"
                np.testing.assert_array_equal(cache.get(changed,lambda:payload(2))["values"],payload(2)["values"])
            self.assertEqual(cache.snapshot()["misses"],5)

    def test_low_space_atomic_write_failure_and_recovery_preserve_entries(self):
        with tempfile.TemporaryDirectory() as root:
            with FeatureCache(root,min_free_bytes=100) as cache:
                cache.get({"frame":1},payload)
                with patch("pedestrian_behavior.data.feature_cache.shutil.disk_usage",return_value=SimpleNamespace(free=100)):
                    with self.assertRaises(OSError):cache.get({"frame":2},payload)
                with patch("pedestrian_behavior.data.feature_cache.os.replace",side_effect=OSError("write failed")):
                    with self.assertRaises(OSError):cache.get({"frame":2},payload)
                self.assertEqual(cache.snapshot()["entries"],1)
                self.assertEqual(list(cache.payload.glob("*.tmp")),[])
            # Simulate a killed writer between payload replacement and index commit.
            entries=Path(root)/"entries"
            (entries/("f"*64+".npz")).write_bytes(b"unindexed disposable write")
            (entries/"interrupted.tmp").write_bytes(b"partial")
            with FeatureCache(root,min_free_bytes=0) as cache:
                self.assertEqual(len(list(entries.iterdir())),1)
                cache.get({"frame":1},lambda:self.fail("existing entry changed"))

    def test_corruption_missing_indexed_payload_and_nonfinite_fail_without_recompute(self):
        for kind in ("bytes","missing","nan"):
            with tempfile.TemporaryDirectory() as root:
                with FeatureCache(root,min_free_bytes=0) as cache:
                    key=identity_digest({"x":1})
                    cache.get({"x":1},payload)
                    path=cache.path(key)
                    if kind=="missing":path.unlink()
                    elif kind=="bytes":path.write_bytes(b"broken")
                    else:
                        with self.assertRaises(ValueError):cache.get({"x":2},lambda:payload(np.nan))
                        self.assertEqual(cache.snapshot()["entries"],1)
                        continue
                    with self.assertRaises(ValueError):cache.get({"x":1},lambda:self.fail("corruption silently recomputed"))
                if kind=="missing":
                    with self.assertRaises(ValueError):FeatureCache(root,min_free_bytes=0)

    def test_single_owner_and_pickle_free_payload(self):
        with tempfile.TemporaryDirectory() as root:
            with FeatureCache(root,min_free_bytes=0) as cache:
                with self.assertRaises(RuntimeError):FeatureCache(root,min_free_bytes=0)
                cache.get({"unicode":"é"},payload)
                for path in cache.payload.glob("*.npz"):
                    with np.load(path,allow_pickle=False) as a:
                        self.assertEqual(a["values"].dtype,np.float32)
                        self.assertEqual(json.loads(str(a["__identity__"])),{"unicode":"é"})
                with self.assertRaises(ValueError):cache.get({"bad":1},lambda:{"x":np.array([object()])})
            with FeatureCache(root,min_free_bytes=0):pass

    def test_measured_cost_metadata_survives_payload_eviction_and_reopen(self):
        with tempfile.TemporaryDirectory() as root:
            with FeatureCache(root,min_free_bytes=0) as cache:
                identity={"kind":"cost","source":"loki","frame":1}
                self.assertIsNone(cache.metadata(identity))
                cache.record_metadata(identity,{"tokens":123})
                cache.get({"frame":1},payload)
                cache.trim(0)
                self.assertEqual(cache.metadata(identity),{"tokens":123})
                with self.assertRaises(ValueError):cache.record_metadata(identity,{"tokens":124})
            with FeatureCache(root,min_free_bytes=0) as cache:
                self.assertEqual(cache.metadata(identity),{"tokens":123})
                self.assertEqual(cache.snapshot()["resident_bytes"],0)
                with cache.db:
                    cache.db.execute('UPDATE metadata SET value=\'{"tokens":124}\'')
                with self.assertRaises(ValueError):cache.metadata(identity)

    def test_hit_payload_validator_runs_even_with_valid_file_checksum(self):
        with tempfile.TemporaryDirectory() as root, FeatureCache(root,min_free_bytes=0) as cache:
            cache.get({"x":1},payload)
            with self.assertRaises(ValueError):
                cache.get({"x":1},lambda:self.fail("recomputed"),lambda _:(_ for _ in ()).throw(ValueError("bad shape")))


class FeatureGeometryTest(unittest.TestCase):
    def fixture(self):
        points=np.array([[0,0,100,0],[20,0,0,0],[20.01,0,0,0],[40,0,-100,0],[100,0,0,0]],np.float32)
        features=np.arange(5*64,dtype=np.float32).reshape(5,64)
        boxes={"a":(np.array([0.,0,0]),np.array([2.,2,2]),0.),
               "b":(np.array([40.,0,0]),np.array([2.,2,2]),0.)}
        anchors={k:np.eye(4) for k in boxes}
        return points,features,boxes,anchors

    def test_union_keeps_all_representatives_once_and_reconstructs_each_crop(self):
        points,features,boxes,anchors=self.fixture()
        arrays=point_union(points,features,np.arange(5),boxes,np.eye(4),anchors)
        np.testing.assert_array_equal(arrays["native_indices"],[0,1,2,3])
        a=point_crop(arrays,"a");b=point_crop(arrays,"b")
        np.testing.assert_array_equal(a["native_indices"],[0,1])
        np.testing.assert_array_equal(b["native_indices"],[1,2,3])
        np.testing.assert_array_equal(a["features"],features[:2])
        self.assertEqual(a["relative_xyz"][0,2],100)  # No height cutoff.
        self.assertFalse(a["inside_box"].any())  # Context-only is still valid.
        self.assertEqual(len(arrays["features"]),4)  # Shared boundary voxel not duplicated.
        with self.assertRaises(ValueError):point_crop(arrays,"absent")
        arrays["features"][0,0]=np.inf
        with self.assertRaises(ValueError):validate_points(arrays)

    def test_rotated_anchor_and_missing_scan(self):
        points,features,boxes,anchors=self.fixture()
        anchors["a"][:2,:2]=[[0,-1],[1,0]]
        arrays=point_union(points,features,np.arange(5),boxes,np.eye(4),anchors)
        np.testing.assert_allclose(point_crop(arrays,"a")["relative_xyz"][1],[0,-20,0],atol=1e-6)
        empty=point_union(points[:0],features[:0],np.zeros(0,np.int64),boxes,np.eye(4),anchors)
        self.assertEqual(point_crop(empty,"a")["features"].shape,(0,64))
        with self.assertRaises(ValueError):point_union(points,features,np.arange(5),boxes,None,anchors)

    def test_rgb_missing_numeric_zero_and_shapes(self):
        arrays={"cls":np.zeros((2,768),np.float32),"valid":np.array([1,0],bool),
                "source_keys":np.array(["a",""]),"times_us":np.array([0,-1]),"crop_info":np.array(["{}","{}"])}
        validate_rgb(arrays)
        arrays["cls"][1,0]=1
        with self.assertRaises(ValueError):validate_rgb(arrays)


class SupplyTest(unittest.TestCase):
    def catalog(self):
        from collections import defaultdict
        tracks = {k:{"locator":k,"track_id":k,"group":"scene","slots":3,"source_keys":[1,2,None],
                     "times_us":[100,200,None],"point_usable":np.array([1,1,0],bool),"sha256":k}
                  for k in ("a","b")}
        reader = SimpleNamespace(read_frames=lambda *_:{1:{"ego_pose":np.eye(4)}})
        return SimpleNamespace(tracks=tracks,identity={"dataset":"test","source":"frozen"},anchors={},
            frames=defaultdict(list,{("scene","1"):["a","b"],("scene","2"):["a","b"]}),
            manifest={"scenes":{"scene":{}}},source_identity=lambda _:"native-stats",
            native_reader=lambda _:reader)

    def test_rgb_3x_native_only_independent_of_3d_and_no_encoder_on_hit(self):
        from PIL import Image
        catalog = self.catalog()
        catalog.tracks["a"]["point_usable"][:] = False
        native = [(1,Image.new("RGB",(100,100),"red"),{"a":[0,0,.2,.4]}),(2,None,{})]
        calls=[]
        def encode(kind,value):
            calls.append(kind)
            self.assertEqual(tuple(value.shape),(1,3,256,256))
            return np.ones((1,768),np.float32)
        with tempfile.TemporaryDirectory() as root, \
             FeatureCache(Path(root)/"rgb",min_free_bytes=0) as rgb, \
             FeatureCache(Path(root)/"points",min_free_bytes=0) as points, \
             patch("pedestrian_behavior.experiments.e002a_features.native_camera",return_value=iter(native)):
            supply=FeatureSupply({"test":catalog},rgb,points,encode,{"version":1})
            first=supply.rgb("test","a")
            np.testing.assert_array_equal(first["valid"],[1,0,0])
            self.assertEqual(json.loads(first["crop_info"][0])["bounds_pixels"],[-50.,-40.,70.,80.])
            self.assertEqual(first["times_us"].tolist(),[100,200,-1])
            supply.encoders=lambda *_:self.fail("hit loaded encoder")
            with patch.object(catalog,"native_reader",side_effect=AssertionError("native read on hit")):
                second=supply.rgb("test","a")
            for k in first:np.testing.assert_array_equal(first[k],second[k])
            self.assertEqual(calls,["rgb"])

    def test_shared_union_streams_misses_once_and_never_reads_native_on_hit(self):
        catalog=self.catalog()
        calls=[];reads=[]
        def native(reader,group,keys,tracks,frozen,**kwargs):
            reads.append(list(keys))
            self.assertFalse(kwargs["include_rgb"])
            self.assertEqual(set(tracks),{"a","b"})
            for key in keys:
                yield {"key":key,"points":np.array([[0,0,0,0],[10,0,0,0],[30,0,0,0]],np.float32),
                       "world_from_native":np.eye(4),"boxes":{k:(np.array([x,0.,0]),np.ones(3),0.) for k,x in (("a",0.),("b",30.))}}
        def encode(kind,value):
            calls.append(kind)
            return np.ones((3,64),np.float32),None,np.arange(3),np.arange(3)
        with tempfile.TemporaryDirectory() as root, \
             FeatureCache(Path(root)/"rgb",min_free_bytes=0) as rgb, \
             FeatureCache(Path(root)/"points",min_free_bytes=0) as points, \
             patch("pedestrian_behavior.experiments.e002a_features.native_scene",side_effect=native):
            supply=FeatureSupply({"test":catalog},rgb,points,encode,{"version":1})
            self.assertIsNone(supply.point_cost("test","a",0))
            a=list(supply.iter_points("test","a"))
            self.assertEqual(supply.point_cost("test","a",0)["tokens"],2)
            self.assertEqual(supply.point_cost("test","a",2)["tokens"],0)
            self.assertEqual([len(c["features"]) for c in a],[2,2,0])
            self.assertEqual(reads,[[1,2]])
            supply.encoders=lambda *_:self.fail("hit loaded encoder")
            with patch.object(catalog,"native_reader",side_effect=AssertionError("native read on hit")):
                b=list(supply.iter_points("test","b"))
            self.assertEqual([len(c["features"]) for c in b],[2,2,0])
            self.assertEqual(calls,["points","points"])
            self.assertEqual(points.snapshot()["entries"],2)
            # No trainable projection/LayerNorm or geometry appended to the stored feature64.
            np.testing.assert_array_equal(b[0]["features"],np.ones((2,64),np.float32))
            np.testing.assert_array_equal(b[0]["relative_xyz"][:,0],[-20,0])
            with self.assertRaises(ValueError):supply.points("test","a",3)

    def test_selection_rejects_test_or_unknown_locator(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/"selection.json"
            path.write_text(json.dumps({"tracks":[{"dataset":"test","locator":"not-eligible"}]}))
            with self.assertRaises(ValueError):selected_tracks({"test":self.catalog()},path)

    def test_catalog_uses_full_frozen_train_validation_and_never_opens_test_archives(self):
        from pedestrian_behavior.data.preparation import checksum
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            collection=root/"experiments/E001/reader-preparation/loki"
            setup=root/"experiments/E001/data-setup/loki/setup.json"
            (collection/"tracks").mkdir(parents=True);setup.parent.mkdir(parents=True)
            transform=root/"transform.json";transform.write_text("{}")
            manifest=collection/"manifest.json";manifest.write_text(json.dumps({"sources":{"transform":{}}}))
            entries=[]
            for split in ("training","validation","test"):
                entries.append({"locator":json.dumps({"scenario":split,"track_id":"a"}),"group":split,
                                "archive":split+".npz","slots":2,"split":split,"exclusions":[]})
                if split!="test":(collection/"tracks"/(split+".npz")).write_bytes(b"frozen-reference")
            setup.write_text(json.dumps({"status":"complete","collection":{"manifest_sha256":checksum(manifest)},
                                        "groups":{s:s for s in ("training","validation","test")},"tracks":entries}))
            def load(path):
                self.assertNotEqual(path.name,"test.npz")
                entry=next(e for e in entries if e["archive"]==path.name)
                return {"track_locator":entry["locator"],"source_frames":["0000","0002"],"source_time_us":[0,200000],
                        "anchor_valid":True,"availability":[{"native_3d":True},{"native_3d":False}]}, \
                       {"ped_position_valid":np.ones(2,bool),"ego_pose_valid":np.ones(2,bool)}
            with patch("pedestrian_behavior.experiments.e002a_features.load_track",side_effect=load):
                catalog=FeatureCatalog(root,"loki",{},transform)
            self.assertEqual({t["split"] for t in catalog.tracks.values()},{"training","validation"})
            self.assertEqual(len(catalog.tracks),2)
            for track in catalog.tracks.values():np.testing.assert_array_equal(track["point_usable"],[1,0])
            self.assertEqual(len(catalog.frames),2)

    def test_preparation_failure_retains_local_status_and_propagates_without_retry(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            args=SimpleNamespace(outputs_root=root,rgb_cache_root=root/"rgb",point_cache_root=root/"points",
                                 output=root/"experiments/E002a/failed",probe=root/"probe")
            with patch("pedestrian_behavior.experiments.e002a_features.load_configuration",side_effect=RuntimeError("asset failure")) as load:
                with self.assertRaisesRegex(RuntimeError,"asset failure"):run(args)
                self.assertEqual(load.call_count,1)
            record=json.loads((args.output/"cache.json").read_text())
            self.assertEqual(record["status"],"failed")
            self.assertIn("asset failure",record["failures"][0])
            self.assertEqual(record["config"]["output"],str(args.output))

    def test_cache_roots_cannot_overlap_data_or_each_other(self):
        with tempfile.TemporaryDirectory() as root:
            base=Path(root)
            args=SimpleNamespace(outputs_root=base,rgb_cache_root=base/"rgb",point_cache_root=base/"points",
                                 output=base/"experiments/E002a/check")
            validate_roots(args,{})
            args.point_cache_root=args.rgb_cache_root/"nested"
            with self.assertRaises(ValueError):validate_roots(args,{})
            args.point_cache_root=base/"experiments/E001/cache"
            with self.assertRaises(ValueError):validate_roots(args,{})

    def test_missing_loki_image_is_missing_rgb_not_a_header_open_failure(self):
        from pedestrian_behavior.inspection.native_features import native_camera
        module="pedestrian_behavior.inspection.native_features."
        reader=SimpleNamespace(dataset="loki",root=Path("unused"))
        with patch(module+"loki.frame_files",return_value=[("0000",{"image":None})]), \
             patch(module+"read_loki_boxes",return_value=({},{})) as boxes, \
             patch(module+"Image.open",side_effect=AssertionError("missing image opened")):
            self.assertEqual(list(native_camera(reader,"scene",["0000"],["a"],{"annotation_odometry_sha256":"frozen"})),[("0000",None,{})])
            self.assertEqual(boxes.call_args.args[2],set())

    def test_source_stat_change_invalidates_new_catalog_without_native_decode(self):
        with tempfile.TemporaryDirectory() as root:
            scene=Path(root)/"scene";scene.mkdir()
            source=scene/"image_0000.png";source.write_bytes(b"one")
            catalog=FeatureCatalog.__new__(FeatureCatalog)
            catalog.dataset="loki";catalog.config={"loki_root":root};catalog.manifest={};catalog.source_snapshots={}
            first=catalog.source_identity("scene")
            source.write_bytes(b"changed-content")
            catalog.source_snapshots={}
            self.assertNotEqual(first,catalog.source_identity("scene"))


class RNGTest(unittest.TestCase):
    def test_python_numpy_torch_backend_and_error_restoration(self):
        for fail in (False,True):
            random.seed(9);np.random.seed(9);torch.manual_seed(9)
            expected=(random.random(),np.random.rand(),torch.rand(3))
            random.seed(9);np.random.seed(9);torch.manual_seed(9)
            old=(torch.are_deterministic_algorithms_enabled(),torch.backends.cudnn.benchmark,
                 torch.backends.cudnn.deterministic,torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
            try:
                with isolated_extraction():
                    random.random();np.random.rand();torch.rand(10)
                    self.assertFalse(torch.are_deterministic_algorithms_enabled())
                    if fail:raise RuntimeError("extractor failure")
            except RuntimeError:
                self.assertTrue(fail)
            self.assertEqual(random.random(),expected[0]);self.assertEqual(np.random.rand(),expected[1])
            torch.testing.assert_close(torch.rand(3),expected[2],rtol=0,atol=0)
            self.assertEqual(old,(torch.are_deterministic_algorithms_enabled(),torch.backends.cudnn.benchmark,
                 torch.backends.cudnn.deterministic,torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32))

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA required")
    def test_cuda_rng_is_unchanged(self):
        torch.manual_seed(9);expected=torch.rand(10,device="cuda")
        torch.manual_seed(9)
        with isolated_extraction():torch.rand(100,device="cuda")
        torch.testing.assert_close(torch.rand(10,device="cuda"),expected,rtol=0,atol=0)

    def test_lazy_initialization_and_failed_inference_restore_rng(self):
        class Encoder:
            def __init__(self,*_):torch.rand(10)
            def __call__(self,_):torch.rand(10);return payload()
        with patch("pedestrian_behavior.experiments.e002a_features.ImageEncoder",Encoder):
            frozen=FrozenEncoders({"dino_code":"unused","dino_weights":"unused"},"cpu")
            before=torch.get_rng_state().clone()
            frozen("rgb",None);frozen("rgb",None)
            torch.testing.assert_close(torch.get_rng_state(),before,rtol=0,atol=0)
            self.assertEqual(frozen.loads,{"rgb":1,"points":0})
            frozen.release()


if __name__ == "__main__":
    unittest.main()
