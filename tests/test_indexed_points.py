"""Acceptance for target-specific resolution, immutable shards and selective integrity."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from pedestrian_behavior.data.indexed_points import build_frame, IndexedShard, ShardWriter, CHUNK_ROWS


def fixture():
    xyz = np.array([[.04, 0, 0], [.14, 0, 0], [1.04, 0, 0], [1.14, 0, 0], [19.99, 0, 0], [20.01, 0, 0]], np.float32)
    return {"coordinates": xyz, "features": np.arange(len(xyz)*64, dtype=np.float32).reshape(-1,64),
        "native_indices": np.arange(len(xyz), dtype=np.int64), "world_from_native": np.eye(4),
        "boxes": np.array([[.04,0,0,.1,.1,.1,0], [1.04,0,0,.1,.1,.1,0]], np.float64),
        "anchors": np.repeat(np.eye(4)[None],2,0), "locators": np.array(["a", "b"])}


def selected(payload, metadata, target):
    s = metadata["selections"][target]
    return payload["indices"][s["index_start"]:s["index_start"]+s["count"]]


class IndexedPointsTest(unittest.TestCase):
    def test_two_nearby_targets_choose_own_fine_and_other_coarse(self):
        arrays = fixture()
        payload, metadata = build_frame(arrays)
        for i in range(2):
            rows = selected(payload, metadata, i)
            fine = payload["fine"][rows].astype(bool)
            np.testing.assert_array_equal(payload["native_reference"][rows[fine]], [2*i, 2*i+1])
            self.assertFalse(np.intersect1d(payload["cell"][rows[fine]], payload["cell"][rows[~fine]]).size)
        # Adding another target changes stored rows, but not this target's supplied values.
        one = {**arrays, "boxes": arrays["boxes"][:1], "anchors": arrays["anchors"][:1], "locators": arrays["locators"][:1]}
        single, sm = build_frame(one)
        r0, r1 = selected(single, sm, 0), selected(payload, metadata, 0)
        np.testing.assert_array_equal(single["coordinates"][r0], payload["coordinates"][r1])
        np.testing.assert_array_equal(single["features"][r0], payload["features"][r1])

    def test_negative_grid_rotated_box_and_empty(self):
        arrays = fixture()
        arrays["coordinates"][:,0] *= -1
        arrays["boxes"][:,0] *= -1
        arrays["boxes"][:,6] = np.pi/2
        payload, meta = build_frame(arrays)
        self.assertEqual(payload["features"].dtype, np.float16)
        self.assertEqual(payload["indices"].dtype, np.uint32)
        self.assertTrue(all(s["original_target_tokens"] == 1 for s in meta["selections"]))
        empty = {**fixture(), "coordinates": np.zeros((0,3),np.float32), "features":np.zeros((0,64),np.float32), "native_indices":np.zeros(0,np.int64)}
        with tempfile.TemporaryDirectory() as folder:
            writer = ShardWriter(Path(folder)/"empty", {})
            writer.add("0", empty)
            sha = writer.finish()
            reader = IndexedShard(writer.root, sha)
            self.assertEqual(reader.crop("0","a").shape,(0,68))
            reader.close()

    def test_shared_shard_roundtrip_multiple_frames_and_fresh_reader(self):
        with tempfile.TemporaryDirectory() as folder:
            writer = ShardWriter(Path(folder)/"scene", {"test":True})
            writer.add("1", fixture()); writer.add("2", fixture())
            with self.assertRaises(ValueError): writer.add("2", fixture())
            sha = writer.finish()
            expected, meta = build_frame(fixture())
            for _ in range(2):
                reader = IndexedShard(writer.root, sha, max_verified_chunks=2)
                for frame in ("1","2"):
                    actual = reader.crop(frame,"a")
                    rows = selected(expected,meta,0)
                    np.testing.assert_array_equal(actual[:,:64], expected["features"][rows].astype(np.float32))
                    np.testing.assert_allclose(actual[:,64:67], (expected["coordinates"][rows]-fixture()["boxes"][0,:3])/20,atol=1e-6)
                self.assertLessEqual(len(reader.verified),2)
                with self.assertRaises(ValueError):reader.crop("1","unknown")
                reader.close()

    def test_corrupt_truncated_manifest_and_later_mutation_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            writer = ShardWriter(Path(folder)/"scene", {})
            writer.add("0", fixture()); sha = writer.finish()
            with self.assertRaises(ValueError):IndexedShard(writer.root,"0"*64)
            reader = IndexedShard(writer.root,sha)
            reader.crop("0","a")
            path = writer.root/"features.bin"
            with path.open("r+b") as stream:stream.write(b"xx")
            with self.assertRaisesRegex(ValueError,"changed"):reader.crop("0","a")
            reader.close()
            reader = IndexedShard(writer.root,sha)
            with self.assertRaisesRegex(ValueError,"checksum"):reader.crop("0","a")
            reader.close()
            path.write_bytes(b"")
            with self.assertRaisesRegex(ValueError,"Truncated"):IndexedShard(writer.root,sha)

    def test_chunk_verification_is_selective_and_unfinished_shard_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            writer = ShardWriter(Path(folder)/"scene", {})
            with self.assertRaises(FileNotFoundError):IndexedShard(writer.root,"0"*64)
            arrays = fixture()
            # Space rows apart so the target only gathers one distant subset of a large table.
            n = CHUNK_ROWS*3
            arrays["coordinates"] = np.c_[np.arange(n)*.3, np.zeros((n,2))].astype(np.float32)
            arrays["features"] = np.ones((n,64),np.float32)
            arrays["native_indices"] = np.arange(n,dtype=np.int64)
            writer.add("0", arrays); sha = writer.finish()
            reader = IndexedShard(writer.root,sha)
            reader.crop("0","a")
            self.assertLess(reader.stats["verified_bytes"],sum(v["bytes"] for v in reader.record["arrays"].values()))
            checked = reader.stats["checked_chunks"]
            reader.crop("0","a")
            self.assertEqual(reader.stats["checked_chunks"],checked)
            reader.close()

    def test_range_boundary_nonfinite_and_half_overflow(self):
        arrays = fixture()
        arrays["boxes"][:,0] = 0
        arrays["coordinates"][0] = [20,0,0]
        arrays["coordinates"][1] = [20.25,0,0]
        payload,meta = build_frame(arrays)
        rows=selected(payload,meta,0)
        self.assertTrue((np.linalg.norm(payload["coordinates"][rows,:2],axis=1)<=20).all())
        arrays["features"][0,0]=np.inf
        with self.assertRaises(ValueError):build_frame(arrays)
        arrays["features"][0,0]=1e10
        with np.errstate(over="ignore"), self.assertRaisesRegex(ValueError,"FP16"):
            build_frame(arrays)


if __name__ == "__main__":
    unittest.main()
