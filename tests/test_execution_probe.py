"""Memory paths must preserve information and trainable-readout gradients."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pedestrian_behavior.inspection.execution_probe import (ExecutionGraph, compare_gradients,
                                                            crop_tokens, rotary, select_tracks)
from pedestrian_behavior.inspection.native_features import native_scene


class ExecutionProbeTest(unittest.TestCase):
    def test_development_length_selection_never_uses_behavior(self):
        entries = [{"slots":n,"locator":str(n),"behavior":"irrelevant"} for n in (100, 1, 20, 50, 10)]
        self.assertEqual([(name,e["slots"]) for name,e in select_tracks(entries)],
                         [("short",1),("typical",20),("long",100)])
        with self.assertRaises(ValueError):select_tracks(entries[:2])

    def test_all_voxels_in_horizontal_crop_without_height_filter_or_cap(self):
        offsets = np.array([[0,0,100],[20,0,0],[20.001,0,0],[0,-20,-100]],np.float32)
        features = np.arange(4*64,dtype=np.float32).reshape(4,64)
        tokens,keep = crop_tokens(features,np.zeros((4,3)),offsets,np.array([1,0,0,0],bool))
        np.testing.assert_array_equal(keep,[1,1,0,1])
        np.testing.assert_array_equal(tokens[:,:64],features[keep])
        np.testing.assert_allclose(tokens[:,64:67],offsets[keep]/20)
        with self.assertRaises(ValueError):crop_tokens(features+np.nan,np.zeros((4,3)),offsets,np.zeros(4,bool))

    def test_rotary_preserves_norm_and_signed_relative_attention(self):
        torch.manual_seed(5)
        q=torch.randn(1,4,5,64);k=torch.randn(1,4,5,64)
        torch.testing.assert_close(rotary(q).norm(dim=-1),q.norm(dim=-1))
        # The score for the same content depends on offset, not track-start position.
        torch.testing.assert_close((rotary(q)[:,:,1]*rotary(k)[:,:,3]).sum(-1),
                                   (rotary(q[:,:,1:])[:,:,0]*rotary(k[:,:,1:])[:,:,2]).sum(-1))

    def test_all_memory_paths_preserve_outputs_gradients_and_update(self):
        old=torch.get_num_threads();torch.set_num_threads(2)
        try:
            torch.manual_seed(8)
            frames=[torch.randn(9,68),torch.zeros(0,68),torch.randn(13,68)]
            errors=compare_gradients(frames,torch.randn(3,768),torch.tensor([1.,0.,1.]),"cpu")
            self.assertLess(max(errors.values()),1e-5)
            graph=ExecutionGraph()
            for mode in ("ordinary","recompute","offload"):
                with self.assertRaises(ValueError):graph([frames[0]+torch.nan],torch.zeros(1,768),torch.ones(1),mode=mode)
        finally:torch.set_num_threads(old)

    def test_lidar_only_native_read_never_opens_camera(self):
        reader=SimpleNamespace(dataset="loki",root=Path("unused"),extrinsic=np.eye(4),
                               read_frames=lambda *_:{"1":{"ego_pose":np.eye(4)}})
        frame=SimpleNamespace(paths={"pointcloud":None,"image":"must-not-be-opened"},pedestrians={})
        module="pedestrian_behavior.inspection.native_features."
        with patch(module+"read_loki_boxes",side_effect=AssertionError("camera annotation read")), \
             patch(module+"loki.frame_files",return_value=[("1",{})]), \
             patch(module+"loki.read_frame",return_value=frame), \
             patch(module+"Image.open",side_effect=AssertionError("image read")):
            result=list(native_scene(reader,"scene",["1"],[],{},include_rgb=False))
        self.assertEqual(result[0]["camera"],{})
        self.assertIsNone(result[0]["image"])


if __name__ == "__main__":
    unittest.main()
