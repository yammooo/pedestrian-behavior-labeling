"""Bounded semantic transfer: native range correspondence and honest scoring."""

import unittest

import numpy as np

from pedestrian_behavior.datasets.road_waymo import range_points, range_semantics
from pedestrian_behavior.experiments.e002a_semantic_check import aligned_semantics, mapped_labels, NAMES, PALETTE
from pedestrian_behavior.inspection.segmentation import semantic_bev, semantic_confusion, semantic_metrics


class SemanticCheckTest(unittest.TestCase):
    def test_range_labels_use_exact_range_mask_and_semantic_channel(self):
        values = np.array([[2,7,0,0],[0,8,0,0],[np.nan,9,0,0],[4,11,0,0]])
        labels = np.array([[101,7],[102,1],[103,2],[104,18]])
        np.testing.assert_array_equal(range_semantics(labels.ravel(),[1,4,2],values.ravel(),[1,4,4]),[7,18])
        c = "[LiDARCalibrationComponent]."
        calibration = {c+"extrinsic.transform":np.eye(4).ravel(),c+"beam_inclination.values":[0]}
        points = range_points(values.ravel(),[1,4,4],calibration,with_strength=True)
        np.testing.assert_array_equal(points[:,3],[7,11])
        with self.assertRaises(ValueError):
            range_semantics(labels.ravel(),[2,2,2],values.ravel(),[1,4,4])
        with self.assertRaises(ValueError):
            range_semantics(labels.astype(float).ravel(),[1,4,2],values.ravel(),[1,4,4])

    def test_laser_return_order_retains_unannotated_context(self):
        def scan(laser):
            r={"key.laser_name":laser}
            for ret in (1,2):
                p=f"[LiDARComponent].range_image_return{ret}."
                r[p+"values"]=[1,0,0,0,0,0,0,0,2,0,0,0]
                r[p+"shape"]=[1,3,4]
            return r
        seg={"key.laser_name":1}
        for ret, ids in ((1,[7,0,18]),(2,[22,0,15])):
            p=f"[LiDARSegmentationLabelComponent].range_image_return{ret}."
            seg[p+"values"]=np.c_[[90,91,92],ids].ravel()
            seg[p+"shape"]=[1,3,2]
        gt, parts=aligned_semantics([scan(2),scan(1)],[seg])
        np.testing.assert_array_equal(gt,[-1,-1,-1,-1,7,18,22,15])
        self.assertEqual([(p['laser'],p['return'],p['start'],p['stop']) for p in parts],
                         [(2,1,0,2),(2,2,2,4),(1,1,4,6),(1,2,6,8)])
        np.testing.assert_array_equal(parts[2]['valid_flat_pixels'],[0,2])
        with self.assertRaises(ValueError):
            aligned_semantics([scan(2)],[seg])
        with self.assertRaises(ValueError):
            aligned_semantics([scan(1)],[seg,seg])

    def test_mapping_does_not_drop_unmatched_prediction_errors(self):
        gt,pred=mapped_labels(np.array([7,18,19,16,0,21,-1]),np.array([6,11,10,15,6,12,6]))
        np.testing.assert_array_equal(gt,[1,5,5,8,-1,-1,-1])
        np.testing.assert_array_equal(pred,[1,9,5,8,1,6,1])
        cm=semantic_confusion(gt,pred,9)
        self.assertEqual(cm.sum(),4)
        self.assertEqual(cm[5,9],1)
        self.assertEqual(cm[:,1].sum(),1)
        with self.assertRaises(ValueError):
            mapped_labels(np.array([23]),np.array([0]))

    def test_hand_computed_metrics_keep_false_positives_and_empty_classes(self):
        cm=semantic_confusion(np.array([0,0,1,1,-1]),np.array([0,3,0,1,1]),3)
        m=semantic_metrics(cm)
        self.assertEqual(m['points'],4)
        self.assertEqual(m['accuracy'],.5)
        self.assertEqual(m['supported_classes'],2)
        self.assertAlmostEqual(m['classes'][0]['iou'],1/3)
        self.assertEqual(m['classes'][1]['iou'],.5)
        self.assertAlmostEqual(m['supported_macro_iou'],5/12)
        self.assertIsNone(m['classes'][2]['iou'])
        self.assertEqual(m['unmatched_predictions'],1)
        empty=semantic_metrics(np.zeros((3,4),dtype=int))
        self.assertIsNone(empty['accuracy'])
        self.assertIsNone(empty['supported_macro_iou'])
        with self.assertRaises(ValueError):
            semantic_confusion(np.array([3]),np.array([0]),3)

    def test_bev_matching_pixels_and_malformed_coordinates(self):
        xyz=np.array([[0,0,0],[10,0,0],[0,10,0]],dtype=np.float32)
        image=semantic_bev(xyz,np.array([1,0,-1]),np.array([1,9,2]),NAMES,PALETTE,'test')
        self.assertEqual(image.size,(1980,850))
        # The center point has identical class color in GT and prediction panels.
        self.assertEqual(image.getpixel((340,400)),tuple(PALETTE[1]))
        self.assertEqual(image.getpixel((1000,400)),tuple(PALETTE[1]))
        with self.assertRaises(ValueError):
            semantic_bev(xyz+np.nan,np.array([1,0,-1]),np.array([1,9,2]),NAMES,PALETTE,'test')
