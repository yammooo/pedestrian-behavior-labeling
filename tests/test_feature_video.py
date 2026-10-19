"""Known crop geometry, metric radii, missing observations and track selection."""

import unittest

import numpy as np
from PIL import Image, ImageColor

from pedestrian_behavior.experiments.e002a_videos import choose_tracks
from pedestrian_behavior.inspection.feature_probe import square_crop
from pedestrian_behavior.inspection.feature_video import (RING_COLORS, VIDEO_SIZE, bev_pixels,
    crop_outlines, render_track_frame)


class FeatureVideoTest(unittest.TestCase):
    def test_nested_rgb_windows_and_metric_centering(self):
        np.testing.assert_allclose(crop_outlines(), [(160,480),(320/3,1600/3),(0,640)])
        np.testing.assert_allclose(bev_pixels(np.array([[0,0,0],[15,0,0],[0,-20,1],[-30,0,-1]])),
                                   [[320,320],[320,170],[520,320],[320,620]])
        image=Image.new("RGB",(200,100),"blue")
        small,info=square_crop(image,[.4,.2,.6,.8],expansion=3)
        big,other=square_crop(image,[.4,.2,.6,.8],expansion=3,output_size=640)
        self.assertEqual(small.shape,(256,256,3))
        self.assertEqual(big.shape,(640,640,3))
        self.assertEqual(info,other)
        with self.assertRaises(ValueError):
            square_crop(image,[.4,.2,.6,.8],output_size=0)

    def test_render_outlines_boundaries_and_missing_frame(self):
        identity=np.eye(4)
        frame={"image":Image.new("RGB",(200,100),"blue"),"camera":{"target":[.4,.2,.6,.8]},
               "boxes":{"target":(np.zeros(3),np.array([2,1,2]),0.)},"world_from_native":identity}
        palette=np.array([[255,0,0],[0,255,0],[0,0,255],[255,255,0]],dtype=np.uint8)
        offsets=np.array([[0,0,0],[15,0,0],[20.00001,0,0],[31,0,0]])
        image,evidence=render_track_frame(frame,"target",identity,offsets,np.arange(4),palette,
                                          ["a","b","c","d"],"Known geometry")
        self.assertEqual(image.size,VIDEO_SIZE)
        self.assertEqual(evidence["points_within_radii"],[2,2,3])
        self.assertTrue(evidence["rgb_valid"] and evidence["lidar_valid"])
        self.assertEqual(image.getpixel((25+160,90+320)),ImageColor.getrgb(RING_COLORS[0]))
        self.assertEqual(image.getpixel((760+320,90+170)),ImageColor.getrgb(RING_COLORS[0]))
        missing,evidence=render_track_frame(None,"target",identity,None,None,palette,["a","b","c","d"],"Missing")
        self.assertEqual(missing.size,VIDEO_SIZE)
        self.assertFalse(evidence["rgb_valid"] or evidence["lidar_valid"])
        self.assertEqual(evidence["points_within_radii"],[0,0,0])
        with self.assertRaises(ValueError):
            render_track_frame(frame,"target",identity,offsets+np.nan,np.arange(4),palette,["a","b","c","d"],"Bad")

    def test_track_selection_prefers_distinct_groups_and_uses_no_labels(self):
        candidates=[{"group":g,"locator":str(i),"paired_slots":n,"slots":100}
                    for i,(g,n) in enumerate((("a",100),("a",99),("b",90),("c",80),("d",70),("e",60)))]
        selected=choose_tracks(candidates)
        self.assertEqual([c["group"] for c in selected],["a","b","c","d","e"])
        self.assertEqual(len(choose_tracks(candidates[:5])),5)
        with self.assertRaises(ValueError):
            choose_tracks(candidates[:4])


if __name__=="__main__":
    unittest.main()
