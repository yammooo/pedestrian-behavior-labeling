"""Source-only linear fitting, taxonomy control and scene-disjoint sampling."""

import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from pedestrian_behavior.experiments.e002a_linear_probe import frame_selection, restricted_predictions
from pedestrian_behavior.inspection.semantic_probe import evaluate, fit_linear, standardization
from pedestrian_behavior.training import reload_checkpoint


class SemanticProbeTest(unittest.TestCase):
    def test_even_frame_selection_preserves_scene_split_and_excludes_stress(self):
        samples = {"a":{"random":True,"split":"training"},"b":{"random":True,"split":"validation"},
                   "stress":{"random":False,"split":"training"}}
        rows = frame_selection(samples,{"a":list(range(30)),"b":list(range(40)),"stress":[1]})
        self.assertEqual([r["key"] for r in rows if r["group"]=="a"],[0,5,11,17,23,29])
        self.assertEqual(len(rows),12)
        self.assertEqual({r["group"] for r in rows if r["split"]=="validation"},{"b"})
        with self.assertRaises(ValueError):
            frame_selection(samples,{"a":[1],"b":list(range(40))})
        samples["b"]["split"]="test"
        with self.assertRaises(ValueError):
            frame_selection(samples,{"a":list(range(30)),"b":list(range(40))})

    def test_taxonomy_control_merges_probabilities_not_native_argmax(self):
        logits=np.full((2,16),-20.,dtype=np.float32)
        # Terrain dominates native argmax but is outside the common taxonomy.
        logits[0,13]=20;logits[0,12]=4;logits[0,6]=3
        # Two vehicle classes together beat pedestrian despite a lower native maximum.
        logits[1,[2,3]]=1;logits[1,6]=1.5
        np.testing.assert_array_equal(restricted_predictions(logits),[6,0])
        with self.assertRaises(ValueError):
            restricted_predictions(logits+np.nan)

    def test_normalization_uses_only_training_features_and_handles_constants(self):
        x=np.array([[1,7],[3,7]],dtype=np.float32)
        stats=standardization(x)
        np.testing.assert_allclose(stats['mean'],[2,7])
        np.testing.assert_allclose(stats['std'],[1,1e-6])
        self.assertEqual(stats['mean'].dtype,np.float32)
        with self.assertRaises(ValueError):
            standardization(x+np.inf)

    def test_linear_fit_unequal_batch_checkpoint_and_fixed_features(self):
        torch.set_num_threads(2)
        x=np.array([[-3],[-2],[-1],[1],[2],[3],[4]],dtype=np.float32)
        y=np.array([0,0,0,1,1,1,1],dtype=np.int64)
        before=x.copy();vx=np.array([[-2],[2]],dtype=np.float32);vy=np.array([0,1],dtype=np.int64)
        settings={'learning_rate':.1,'weight_decay':0.,'epochs':30,'patience':10,'batch_size':3}
        rows=[]
        with tempfile.TemporaryDirectory() as tmp:
            model,stats,result,pred=fit_linear(x,y,vx,vy,2,'cpu',tmp,settings,lambda row,history:rows.append(row))
            np.testing.assert_array_equal(pred,vy)
            np.testing.assert_array_equal(x,before)
            self.assertTrue(all(row['steps']==3 for row in rows))
            restored=torch.nn.Linear(1,2)
            checkpoint=reload_checkpoint(Path(tmp)/'best.pt',restored,'cpu')
            features=torch.from_numpy((vx-stats['mean'])/stats['std'])
            metrics,repeated=evaluate(restored,features,torch.from_numpy(vy),batch_size=1)
            np.testing.assert_array_equal(pred,repeated)
            self.assertEqual(checkpoint['selection'],result['selection'])
            self.assertEqual(metrics['accuracy'],1.)
            self.assertEqual((Path(tmp)/'last.pt').is_file(),True)
            self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
