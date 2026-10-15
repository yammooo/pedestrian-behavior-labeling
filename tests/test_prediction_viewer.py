"""Readable acceptance for prediction review, independent of trained weights."""

from http.server import ThreadingHTTPServer
import json
import math
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from pedestrian_behavior.data.preparation import checksum, load_track, prepare_collection
from pedestrian_behavior.evaluation import load_predictions, prediction_metrics, save_predictions
from pedestrian_behavior.inspection.prediction_context import ContextCache, native_frames, render_context_frame
from pedestrian_behavior.inspection.prediction_index import PredictionIndex, prepare_track, track_scores, write_json
from pedestrian_behavior.inspection.predictions import Viewer, handler
from test_track_preparation import native_readers, pose

CLASSES = ["MOVING", "STOPPED", "WAITING_TO_CROSS", "CROSSING"]
AXES = {"position": ["missing", "valid"], "transition": ["near", "steady", "unknown"]}


def prediction(name="p", correct=False):
    targets = np.array([0]+[1]*9+[-1])
    logits = np.tile([math.log(.75), math.log(.25), -100., -100.], (11, 1)).astype(np.float32)
    logits[0] = [0, 0, -100, -100]
    if correct:
        logits[1:10, :2] = [math.log(.25), math.log(.75)]
    return {"locator": json.dumps({"scenario": "scene", "track_id": name}, sort_keys=True),
            "group": "scene", "archive": "unavailable/"+name+".npz", "logits": logits,
            "targets": targets, "gt_valid": targets >= 0,
            "source_times_us": [i*200000 for i in range(11)], "grid_times_us": [i*200000 for i in range(11)],
            "conditions": {"position": np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]),
                           "transition": np.array([0]*10+[2])}, "source_flags": {"native_3d": np.ones(11, bool)}}


def run_fixture(root, count=4):
    outputs = root/"outputs"
    for i in range(count):
        directory = outputs/"experiments"/(["E001", "E001b", "E001c"][i%3])/"runs"/f"run{i}"
        write_json(directory/"config.json", {"source": "loki", "configuration": "test", "variant": "B",
                   "classes": CLASSES, "kind": "comparison", "settings": {"seed": i, "epochs": 75, "patience": 8}})
        write_json(directory/"status.json", {"status": "complete"})
        target = directory/"loki-test"; target.mkdir()
        save_predictions(target/"predictions.npz", [prediction("p", correct=i%2 == 1), prediction("q", correct=i%2 == 1)])
    return outputs


class PredictionViewerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_per_track_scores_supported_classes_and_unlabeled_context(self):
        track = prepare_track(prediction(), 4)
        result = track_scores(track)
        self.assertAlmostEqual(result["accuracy"], .1)
        self.assertAlmostEqual(result["macro_f1"], 1/11)
        self.assertAlmostEqual(result["ce"], (math.log(2)+9*math.log(4))/10, places=6)
        self.assertEqual(result["support"], [1, 9, 0, 0])
        selected = track_scores(track, track["conditions"]["position"] == 0)
        self.assertAlmostEqual(selected["accuracy"], 1)
        self.assertAlmostEqual(selected["macro_f1"], 1)
        self.assertAlmostEqual(selected["ce"], math.log(2), places=6)
        empty = track_scores(track, track["conditions"]["transition"] == 2)
        self.assertEqual([empty[k] for k in ("accuracy", "macro_f1", "ce")], [None]*3)
        track["logits"][-1] = [10000, -10000, 0, 0]
        np.testing.assert_equal(track_scores(track), result)
        invalid = prediction(); invalid["logits"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Invalid"):
            prepare_track(invalid, 4)

    def test_catalog_four_models_scopes_confusions_sorting_and_cache(self):
        outputs = run_fixture(self.root)
        index = PredictionIndex(outputs, self.root/"cache", AXES)
        catalog = index.catalog()["runs"]
        self.assertEqual(len(catalog), 4)
        runs = [r["id"] for r in sorted(catalog, key=lambda r:r["seed"])]
        query = index.query(runs, "loki")
        self.assertEqual((query["common"], query["total"]), (2, 2))
        self.assertEqual(query["rows"][0]["values"], [.1, 1, .1, 1])
        delta = index.query(runs, "loki", sort="1-0")
        self.assertAlmostEqual(delta["rows"][0]["rank"], .9)
        matching = index.query(runs, "loki", axis="position", value=0)
        self.assertEqual(matching["rows"][0]["values"], [1]*4)
        self.assertEqual(matching["rows"][0]["matching_gt"], 1)
        whole = index.query(runs, "loki", axis="position", value=0, scope="whole")
        self.assertEqual(whole["rows"][0]["scores"][0]["gt_count"], 10)
        self.assertEqual(whole["rows"][0]["matching_gt"], 1)
        empty = index.query(runs, "loki", axis="transition", value=2)
        self.assertEqual(empty["rows"][0]["values"], [None]*4)
        self.assertEqual(index.query(runs, "loki", axis="transition", value=2, minimum=1)["total"], 0)
        confusion = index.query(runs, "loki", metric="confusion_rate", confusion=(1, 0))
        self.assertEqual(confusion["rows"][0]["values"], [1, 0, 1, 0])
        self.assertEqual(confusion["rows"][0]["scores"][0]["confusion_errors"], 9)
        self.assertEqual(index.query(runs, "loki", search='"q"')["total"], 1)
        # Full prediction scores agree with the existing experiment operation.
        loaded = index.load(runs[0], "loki")
        aggregate = prediction_metrics(list(loaded["tracks"].values()), 4)
        self.assertAlmostEqual(aggregate["mean_track_accuracy"], .1)
        self.assertTrue(list((self.root/"cache/indices").glob("*.json")))
        other_index = PredictionIndex(outputs, self.root/"cache", AXES);other_index.catalog()
        with patch("pedestrian_behavior.inspection.prediction_index.track_scores", side_effect=AssertionError("cache not reused")):
            other_index.load(runs[0], "loki")
        archive = Path(catalog[0]["path"])/"loki-test/predictions.npz"
        save_predictions(archive, [prediction("p", True)])
        changed = index.load(catalog[0]["id"], "loki")
        self.assertNotEqual(changed["sha256"], loaded["sha256"])
        self.assertEqual(len(changed["tracks"]), 1)

    def test_alignment_common_population_and_condition_mismatch(self):
        outputs=run_fixture(self.root, 2);index=PredictionIndex(outputs,self.root/"cache",AXES)
        runs=[r["id"] for r in index.catalog()["runs"]]
        path=Path(index.runs[runs[1]]["path"])/"loki-test/predictions.npz"
        save_predictions(path,[prediction("p")])
        query=index.query(runs,"loki")
        self.assertEqual(query["missing"],[1,0])
        changed=prediction("p");changed["source_times_us"][0]=1;save_predictions(path,[changed])
        with self.assertRaisesRegex(ValueError,"Incompatible source_times"):
            index.query(runs,"loki")
        changed=prediction("p");changed["conditions"]["position"][:]=1;save_predictions(path,[changed])
        self.assertNotIn("position",index.query(runs,"loki")["axes"])
        with self.assertRaisesRegex(ValueError,"Condition unavailable"):
            index.query(runs,"loki",axis="position")
        with self.assertRaisesRegex(ValueError,"distinct"):
            index.query([runs[0]]*2,"loki")

    def test_npz_fields_are_read_once_and_results_remain_independent(self):
        path=self.root/"predictions.npz";save_predictions(path,[prediction("p"),prediction("q")])
        import numpy.lib.npyio as npyio
        calls=[];original=npyio.NpzFile.__getitem__
        def read(saved,key):
            calls.append(key);return original(saved,key)
        with patch.object(npyio.NpzFile,"__getitem__",read):
            tracks=load_predictions(path)
        self.assertEqual(len(calls),len(set(calls)))
        tracks[0]["logits"][0,0]=999
        self.assertNotEqual(tracks[1]["logits"][0,0],999)

    def test_http_review_persistence_errors_and_media_is_read_only(self):
        outputs=run_fixture(self.root);before={p:checksum(p) for p in outputs.rglob("*") if p.is_file()}
        viewer=Viewer(outputs,self.root/"cache",AXES);self.addCleanup(viewer.close)
        server=ThreadingHTTPServer(("127.0.0.1",0),handler(viewer));self.addCleanup(server.server_close)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();self.addCleanup(server.shutdown)
        base=f"http://127.0.0.1:{server.server_port}"
        def get(path):
            with urlopen(base+path) as response:return json.load(response)
        runs=[r["id"] for r in get('/api/runs')["runs"]]
        data=get('/api/tracks?runs='+','.join(runs))
        state={"runs":','.join(runs),"dataset":"loki","locator":data["rows"][0]["locator"],"frame":2}
        body={"state":state,"hashes":data["hashes"],"note":"Check boundary <not HTML>","reason":"diagnostic"}
        request=Request(base+'/api/reviews',json.dumps(body).encode(),{'Content-Type':'application/json'})
        with urlopen(request) as response:self.assertEqual(response.status,201)
        saved=get('/api/reviews');self.assertEqual(saved[0]["note"],body["note"])
        body["hashes"]=['changed']*4
        with self.assertRaises(HTTPError):urlopen(Request(base+'/api/reviews',json.dumps(body).encode(),{'Content-Type':'application/json'}))
        with self.assertRaises(HTTPError):urlopen(base+'/media?key=../x&image=../../file')
        with self.assertRaises(HTTPError):urlopen(Request(base+'/api/reviews',b'{}',{'Content-Type':'application/json','Origin':'https://example.com'}))
        self.assertEqual(before,{p:checksum(p) for p in outputs.rglob("*") if p.is_file()})
        with urlopen(base+'/') as response:
            self.assertEqual(response.status,200)
            self.assertIn(b'Ground truth',response.read())

    def test_native_context_missing_rgb_and_waymo_never_uses_road_box(self):
        # Native frame with ROAD-only 2D: no Waymo crop or yellow-box fallback.
        class Reader:
            dataset='road-waymo';_scene_id='clip';waymo_root=None
            scenes={'clip':{'segment_context_name':'scene','component_paths':{}}}
            camera={};lidar={};poses={};_tracks={'p':{'laser_id':None}}
            road={'clip':{'p':{10:{'annotation':{'box':[.1,.2,.3,.4]}}}}}
            _review_boxes_scene='clip';_review_boxes={('p',10):[.1,.2,.3,.4]}
        metadata={"track_locator":json.dumps({"clip":"clip","tube_uid":"p"}),"source_frames":[10,None]}
        frames=list(native_frames(Reader(),metadata))
        self.assertIsNone(frames[0][2]);self.assertEqual(frames[0][4],[.1,.2,.3,.4])
        self.assertIsNone(frames[1][0])
        info=render_context_frame(frames[0],self.root,0)
        self.assertFalse(info['native_box2d']);self.assertFalse(info['rgb']);self.assertFalse(info['lidar'])
        image=Image.new('RGB',(100,100),'white')
        info=render_context_frame((image,None,[.1,.1,.3,.5],None,[.5,.5,.9,.9],[]),self.root,1)
        self.assertTrue(info['native_box2d']);self.assertTrue(info['road_box2d'])
        self.assertNotEqual((self.root/'0001-rgb.jpg').read_bytes(),(self.root/'0001-rgb-road.jpg').read_bytes())

    def test_media_worker_reuses_context_across_models_and_manual_retry(self):
        viewer=Viewer(run_fixture(self.root,2),self.root/'cache',AXES);self.addCleanup(viewer.close)
        runs=[r['id'] for r in viewer.index.catalog()['runs']]
        locator=next(iter(viewer.index.load(runs[0],'loki')['tracks']))
        ready=threading.Event();release=threading.Event()
        def prepare(*args,**kwargs):
            ready.set();self.assertTrue(release.wait(2));return {'key':'test','frames':[]}
        with patch.object(viewer.context,'prepare',side_effect=prepare) as render:
            first=viewer.media(runs[0],'loki',locator);self.assertTrue(ready.wait(2))
            other=viewer.media(runs[1],'loki',locator)
            self.assertEqual(first['job'],other['job'])
            release.set();viewer.futures[first['job']].result(timeout=2)
            self.assertEqual(viewer.media(runs[1],'loki',locator)['status'],'ready')
            self.assertEqual(render.call_count,1)
        viewer.jobs.clear();viewer.futures.clear()
        with patch.object(viewer.context,'prepare',side_effect=FileNotFoundError('unmounted')) as render:
            first=viewer.media(runs[0],'loki',locator);viewer.futures[first['job']].result(timeout=2)
            self.assertEqual(viewer.jobs[first['job']]['status'],'unavailable')
            self.assertEqual(render.call_count,1)
        with patch.object(viewer.context,'prepare',return_value={'key':'remounted'}) as render:
            retry=viewer.media(runs[0],'loki',locator);viewer.futures[retry['job']].result(timeout=2)
            self.assertEqual(viewer.jobs[retry['job']]['status'],'ready')
            self.assertEqual(render.call_count,1)

    def test_native_loki_cache_alignment_reuse_and_source_invalidation(self):
        reader,_=native_readers(self.root,[0,200000],[pose(),pose()],{'p':{0:[1,2,0],1:[2,2,0]}})
        scenario=reader.root/'scenario_000'
        for key in ('0000','0002'):Image.new('RGB',(100,100),'white').save(scenario/f'image_{key}.png')
        collection=self.root/'outputs/experiments/E001/reader-preparation/loki';prepare_collection(reader,collection)
        archive=next((collection/'tracks').glob('*.npz'));metadata,_=load_track(archive)
        cache=ContextCache(self.root/'outputs',self.root/'cache',loki_root=reader.root)
        counts=[];first=cache.prepare('loki',str(archive),metadata['track_locator'],lambda i,n:counts.append((i,n)))
        self.assertEqual(counts,[(1,2),(2,2)])
        self.assertTrue(first['frames'][0]['rgb']);self.assertFalse(first['frames'][0]['lidar'])
        with self.assertRaisesRegex(ValueError,'timestamps differ'):
            cache.prepare('loki',str(archive),metadata['track_locator'],lambda *_:None,
                          expected={'grid_times_us':[1,2],'source_times_us':[0,200000]})
        with patch('pedestrian_behavior.inspection.prediction_context.native_frames',side_effect=AssertionError('decoded again')):
            second=cache.prepare('loki',str(archive),metadata['track_locator'],lambda *_:None)
        self.assertTrue(second['cache_hit']);self.assertEqual(first['key'],second['key'])
        Image.new('RGB',(100,100),'black').save(scenario/'image_0000.png')
        third=cache.prepare('loki',str(archive),metadata['track_locator'],lambda *_:None)
        self.assertNotEqual(first['key'],third['key'])


if __name__=='__main__':unittest.main()
