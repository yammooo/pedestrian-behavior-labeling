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
from pedestrian_behavior.inspection.prediction_context import (ContextCache, bev_radius_pixels, native_frames,
                                                                render_context_frame)
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
                           "transition": np.array([0]*10+[2]),
                           "rgb-observed": np.array([1]*11), "lidar-observed": np.array([0, 1]*5+[0])},
            "source_flags": {"native_3d": np.ones(11, bool)}}


def run_fixture(root, count=4):
    outputs = root/"outputs"
    for i in range(count):
        directory = outputs/"experiments"/(["E001", "E001b", "E001c"][i%3])/"runs"/f"run{i}"
        write_json(directory/"config.json", {"source": "loki", "configuration": "test", "variant": "B",
                   "classes": CLASSES, "kind": "comparison", "settings": {"seed": i, "epochs": 75, "patience": 8}})
        write_json(directory/"status.json", {"status": "complete", "kind": "comparison"})
        target = directory/"loki-test"; target.mkdir()
        save_predictions(target/"predictions.npz", [prediction("p", correct=i%2 == 1), prediction("q", correct=i%2 == 1)])
    return outputs


def e002_fixture(root):
    outputs = root/"outputs"
    settings = {"seed": 0, "epochs": 100, "patience": 15}

    def write_run(directory, experiment, arm, *, variant=None, grouped=False, kind="exploratory-source-validation",
                  complete=True, predictions=None):
        model_variant = variant if experiment == "E002b" else None
        config = {"experiment": experiment, "source": "loki", "arm": arm, "configuration": arm,
                  "variant": "Transformer" if experiment == "E002a" else model_variant,
                  "classes": CLASSES, "settings": settings, "kind": kind, "smoke": False,
                  "held_out_tests": False}
        if model_variant is not None:
            config["model_variant"] = model_variant
        status = {"status": "complete" if complete else "incomplete", "kind": kind}
        if grouped:
            config.pop("kind")  # The actual grouped E002b child config relies on its status kind.
            config["execution"] = "shared-g-dditr"
            config["shared_execution_id"] = directory.parent.name
            write_json(directory.parent/"status.json", {"status": "complete", "execution": "shared-g-dditr",
                                                        "held_out_tests": False})
        write_json(directory/"config.json", config)
        write_json(directory/"status.json", status)
        write_json(directory/"provenance.json", {"selected_checkpoint_epoch": 3, "selection": {"epoch": 3}})
        result = directory/"loki-validation"; result.mkdir(parents=True)
        save_predictions(result/"predictions.npz", predictions or [prediction("p"), prediction("q", True)])

    write_run(outputs/"experiments/E002a/runs/road-g-rgb", "E002a", "g-rgb")
    write_run(outputs/"experiments/E002b/runs/loki-g-dditr", "E002b", "g-dditr", variant="norm")
    for variant in ("norm", "temporal-small", "branches-small"):
        write_run(outputs/f"experiments/E002b/runs/shared-loki/{variant}", "E002b", "g-dditr",
                  variant=variant, grouped=True)
    # Complete smoke and incomplete records must stay out of the viewer.
    write_run(outputs/"experiments/E002a/runs/road-g-smoke", "E002a", "g",
              kind="integration-smoke")
    write_json(outputs/"experiments/E002a/runs/road-g-smoke/config.json",
               json.loads((outputs/"experiments/E002a/runs/road-g-smoke/config.json").read_text()) | {"smoke": True})
    write_run(outputs/"experiments/E002a/runs/road-g-incomplete", "E002a", "g", complete=False)
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

    def test_e002_validation_catalog_shared_group_split_and_modality_axes(self):
        outputs = run_fixture(self.root)
        e002_fixture(self.root)
        axes = AXES | {"rgb-observed": ["missing", "valid"], "lidar-observed": ["missing", "valid"]}
        index = PredictionIndex(outputs, self.root/"cache", axes)
        test_runs = index.catalog()["runs"]
        validation = index.catalog("validation")["runs"]
        self.assertEqual(len(test_runs), 4)
        self.assertEqual({run["experiment"] for run in test_runs}, {"E001", "E001b", "E001c"})
        self.assertEqual(len(validation), 5)  # E002a, independent E002b, and three nested group variants.
        self.assertTrue(all(run["split"] == "validation" for run in validation))
        self.assertFalse(any("smoke" in run["relative"] or "incomplete" in run["relative"] for run in validation))
        transformer = next(run for run in validation if run["experiment"] == "E002a")
        self.assertEqual((transformer["architecture"], transformer["configuration"], transformer["selected_epoch"]),
                         ("Transformer", "g-rgb", 3))
        shared = next(run for run in validation if run.get("model_variant") == "temporal-small")
        self.assertEqual((shared["architecture"], shared["variant_label"]), ("Transformer", "Temporal small"))
        selected = [transformer["id"], shared["id"]]
        query = index.query(selected, "loki", split="validation")
        self.assertEqual((query["split"], query["common"]), ("validation", 2))
        self.assertEqual(query["axes"]["rgb-observed"], ["missing", "valid"])
        detail = index.detail(selected, "loki", query["rows"][0]["locator"], "validation")
        self.assertEqual(detail["split"], "validation")
        self.assertEqual(detail["source_times_us"], [0, 200000, 400000, 600000, 800000, 1000000,
                                                       1200000, 1400000, 1600000, 1800000, 2000000])
        with self.assertRaisesRegex(ValueError, "Unknown run, split"):
            index.query([test_runs[0]["id"]], "loki", split="validation")
        test_loaded = index.load(test_runs[0]["id"], "loki", "test")
        val_loaded = index.load(transformer["id"], "loki", "validation")
        test_cache = self.root/"cache/indices"/(__import__("hashlib").sha256(
            f"test:{test_loaded['sha256']}".encode()).hexdigest()+"-v2.json")
        val_cache = self.root/"cache/indices"/(__import__("hashlib").sha256(
            f"validation:{val_loaded['sha256']}".encode()).hexdigest()+"-v2.json")
        self.assertTrue(test_cache.is_file() and val_cache.is_file())
        self.assertNotEqual(test_cache, val_cache)

        # An axis disappears when a selected model does not carry the same saved condition.
        path = Path(shared["path"])/"loki-validation/predictions.npz"
        altered = [prediction("p"), prediction("q", True)]
        for track in altered:
            track["conditions"].pop("rgb-observed")
        save_predictions(path, altered)
        mismatch = index.query(selected, "loki", split="validation")
        self.assertNotIn("rgb-observed", mismatch["axes"])
        self.assertIn("lidar-observed", mismatch["axes"])

        altered[0]["targets"][0] = 1
        altered[0]["gt_valid"][0] = True
        save_predictions(path, altered)
        with self.assertRaisesRegex(ValueError, "Incompatible targets"):
            index.query(selected, "loki", split="validation")

    def test_checkpoint_evaluation_catalog_requires_complete_evaluation_parent(self):
        outputs = run_fixture(self.root)
        e002_fixture(self.root)
        epochs = {"norm": 11, "temporal-small": 12, "branches-small": 17}
        run_root = outputs/"experiments/E002b/runs/checkpoint-eval-loki-g-dditr-test"
        write_json(run_root/"status.json", {"status": "complete", "kind": "checkpoint-evaluation",
            "execution": "checkpoint-evaluation", "origin_execution": "shared-g-dditr",
            "evaluation_only": True, "training_state": "interrupted",
            "training_completed_epochs": 23, "held_out_tests": False, "variants": list(epochs)})
        for variant, epoch in epochs.items():
            child = run_root/variant
            write_json(child/"config.json", {"experiment": "E002b", "source": "loki", "arm": "g-dditr",
                "configuration": "g-dditr", "model_variant": variant, "classes": CLASSES,
                "settings": {"seed": 0, "epochs": 100, "patience": 15},
                "execution": "checkpoint-evaluation", "origin_execution": "shared-g-dditr",
                "evaluation_group_id": run_root.name, "kind": "checkpoint-evaluation", "evaluation_only": True,
                "training_state": "interrupted", "training_completed_epochs": 23,
                "selected_checkpoint_epoch": epoch, "held_out_tests": False,
                "origin_training_run": {"run_id": "original-"+variant, "run_url": "https://wandb.ai/original",
                    "selected_epoch": epoch, "completed_training_epochs": 23}})
            write_json(child/"status.json", {"status": "complete", "kind": "checkpoint-evaluation",
                "evaluation_only": True, "training_state": "interrupted", "training_completed_epochs": 23,
                "selected_checkpoint_epoch": epoch})
            write_json(child/"provenance.json", {"selection": {"epoch": epoch}})
            prediction_dir = child/"loki-validation"
            prediction_dir.mkdir(parents=True)
            save_predictions(prediction_dir/"predictions.npz", [prediction("p"), prediction("q", True)])

        incomplete = outputs/"experiments/E002b/runs/shared-loki-incomplete"
        write_json(incomplete/"status.json", {"status": "incomplete", "execution": "shared-g-dditr",
                                                "held_out_tests": False})
        child = incomplete/"norm"
        write_json(child/"config.json", {"experiment": "E002b", "source": "loki", "arm": "g-dditr",
            "model_variant": "norm", "classes": CLASSES, "settings": {"seed": 0, "epochs": 100, "patience": 15},
            "execution": "shared-g-dditr", "held_out_tests": False})
        write_json(child/"status.json", {"status": "complete", "kind": "exploratory-source-validation"})
        prediction_dir = child/"loki-validation"
        prediction_dir.mkdir(parents=True)
        save_predictions(prediction_dir/"predictions.npz", [prediction("p"), prediction("q", True)])

        index = PredictionIndex(outputs, self.root/"cache", AXES)
        validation = index.catalog("validation")["runs"]
        evaluated = [run for run in validation if run.get("evaluation_only")]
        self.assertEqual(len(evaluated), 3)
        self.assertEqual({run["model_variant"] for run in evaluated}, set(epochs))
        self.assertTrue(all(run["evaluation_label"] == "Evaluation only · interrupted after 23 epochs"
                            and run["training_state"] == "interrupted"
                            and run["training_completed_epochs"] == 23
                            and run["origin_training_run"]["run_id"].startswith("original-")
                            for run in evaluated))
        self.assertFalse(any("shared-loki-incomplete" in run["relative"] for run in validation))

    def test_review_split_persistence_and_legacy_default(self):
        outputs = run_fixture(self.root)
        e002_fixture(self.root)
        viewer = Viewer(outputs, self.root/"cache", AXES)
        self.addCleanup(viewer.close)
        tests = viewer.index.catalog()["runs"]
        test_runs = [run["id"] for run in tests[:2]]
        test_query = viewer.index.query(test_runs, "loki")
        locator = test_query["rows"][0]["locator"]
        legacy = {"runs": ",".join(test_runs), "dataset": "loki", "locator": locator, "frame": 2}
        old_record = viewer.save_case({"state": legacy, "hashes": test_query["hashes"], "note": "old test case"})
        self.assertEqual(old_record["split"], "test")
        self.assertEqual(old_record["state"]["split"], "test")
        self.assertEqual(viewer.notes()[0]["split"], "test")

        validation = viewer.index.catalog("validation")["runs"]
        val_runs = [next(run["id"] for run in validation if run["experiment"] == "E002a"),
                    next(run["id"] for run in validation if run.get("model_variant") == "norm")]
        val_query = viewer.index.query(val_runs, "loki", split="validation")
        val_state = {"runs": ",".join(val_runs), "dataset": "loki", "split": "validation",
                     "locator": val_query["rows"][0]["locator"], "frame": 3}
        val_record = viewer.save_case({"state": val_state, "hashes": val_query["hashes"], "note": "source validation"})
        self.assertEqual(val_record["split"], "validation")
        self.assertEqual(viewer.notes()[0]["split"], "test")
        self.assertEqual(viewer.notes()[1]["state"]["split"], "validation")
        with self.assertRaisesRegex(ValueError, "Unknown run, split"):
            viewer.index.detail(test_runs, "loki", locator, "validation")

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
        self.assertFalse(info["crop_3x"])
        image=Image.new('RGB',(100,100),'white')
        info=render_context_frame((image,None,[.1,.1,.3,.5],None,[.5,.5,.9,.9],[]),self.root,1)
        self.assertTrue(info['native_box2d']);self.assertTrue(info['road_box2d'])
        self.assertNotEqual((self.root/'0001-rgb.jpg').read_bytes(),(self.root/'0001-rgb-road.jpg').read_bytes())
        self.assertTrue(info["crop_3x"])
        self.assertIsNone(info["pedestrian_centered_bev_radius_m"])
        self.assertEqual(bev_radius_pixels(), 302)
        image = Image.new('RGB', (100,100), 'white')
        info = render_context_frame((image,None,[.4,.4,.6,.6],
                                     {"pos_x":1.,"pos_y":2.,"dim_x":.5,"dim_y":.5,"yaw":0.},None,[]),
                                    self.root, 2)
        self.assertTrue(info["crop_3x"])
        self.assertEqual(info["pedestrian_centered_bev_radius_m"], 20)
        with Image.open(self.root/"0002-crop-3x.jpg") as crop:
            self.assertEqual(crop.size, (256,256))
        from pedestrian_behavior.inspection.feature_probe import square_crop
        _, crop_info = square_crop(image, [.4,.4,.6,.6], expansion=3., output_size=256)
        np.testing.assert_allclose(crop_info["bounds_pixels"], [20,20,80,80])

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
