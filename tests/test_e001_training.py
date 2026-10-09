"""Hand-calculated loss/metrics, selection, condition and artifact acceptance."""

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

# Set before earlier CUDA tests create cuBLAS handles in this shared process.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
import wandb

from pedestrian_behavior.data.loading import collate_tracks
from pedestrian_behavior.evaluation import (load_predictions, prediction_metrics, save_predictions,
                                            stratified_metrics, track_cross_entropy)
from pedestrian_behavior.experiments.e001 import STRATA, build_model, observation_conditions
from pedestrian_behavior.experiments.e001_run import configure_logging, log_final, plots, run_attempt
from pedestrian_behavior.training import (fit, reload_checkpoint, save_checkpoint, seed_run, select_epoch, train_epoch)


class E001TrainingTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.evidence = []
        self.previous_determinism = torch.are_deterministic_algorithms_enabled()
        self.addCleanup(torch.use_deterministic_algorithms, self.previous_determinism)
        seed_run(0)

    def tearDown(self):
        if output := os.environ.get("E001_ACCEPTANCE_OUTPUT"):
            Path(output).mkdir(parents=True, exist_ok=True)
            (Path(output)/(self._testMethodName+".json")).write_text(json.dumps(self.evidence, indent=2)+"\n")

    def check(self, name, expected, actual):
        if isinstance(actual, (np.ndarray, torch.Tensor)):
            actual = actual.tolist()
        self.evidence.append({"check": name, "expected": expected, "actual": actual})
        self.assertEqual(expected, actual, name)

    def close(self, name, expected, actual):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
        self.evidence.append({"check": name, "expected": np.asarray(expected).tolist(), "actual": np.asarray(actual).tolist()})

    def sample(self, y, name="track"):
        labels = torch.tensor(y, dtype=torch.int64)
        return {"inputs": torch.ones(len(y), 3), "targets": labels, "gt_valid": labels != -100,
                "locator": name, "archive": name+".npz"}

    def track(self, y, probabilities, name="track"):
        return {"logits": np.log(np.asarray(probabilities, dtype=np.float32)), "targets": np.array(y),
                "gt_valid": np.array(y) != -100, "group": name, "locator": name, "archive": name+".npz"}

    def test_equal_track_hand_calculation_and_masking(self):
        tracks = [self.track([0], [[.5, .2, .2, .1]], "short"),
                  self.track([1]*9, [[.5, .25, .125, .125]]*9, "long")]
        score = prediction_metrics(tracks, 4)
        self.close("equal-track vs pooled macro-F1 and accuracy", [1/6, 1/22, .5, .1],
                   [score["track_weighted"]["macro_f1"], score["raw"]["macro_f1"],
                    score["track_weighted"]["accuracy"], score["raw"]["accuracy"]])
        self.close("one .5 target probability and nine .25: equal-track CE", (math.log(2)+math.log(4))/2, score["track_ce"])
        self.check("raw and equal-track confusion support", [[1., 9., 0., 0.], [1., 1., 0., 0.]],
                   [score["raw"]["support"], score["track_weighted"]["support"]])
        batch = collate_tracks([self.sample([0]), self.sample([1]*9)])
        logits = torch.full((2, 9, 4), float("nan"))
        logits[0, 0] = torch.from_numpy(tracks[0]["logits"][0])
        logits[1] = torch.from_numpy(tracks[1]["logits"])
        logits.requires_grad_()
        losses = track_cross_entropy(logits, batch["targets"], batch["gt_valid"], batch["padding_mask"])
        self.close("per-track CE, NaN padding excluded", [math.log(2), math.log(4)], losses.detach().numpy())
        losses.mean().backward()
        self.check("padding zero loss gradient", True, bool((logits.grad[0, 1:] == 0).all()))
        context = self.track([0, -100], [[.5, .2, .2, .1], [.1, .2, .3, .4]])
        self.check("unlabeled predictions retained, no metric support", [2, 1],
                   [prediction_metrics([context], 4)["denominators"][k] for k in ("context_frames", "gt_frames")])
        b = collate_tracks([self.sample([0, -100])])
        z = torch.from_numpy(context["logits"]).unsqueeze(0).requires_grad_()
        track_cross_entropy(z, b["targets"], b["gt_valid"], b["padding_mask"]).mean().backward()
        self.check("unlabeled zero loss gradient", [0.]*4, z.grad[0, 1])
        for invalid in ("empty", "nonfinite", "target"):
            bad = deepcopy(context)
            if invalid == "empty":
                bad["gt_valid"][:] = False
            elif invalid == "nonfinite":
                bad["logits"][1, 0] = np.nan
            else:
                bad["targets"][0] = 4
            with self.assertRaises(ValueError):
                prediction_metrics([bad], 4)
            with self.assertRaises(ValueError):
                track_cross_entropy(torch.from_numpy(bad["logits"])[None], torch.from_numpy(bad["targets"])[None],
                                    torch.from_numpy(bad["gt_valid"])[None], torch.zeros((1, 2), dtype=torch.bool))
        self.check("empty GT, nonfinite unlabeled logits and out-of-range GT fail", True, True)

    def test_unequal_final_batch_and_nonfinite_training(self):
        model = build_model("K", "A").eval()
        samples = [self.sample([0], "a"), self.sample([1]*3, "b"), self.sample([3], "c")]
        batches = [collate_tracks(samples[:2]), collate_tracks(samples[2:])]
        # Remove dropout and freeze updates so the epoch's hand-computed losses are comparable.
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0
        expected = []
        for b in batches:
            expected.extend(track_cross_entropy(model(b["inputs"], b["lengths"]), b["targets"], b["gt_valid"], b["padding_mask"]).detach().tolist())
        row = train_epoch(model, batches, torch.optim.AdamW(model.parameters(), lr=0, weight_decay=0), torch.device("cpu"), 1.)
        self.close("2+1 batch epoch CE uses all three tracks", float(np.mean(expected)), row["training_ce"])
        self.check("partial batch exposure and padding", [2, 3, 5, 5, 2, 7],
                   [row[k] for k in ("optimizer_steps", "processed_tracks", "context_slots", "gt_slots", "padding_slots", "batch_slots")])
        with torch.no_grad():
            next(model.parameters()).fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            train_epoch(model, batches, torch.optim.AdamW(model.parameters()), torch.device("cpu"), 1.)
        self.check("nonfinite training fails", True, True)

    def test_step_offsets_batch_losses_and_dashboard_axes(self):
        model = build_model("K", "A").eval()
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0
        batches = [collate_tracks([self.sample([0]), self.sample([1, 1, 1])]),
                   collate_tracks([self.sample([3])])]
        expected = [float(track_cross_entropy(model(b["inputs"], b["lengths"]), b["targets"],
                          b["gt_valid"], b["padding_mask"]).mean().detach()) for b in batches]
        steps = []
        result = fit(model, batches, lambda m: {"track_ce": 1., "track_weighted": {"macro_f1": .5}},
                     torch.optim.AdamW(model.parameters(), lr=0, weight_decay=0), torch.device("cpu"), self.root,
                     {}, lambda r, h: None, epochs=2, patience=5, clip_norm=1., on_step=lambda step, loss: steps.append([step, loss]))
        self.check("optimizer steps continue across epoch boundaries", [1, 2, 3, 4], [s[0] for s in steps])
        self.close("each step uses its actual batch's equal-track CE", expected*2, [s[1] for s in steps])
        self.close("epoch CE is track-weighted across the 2+1 batches", [(expected[0]*2+expected[1])/3]*2,
                   [r["training_ce"] for r in result["history"]])
        definitions = {}
        configure_logging(SimpleNamespace(define_metric=lambda name, **kwargs: definitions.update({name: kwargs})))
        self.check("native curves use optimizer step or epoch", ["optimizer_step", "epoch", "epoch"],
                   [definitions[k]["step_metric"] for k in ("train/loss_step", "train/loss_epoch", "val/*")])
        self.check("counter axes and detail panels hidden from auto plots", [True]*3,
                   [definitions[k]["hidden"] for k in ("optimizer_step", "epoch", "details/*")])

    def test_selection_patience_budget_and_checkpoint_reload(self):
        sequence = [(.2, 2.), (.3, 2.), (.3, 1.), (.3, 1.), (.25, .5), (.3, 1.1), (.3, 1.)]
        best, stale, observed = None, 0, []
        for epoch, (f1, ce) in enumerate(sequence, 1):
            best, stale, selected = select_epoch(best, f1, ce, epoch, stale)
            observed.append([best["epoch"], stale, selected])
        self.check("strict F1 resets; lower CE selects; earlier exact tie; five stale epochs",
                   [[1, 0, True], [2, 0, True], [3, 1, True], [3, 2, False], [3, 3, False], [3, 4, False], [3, 5, False]], observed)
        batch = collate_tracks([self.sample([0])])
        model = build_model("K", "A")
        snapshots, rows = {}, []
        metrics = iter(sequence)
        def validate(m):
            f1, ce = next(metrics)
            snapshots[len(snapshots)+1] = m.eval()(batch["inputs"], batch["lengths"]).detach().clone()
            return {"track_ce": ce, "track_weighted": {"macro_f1": f1}}
        result = fit(model, [batch], validate, torch.optim.AdamW(model.parameters(), lr=.001), torch.device("cpu"),
                     self.root, {"normalization": {"mean": [0.], "scale": [1.]}}, lambda row, history: rows.append(row),
                     epochs=30, patience=5, clip_norm=1.)
        self.check("patience stops after epoch seven and selects three", [7, 3, "patience"],
                   [len(rows), result["selection"]["epoch"], result["stopping_reason"]])
        checkpoint = reload_checkpoint(self.root/"best.pt", model, "cpu")
        self.check("best checkpoint carries normalization and epoch", [3, {"mean": [0.], "scale": [1.]}],
                   [checkpoint["epoch"], checkpoint["normalization"]])
        self.close("reloaded selected logits", snapshots[3].numpy(), model.eval()(batch["inputs"], batch["lengths"]).detach().numpy())
        last = torch.load(self.root/"last.pt", weights_only=True)
        self.check("last completed epoch and reason", [7, "patience"], [last["epoch"], last["stopping_reason"]])
        result = fit(model, [batch], lambda m: {"track_ce": 1., "track_weighted": {"macro_f1": .8}},
                     torch.optim.AdamW(model.parameters()), torch.device("cpu"), self.root, {}, lambda r, h: None,
                     epochs=1, patience=5, clip_norm=1.)
        self.check("improving final epoch is budget-limited", "budget", result["stopping_reason"])
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            select_epoch(None, float("nan"), 1., 1, 0)

    def conditions(self, position, labels=None, boxes=None, dataset="loki", extent=None):
        n = len(position)
        y = np.array(labels if labels is not None else [0]*n)
        metadata = {"native_extent_us": extent or [0, (n-1)*200000],
                    "availability": boxes or [{} for _ in range(n)]}
        arrays = {"ped_position_valid": np.array(position, dtype=bool), "ped_velocity_valid": np.array(position, dtype=bool),
                  "ego_pose_valid": np.ones(n, dtype=bool), "ped_position": np.zeros((n, 2)), "ego_position": np.zeros((n, 2))}
        return metadata, arrays, y, y != -100, dataset

    def test_all_stratum_boundaries_native_duration_coverage_counts(self):
        for seconds, expected in ((0, 0), (.999999, 0), (1, 1), (4.999999, 1), (5, 2), (10, 3), (15, 4)):
            c, _ = observation_conditions(*self.conditions([True], extent=[0, round(seconds*1e6)]))
            self.check(f"native duration {seconds}s, singleton grid", expected, int(c["duration"][0]))
        for valid, total, expected in ((0, 4, 0), (1, 4, 1), (2, 4, 1), (3, 4, 2), (4, 4, 2)):
            c, _ = observation_conditions(*self.conditions([True]*valid+[False]*(total-valid)))
            self.check(f"coverage {valid}/{total}", expected, int(c["position-coverage"][0]))
        for count, expected in ((1, 0), (2, 1), (4, 1), (5, 2), (19, 2), (20, 3)):
            c, _ = observation_conditions(*self.conditions([True]*count))
            self.check(f"GT count {count}", expected, int(c["gt-count"][0]))
        for gap, expected in ((0, 0), (1, 1), (5, 1), (6, 2)):
            c, _ = observation_conditions(*self.conditions([False]*8+[True]+[False]*gap+[True]+[False]*9))
            self.check(f"internal gap {gap} slots; ignore leading/trailing", expected, int(c["internal-position-gap"][0]))

    def test_boxes_range_transition_gaps_and_singletons(self):
        boxes = [{"road_2d": True}, {"waymo_2d": True}, {"loki_2d": True}, {}]
        c, flags = observation_conditions(*self.conditions([True]*4, boxes=boxes, dataset="road-waymo"))
        self.check("ROAD OR Waymo 2D, no LOKI alias", [1, 1, 0, 0], c["frame-2d"])
        self.check("individual flags retained", [False, True, False, False], flags["waymo_2d"])
        c, _ = observation_conditions(*self.conditions([True]*4, boxes=boxes))
        self.check("LOKI 2D only", [0, 0, 1, 0], c["frame-2d"])
        for boxes, expected in (([{}]*2, 0), ([{"loki_2d": True}, {}], 1), ([{"loki_2d": True}]*2, 2)):
            c, _ = observation_conditions(*self.conditions([True]*2, boxes=boxes))
            self.check("track annotated 2D "+str(expected), expected, int(c["track-2d"][0]))
        args = self.conditions([True]*8)
        args[1]["ped_position"][:, 0] = [0, 9.999, 10, 19.999, 20, 39.999, 40, 1]
        args[1]["ego_pose_valid"][-1] = False
        c, _ = observation_conditions(*args)
        self.check("physical range exact boundaries and unknown ego", [0, 0, 1, 1, 2, 2, 3, 4], c["range"])
        args[1]["ped_position_valid"][0] = False
        args[1]["ped_velocity_valid"][1] = False
        c, _ = observation_conditions(*args)
        self.check("independent frame position and velocity flags", [[0, 1], [1, 0]], [c["position"][:2].tolist(), c["velocity"][:2].tolist()])
        self.check("unknown pedestrian range", 4, int(c["range"][0]))
        labels = [0, 0, 0, 1, 1, 1, -100, 2, -100, 3, 3]
        c, _ = observation_conditions(*self.conditions([True]*len(labels), labels=labels))
        self.check("midpoint proximity, GT gaps, singleton, constant labeled run", [1, 0, 0, 0, 0, 1, 2, 2, 2, 1, 1], c["transition"])

    def test_slice_reweighting_supported_classes_empty_and_saved_predictions(self):
        tracks = [self.track([0, 0], [[.5, .2, .2, .1]]*2, "short"),
                  self.track([1]*9, [[.5, .25, .125, .125]]*9, "long")]
        for t in tracks:
            n = len(t["targets"])
            t["conditions"] = {"position": np.array([1]+[0]*(n-1))}
            t["source_flags"] = {"road_2d": np.ones(n, dtype=bool)}
            t["source_times_us"] = list(range(n))
        axes = {"position": ("missing", "valid", "empty")}
        slices = stratified_metrics(tracks, 4, axes)["position"]
        self.close("one accepted frame per track in slice; supported two-class mean", 1/3, slices["valid"]["track_weighted"]["macro_f1"])
        self.close("slice track weights recomputed", [1., 1., 0., 0.], slices["valid"]["track_weighted"]["support"])
        self.check("unsupported class F1 unavailable", [None, None], slices["valid"]["track_weighted"]["f1"][2:])
        self.check("empty slice no score, classes or denominators", [None, [], 0],
                   [slices["empty"]["track_weighted"]["macro_f1"], slices["empty"]["track_weighted"]["supported_classes"], slices["empty"]["denominators"]["gt_frames"]])
        save_predictions(self.root/"predictions.npz", tracks)
        loaded = load_predictions(self.root/"predictions.npz")
        self.check("archive reconstructs primary and strata exactly", [prediction_metrics(tracks, 4), slices],
                   [prediction_metrics(loaded, 4), stratified_metrics(loaded, 4, axes)["position"]])
        with np.load(self.root/"predictions.npz", allow_pickle=False) as saved:
            self.check("unpadded float32 logits and offsets", ["float32", [0, 2, 11]], [str(saved["logits"].dtype), saved["offsets"].tolist()])
            self.check("no pickle objects", True, all(saved[k].dtype.kind != "O" for k in saved.files))
        result = {"primary": prediction_metrics(loaded, 4), "strata": {"position": slices}}
        history = [{"epoch": 1, "training_ce": 1., "validation_ce": 1.1, "validation_f1": .2}]
        paths = plots(self.root/"plots", history, {"epoch": 1}, {"fixture": result})
        self.check("curves, four confusion views, per-class and slice PNG/SVG", [8, 16], [len(paths), len(list((self.root/"plots").iterdir()))])
        payloads, configs = [], []
        config = SimpleNamespace(update=lambda value, **kwargs: configs.append(value))
        log_final(SimpleNamespace(log=payloads.append, config=config), {"fixture": result}, paths, {"revision": "fixture"})
        self.check("W&B allowed scalar/table/plot content only", True,
                   all(isinstance(v, (dict, int, float, type(None), wandb.Table, wandb.Image)) for v in payloads[0].values()))
        self.check("provenance/support stay in config, no numeric metadata charts", [False, True, True],
                   ["provenance" in payloads[0], "provenance" in configs[0], "evaluation_support" in configs[0]])
        self.check("all final keys belong to evaluation, strata or details", True,
                   all(k.startswith(("eval/", "strata/", "details/")) for k in payloads[0]))
        self.check("W&B has no checkpoint or prediction artifacts", False, any(".pt" in k or "predictions" in k for k in payloads[0]))
        with self.assertRaisesRegex(RuntimeError, "logging failure"):
            log_final(SimpleNamespace(log=lambda _: (_ for _ in ()).throw(RuntimeError("logging failure")), config=config), {}, [], {})
        self.check("logging failure propagates; local predictions remain", True, (self.root/"predictions.npz").exists())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA acceptance runs on aalto")
    def test_cuda_deterministic_training_loss_and_checkpoint(self):
        for variant in ("A", "B"):
            outcomes = []
            for _ in range(2):
                seed_run(0)
                model = build_model("K", variant).cuda()
                batches = [collate_tracks([self.sample([0, -100, 1]), self.sample([3])])]
                row = train_epoch(model, batches, torch.optim.AdamW(model.parameters(), lr=.001), torch.device("cuda"), 1.)
                z = model.eval()(batches[0]["inputs"].cuda(), batches[0]["lengths"]).detach().cpu()
                outcomes.append((row["training_ce"], z))
            self.check(variant+" repeated seed deterministic CUDA CE/logits", True,
                       outcomes[0][0] == outcomes[1][0] and torch.equal(outcomes[0][1], outcomes[1][1]))
            save_checkpoint(self.root/"cuda.pt", model, {"epoch": 1})
            reloaded = build_model("K", variant).cuda().eval()
            reload_checkpoint(self.root/"cuda.pt", reloaded, "cuda")
            self.check(variant+" CUDA checkpoint prediction equality", True,
                       torch.equal(outcomes[1][1], reloaded(batches[0]["inputs"].cuda(), batches[0]["lengths"]).detach().cpu()))
        self.check("deterministic CUDA accepted masked loss and clipped AdamW", True, True)

    def test_attempt_wandb_failures_preserve_local_evidence_and_fresh_directory(self):
        from pedestrian_behavior.data.features import features, fit_normalization
        from pedestrian_behavior.data.preparation import checksum, load_track, prepare_collection
        from test_track_preparation import native_readers, pose
        people = {name: {0: [10., 5., 1.], 2: [12., 5., 1.]} for name in ("a", "b", "c")}
        readers = native_readers(self.root, [0, 200000, 400000], [pose()]*3, people)
        base = self.root/"artifacts"
        for reader in readers:
            d = reader.dataset
            collection = base/"reader-preparation"/d
            manifest, _ = prepare_collection(reader, collection)
            entries, physical = [], []
            for archive, split in zip(manifest["tracks"], ("training", "validation", "test")):
                m, a = load_track(collection/"tracks"/archive)
                physical.append(features(a, "K"))
                identity = json.loads(m["track_locator"])
                entries.append({"archive": archive, "locator": m["track_locator"], "group": identity.get("clip", identity.get("scenario")), "split": split, "exclusions": []})
            # A tiny synthetic split tests run wiring; group-split policy is tested separately.
            record = {"schema_version": 1, "experiment": "E001", "status": "complete", "dataset": d,
                      "tracks": entries, "collection": {"manifest_sha256": checksum(collection/"manifest.json")},
                      "normalization": {"K": fit_normalization(physical[:1])}, "policy": {"fixture": True},
                      "native_audit": {"verified_overrides": [], "policy": {"fixture": True}},
                      "population": {"fixture": True}, "support": {"fixture": True}}
            destination = base/"data-setup"/d
            destination.mkdir(parents=True)
            (destination/"setup.json").write_text(json.dumps(record))
        for stage in ("init", "step", "epoch", "final", "complete"):
            output = self.root/stage
            attempt_seed = 3 if stage == "complete" else 0
            events = []
            class FakeRun:
                id, url = "test", "test"
                config = SimpleNamespace(update=lambda value, **kwargs: None)
                def define_metric(self, *args, **kwargs):
                    pass
                def log(self, value, **kwargs):
                    events.append(value)
                    if ((stage == "step" and "train/loss_step" in value) or (stage == "epoch" and "epoch" in value)
                            or (stage == "final" and any(k.startswith("eval/") for k in value))):
                        raise RuntimeError(stage+" failure")
                def finish(self, **kwargs):
                    pass
            def initialize(**kwargs):
                self.assertEqual((kwargs["entity"], kwargs["project"]), ("yammo-unipd", "pedestrian-behaviour-labeling"))
                self.assertRegex(kwargs["name"], rf"^E001-loki-K-MLP-seed{attempt_seed}-\d{{8}}-\d{{6}}$")
                self.assertEqual(kwargs["config"]["settings"]["seed"], attempt_seed)
                if stage == "init":
                    raise RuntimeError("init failure")
                return FakeRun()
            with patch("pedestrian_behavior.experiments.e001_run.E001_ROOT", base), \
                 patch("pedestrian_behavior.experiments.e001_run.wandb.init", side_effect=initialize), \
                 patch.dict("pedestrian_behavior.experiments.e001_run.SETTINGS", {"epochs": 1}):
                if stage == "complete":
                    result = run_attempt("loki", "K", "A", output, "cpu", seed=attempt_seed, epochs=1, patience=8)
                    self.check("successful attempt selects only source validation", [1, "budget"], [result["selection"]["epoch"], result["stopping_reason"]])
                    config = json.loads((output/"config.json").read_text())
                    provenance = json.loads((output/"provenance.json").read_text())
                    self.assertEqual((config["settings"]["seed"], config["settings"]["epochs"], config["settings"]["patience"], provenance["seed"]), (3, 1, 8, 3))
                else:
                    with self.assertRaisesRegex(RuntimeError, stage+" failure"):
                        run_attempt("loki", "K", "A", output, "cpu")
            self.check(stage+" attempt persists status/provenance", ["complete" if stage == "complete" else "failed", True],
                       [json.loads((output/"status.json").read_text())["status"], (output/"provenance.json").exists()])
            if stage in ("epoch", "final", "complete"):
                self.check(stage+" failure retains completed epoch and checkpoints", [True, True, True],
                           [(output/f).exists() for f in ("history.json", "best.pt", "last.pt")])
            if stage != "init":
                self.check(stage+": step evidence exists before external logging", [1, 1],
                           [len((output/"steps.jsonl").read_text().splitlines()),
                            json.loads((output/"steps.jsonl").read_text().splitlines()[0])["optimizer_step"]])
            if stage in ("final", "complete"):
                self.check(stage+": both final evaluations survive W&B failure", [True, True],
                           [(output/(d+"-test")/"predictions.npz").exists() for d in ("loki", "road-waymo")])
                self.check("exactly one epoch and one step record", [1, 1],
                           [sum("epoch" in v for v in events), sum("train/loss_step" in v for v in events)])
                epoch = next(v for v in events if "epoch" in v)
                self.check("epoch history exposes only the agreed curves and axis", ["epoch", "train/loss_epoch", "val/loss", "val/macro_f1"], sorted(epoch))
            with self.assertRaises(FileExistsError):
                run_attempt("loki", "K", "A", output, "cpu")


if __name__ == "__main__":
    unittest.main()
