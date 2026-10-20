"""Train only a new nine-class linear head over frozen nuScenes D-DITR features."""

import argparse
from datetime import datetime, timezone
import hashlib
import html
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from pedestrian_behavior.data.preparation import checksum
from pedestrian_behavior.datasets import road_waymo as road
from pedestrian_behavior.inspection.native_features import native_scene
from pedestrian_behavior.inspection.pretrained import PointEncoder, DITR_SHA256, DITR_REVISION
from pedestrian_behavior.inspection.road_waymo import aligned_frames
from pedestrian_behavior.inspection.segmentation import semantic_bev, semantic_confusion, semantic_metrics
from pedestrian_behavior.inspection.semantic_probe import fit_linear
from .e002a_point_check import context
from .e002a_probe import write_json
from .e002a_semantic_check import (GT_MAP, PRED_MAP, NAMES, NUSCENES_NAMES, PALETTE,
                                 aligned_semantics, mapped_labels, plot_confusion)

SETTINGS = {"seed":0, "frames_per_clip":6, "train_points_per_frame":8192, "batch_size":8192,
            "epochs":50, "patience":10, "learning_rate":.001, "weight_decay":.0001,
            "loss":"unweighted point CE", "selection":"validation supported-class macro-IoU, lower CE, earlier epoch",
            "normalization":"sampled training features only; per-channel mean/std; std floor 1e-6"}


def frame_selection(samples, timestamps):
    selected = []
    for group, sample in sorted(samples.items()):
        if not sample["random"]:
            continue
        if sample["split"] not in ("training","validation"):
            raise ValueError("Held-out clip in linear probe")
        keys = sorted(set(timestamps.get(group, [])))
        if len(keys) < SETTINGS["frames_per_clip"]:
            raise ValueError(f"Insufficient annotated frames in {group}")
        indices = np.linspace(0,len(keys)-1,SETTINGS["frames_per_clip"],dtype=int)
        selected.extend({"group":group,"key":keys[i],"split":sample["split"]} for i in indices)
    groups = {s: {f["group"] for f in selected if f["split"] == s} for s in ("training","validation")}
    if not groups["training"] or not groups["validation"] or groups["training"] & groups["validation"]:
        raise ValueError("Separate nonempty training and validation clips required")
    return selected


def restricted_predictions(logits):
    """Taxonomy-only control: sum native probabilities within each common class."""
    if logits.ndim != 2 or logits.shape[1] != 16 or not np.isfinite(logits).all():
        raise ValueError("Expected finite native 16-class logits")
    probability = np.exp(logits-logits.max(1,keepdims=True))
    return np.stack([probability[:,PRED_MAP==i].sum(1) for i in range(9)],axis=1).argmax(1)


def fingerprint(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def report(output, record):
    lines = ["# E002a frozen-feature linear semantic probe", "", f"Status: **{record['status']}**.", "",
        "Fresh Linear(64,9), frozen/eval D-DITR. Six evenly spaced semantic timestamps in each of the "
        "same eight training/two validation development clips. Train on up to8192 uniformly sampled supported "
        "TOP points/frame; validate on every supported TOP return. Full scene input, same 5 cm preprocessing. "
        "No behavior training, LOKI labels, held-out access or W&B. This is Waymo-supervised head adaptation, not zero-shot segmentation.", "",
        "The nine-class taxonomy and exclusions match the prior diagnostic. The original unrestricted head and "
        "a taxonomy-only control (sum native probabilities per shared class; exclude unmatched native classes) "
        "are evaluated on the exact same validation points. This control separates taxonomy restriction from learning. "
        "Excluded GT never contributes false positives. These development clips have already been inspected; "
        "scores are exploratory, not final generalization estimates.", "",
        "[Settings, selection, history and provenance](linear-probe.json) · [Visual gallery](index.html)", ""]
    if "results" in record:
        results = record["results"]
        lines += [f"Selected epoch: **{record['training']['selection']['epoch']}**, stopping: {record['training']['stopping_reason']}.", "",
            "| Class | Validation GT points | Original IoU % | Taxonomy-only IoU % | Learned IoU % | Learned precision % | Learned recall % |", "|---|---:|---:|---:|---:|---:|---:|"]
        fmt = lambda v: "—" if v is None else f"{100*v:.2f}"
        for i,name in enumerate(NAMES):
            a,b,c = (results[k]["classes"][i] for k in ("original","restricted","learned"))
            lines.append(f"| {name} | {c['support']:,} | {fmt(a['iou'])} | {fmt(b['iou'])} | {fmt(c['iou'])} | {fmt(c['precision'])} | {fmt(c['recall'])} |")
        lines += ["", "| Metric | Original | Taxonomy-only | Learned |", "|---|---:|---:|---:|"]
        for key in ("supported_macro_iou","accuracy"):
            lines.append(f"| {key} | "+" | ".join(fmt(results[k][key]) for k in ("original","restricted","learned"))+" |")
        lines += ["", "Class supports and confusion matrices are pooled native returns; repeated returns/frames are correlated. "
            "Macro-IoU averages GT-supported classes only. Training/validation class support and per-clip/frame scores "
            "are retained in the numerical record. No uncertainty intervals or benchmark claim.", "",
            "[Learned confusion PNG](learned-confusion/confusion.png) / [SVG](learned-confusion/confusion.svg) · "
            "[Training curves](history.png) / [SVG](history.svg). Pickle-free frame caches retain features, coordinates, "
            "GT, native point/pixel identities and both original predictions. Separate validation prediction archives "
            "and the head-only best.pt/last.pt support reconstruction without backbone inference."]
    lines += ["", "A successful probe supports accessibility of common semantics from these frozen features with Waymo "
        "supervision. It does not prove behavior usefulness or LOKI transfer. A failed probe would not rule out "
        "nonlinear readouts, fine-tuning or other features. No model or preprocessing search is performed."]
    (output/"report.md").write_text("\n".join(lines)+"\n")
    body = ["<!doctype html><html lang='en'><meta charset='utf-8'><title>Frozen-feature linear probe</title>",
        "<style>body{font:16px system-ui;background:#171b22;color:white;margin:24px}img{width:100%;max-width:1980px}a{color:#8cf}</style>",
        f"<h1>Frozen D-DITR, fresh linear head</h1><p>Status: {html.escape(record['status'])}. "
        "Waymo-supervised development diagnostic, not zero-shot or behavior evaluation.</p>",
        "<p><a href='report.md'>Results and protocol</a> · <a href='linear-probe.json'>Numerical evidence</a></p>"]
    if (output/"history.png").exists():
        body.append("<img src='history.png' alt='Training and validation history'>")
    for item in record.get("visuals",[]):
        body.append(f"<h2>{html.escape(item)}</h2><img src='panels/{item}.png' alt='Ground truth, prediction and errors'>")
    (output/"index.html").write_text("\n".join(body)+"</html>\n")


def run(args):
    import pyarrow.parquet as pq
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(0)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    probe, output = args.probe.resolve(), args.output.resolve()
    root = probe.parents[2]
    if not output.is_relative_to(root/"experiments/E002a"):
        raise ValueError("Output must be under outputs/experiments/E002a")
    source = json.loads((probe/"probe.json").read_text())
    if source["status"] != "complete":
        raise ValueError("Completed native probe required")
    output.mkdir(parents=True,exist_ok=False)
    for name in ("features","predictions","panels","learned-confusion"):
        (output/name).mkdir()
    config = source["config"]
    protected = source["protected"].copy()
    for path in (probe/"probe.json",Path(config["ditr_config"]),Path(config["ditr_weights"])):
        protected[str(path)] = checksum(path)
    record = {"status":"incomplete","started_utc":datetime.now(timezone.utc).isoformat(),"settings":SETTINGS,
        "revision":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "git_diff":subprocess.check_output(["git","diff","HEAD"],text=True),"command":sys.orig_argv,
        "dependencies":sorted(f"{d.metadata['Name']}=={d.version}" for d in distributions()),
        "torch":torch.__version__,"cuda":torch.version.cuda,"device":args.device,"names":NAMES,
        "gt_map":GT_MAP.tolist(),"prediction_map":PRED_MAP.tolist(),"checkpoint_sha256":DITR_SHA256,
        "author_revision":DITR_REVISION,"frames":[],"visuals":[],"failures":[]}
    started = time.monotonic()
    write_json(output/"linear-probe.json",record)
    try:
        for path,digest in protected.items():
            if checksum(path)!=digest:
                raise ValueError(f"Protected source changed: {path}")
        _,manifest,setup,reader = context("road-waymo",root,config,probe,protected)
        timestamps, paths = {}, {}
        samples = source["samples"]["road-waymo"]
        for group,sample in samples.items():
            if not sample["random"]:
                continue
            if setup["groups"][group]!=sample["split"]:
                raise ValueError("Frozen split mismatch")
            scene = reader.scenes[group]
            lidar_path = road.component_path(scene,"lidar",reader.waymo_root)
            paths[group] = lidar_path.parent.parent/"lidar_segmentation"/lidar_path.name
            protected[str(paths[group])] = checksum(paths[group])
            timestamps[group] = pq.read_table(paths[group],columns=["key.frame_timestamp_micros"])
            timestamps[group] = timestamps[group].column(0).to_pylist()
        selection = frame_selection(samples,timestamps)
        if len(selection)!=60 or sum(f["split"]=="training" for f in selection)!=48:
            raise ValueError("Expected 48 training/12 validation frames from existing ten clips")
        record["selection"] = selection
        write_json(output/"linear-probe.json",record)
        encoder = PointEncoder(config["ditr_code"],config["ditr_config"],config["ditr_weights"],args.device)
        if tuple(encoder.names)!=NUSCENES_NAMES or encoder.model.training or any(p.requires_grad for p in encoder.model.parameters()):
            raise ValueError("Backbone must be the frozen/eval released model")
        before = fingerprint(encoder.model)
        record["hardware"] = torch.cuda.get_device_name() if args.device=="cuda" else "CPU"
        rng = np.random.default_rng(0)
        for group in sorted(timestamps):
            keys = [f["key"] for f in selection if f["group"]==group]
            scene, frozen = reader.scenes[group], manifest["scenes"][group]
            reader.index_scene(group)
            scene_name = scene["segment_context_name"]
            scans = iter(aligned_frames(road.frame_groups(road.component_rows(road.component_path(scene,"lidar",reader.waymo_root),scene_name)),keys))
            labels = iter(aligned_frames(road.frame_groups(road.component_rows(paths[group],scene_name)),keys))
            identities = [json.loads(e["locator"]) for e in setup["tracks"] if e["group"]==group and not e["exclusions"]]
            tracks = [i["tube_uid"] for i in identities]
            for frame in native_scene(reader,group,keys,tracks,frozen):
                gt,parts = aligned_semantics(next(scans),next(labels))
                points = frame["points"]
                if points is None or len(gt)!=len(points):
                    raise ValueError("Native semantic/point mismatch")
                target,_ = mapped_labels(gt,np.zeros(len(gt),dtype=np.int16))
                indices = np.flatnonzero(target>=0)
                full_support = np.bincount(target[indices],minlength=9).tolist()
                if setup["groups"][group]=="training" and len(indices)>SETTINGS["train_points_per_frame"]:
                    indices = np.sort(rng.choice(indices,SETTINGS["train_points_per_frame"],replace=False))
                if not len(indices):
                    raise ValueError("Selected semantic frame has no supported GT")
                features,logits,_,inverse = encoder(points)
                native_pred = logits.argmax(1)[inverse[indices]]
                original = PRED_MAP[native_pred]
                restricted = restricted_predictions(logits[inverse[indices]])
                return_id = np.zeros(len(points),dtype=np.int8)
                pixel = np.full(len(points),-1,dtype=np.int32)
                for part in parts:
                    if part["laser"]==1:
                        sl = slice(part["start"],part["stop"])
                        return_id[sl],pixel[sl] = part["return"],part["valid_flat_pixels"]
                if (return_id[indices]==0).any():
                    raise ValueError("Scored points must be native TOP returns")
                name = f"{group}-{frame['key']}"
                np.savez_compressed(output/"features"/(name+".npz"),features=features[inverse[indices]],targets=target[indices],
                    xyz=points[indices,:3],native_indices=indices,gt_waymo=gt[indices],return_id=return_id[indices],flat_pixel=pixel[indices],
                    original_prediction=original,restricted_prediction=restricted)
                item = {"name":name,"group":group,"key":frame["key"],"split":setup["groups"][group],
                    "segment":scene_name,"segmentation_path":str(paths[group]),"points":len(indices),
                    "full_supported_points":int(sum(full_support)),"full_class_support":full_support,
                    "cached_class_support":np.bincount(target[indices],minlength=9).tolist(),
                    "sha256":checksum(output/"features"/(name+".npz"))}
                record["frames"].append(item)
            write_json(output/"linear-probe.json",record)
            print("Extracted",group,setup["groups"][group],flush=True)
        after = fingerprint(encoder.model)
        if before!=after or any(p.grad is not None for p in encoder.model.parameters()):
            raise ValueError("Frozen backbone changed during extraction")
        record["backbone_state_sha256_before"] = before
        record["backbone_state_sha256_after"] = after
        del encoder
        if args.device=="cuda":
            torch.cuda.empty_cache()
        arrays = {}
        for split in ("training","validation"):
            xs,ys = [],[]
            for item in record["frames"]:
                if item["split"]==split:
                    with np.load(output/"features"/(item["name"]+".npz"),allow_pickle=False) as a:
                        xs.append(a["features"]);ys.append(a["targets"])
            arrays[split] = np.concatenate(xs),np.concatenate(ys)
        def on_epoch(row,history):
            record["history"] = history
            write_json(output/"linear-probe.json",record)
            if row["epoch"]%5==0 or row["stopping_reason"]:
                print("Epoch",row["epoch"],"validation macro-IoU",round(row["validation_macro_iou"],4),flush=True)
        head,stats,training,predictions = fit_linear(*arrays["training"],*arrays["validation"],9,args.device,output,SETTINGS,on_epoch)
        record["training"] = training
        record["head_parameters"] = sum(p.numel() for p in head.parameters())
        cm = {name:np.zeros((9,10),dtype=np.int64) for name in ("original","restricted","learned")}
        offset = 0
        for item in record["frames"]:
            if item["split"]!="validation":
                continue
            with np.load(output/"features"/(item["name"]+".npz"),allow_pickle=False) as a:
                pred = predictions[offset:offset+len(a["targets"])];offset+=len(pred)
                np.savez_compressed(output/"predictions"/(item["name"]+".npz"),targets=a["targets"],prediction=pred,native_indices=a["native_indices"])
                item["results"] = {}
                for name,value in (("original",a["original_prediction"]),("restricted",a["restricted_prediction"]),("learned",pred)):
                    matrix = semantic_confusion(a["targets"],value,9);cm[name]+=matrix
                    item["results"][name] = semantic_metrics(matrix)
                for name,value,title in (("original",a["original_prediction"],"Original nuScenes head"),
                                         ("restricted",a["restricted_prediction"],"Taxonomy-only restriction"),("learned",pred,"Fresh linear head, frozen features")):
                    label = item["name"]+"-"+name
                    semantic_bev(a["xyz"],a["targets"],value,NAMES,PALETTE,label,prediction_title=title).save(output/"panels"/(label+".png"))
                    record["visuals"].append(label)
        record["results"] = {name:semantic_metrics(matrix) for name,matrix in cm.items()}
        plot_confusion(output/"learned-confusion",cm["learned"],title="Frozen D-DITR + learned linear head: 12 Waymo validation frames")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        history = training["history"]
        fig,axes = plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
        for key in ("training_ce","validation_ce"):
            axes[0].plot([r["epoch"] for r in history],[r[key] for r in history],label=key)
        axes[0].legend();axes[0].set(xlabel="Epoch",ylabel="Point CE")
        axes[1].plot([r["epoch"] for r in history],[r["validation_macro_iou"] for r in history])
        axes[1].axvline(training["selection"]["epoch"],color="gray",linestyle="--")
        axes[1].set(xlabel="Epoch",ylabel="Validation supported-class macro-IoU")
        for ext in ("png","svg"):
            fig.savefig(output/("history."+ext),dpi=160)
        plt.close(fig)
        record["status"] = "complete"
    except BaseException:
        record["status"] = "failed";record["failures"].append(traceback.format_exc());raise
    finally:
        changed = [p for p,d in protected.items() if not Path(p).is_file() or checksum(p)!=d]
        record.update(protected=protected,changed_files=changed,protected_files=len(protected),seconds=time.monotonic()-started,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        if changed:
            record["status"] = "failed"
        write_json(output/"linear-probe.json",record);report(output,record)
        if changed:
            raise ValueError("Protected sources changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",choices=("cpu","cuda"),required=True)
    run(parser.parse_args())
