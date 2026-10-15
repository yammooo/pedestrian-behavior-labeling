"""Collect frozen E001-family evidence, audit E001c, and export seed comparisons."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from pedestrian_behavior.data.boxes import COLUMNS, FLAGS, INPUT_COLUMNS
from pedestrian_behavior.data.preparation import checksum
from pedestrian_behavior.evaluation import confusion_metrics, load_predictions, prediction_metrics, stratified_metrics
from pedestrian_behavior.experiments.e001b_run import STRATA
from pedestrian_behavior.training import select_epoch

SWEEPS = {"E001": "seeds-20261009-084417-412679", "E001b": "seeds-20261009-140118-403993",
          "E001c": "seeds-20261015-114954-843530"}
DIRS = [("road-waymo", "road-waymo"), ("road-waymo", "loki"), ("loki", "loki"), ("loki", "road-waymo")]
NAMES = ["R→R", "R→L", "L→L", "L→R"]
MODELS = [("E001", f, a) for a in ("A", "B") for f in ("RAW", "K", "K+T", "K+T+R")]
MODELS += [("E001b", f, "B") for f in ("availability", "geometry")]
NEW = [("E001c", f, "B") for f in INPUT_COLUMNS]
CLASSES = ["MOVING", "STOPPED", "WAITING_TO_CROSS", "CROSSING"]
MODELS += NEW
PLOT_MODELS = [("E001", "RAW", "B"), ("E001", "K+T+R", "B"), ("E001b", "availability", "B"), ("E001b", "geometry", "B")]+NEW
PAIRS = [(PLOT_MODELS[i], NEW[0]) for i in (0, 1, 3)]+[(NEW[0], NEW[1]), (NEW[1], NEW[2]), (NEW[0], NEW[2])]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def moments(values, scale=100):
    if all(v is None for v in values):
        return {"mean": None, "std": None, "seeds": values}
    assert len(values) == 5 and all(v is not None for v in values)
    values = np.asarray(values, dtype=np.float64)*scale
    assert np.isfinite(values).all()
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1)), "seeds": values.tolist()}


def paired(before, after):
    if all(v is None for v in list(before)+list(after)):
        return moments([None]*5, 1) | {"positive_seeds": None}
    differences = np.asarray(after)-np.asarray(before)
    return moments(differences.tolist()) | {"positive_seeds": int((differences > 0).sum())}


def fmt(values, scale=100):
    result = moments(list(values), scale)
    return "—" if result["mean"] is None else f"{result['mean']:.2f} ± {result['std']:.2f}"


def title(model):
    return ("MLP" if model[2] == "A" else "BiLSTM")+" "+model[1]


def table(headers, rows):
    return "\n".join(["| "+" | ".join(headers)+" |", "| "+" | ".join(["---"]*len(headers))+" |"]+
                      ["| "+" | ".join(map(str, row))+" |" for row in rows])+"\n"


def csv_file(root, name, rows):
    with (root/name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def signature(tracks):
    digest = hashlib.sha256()
    for track in tracks:
        digest.update(track["locator"].encode())
        for key in ("targets", "gt_valid"):
            digest.update(track[key].tobytes())
        digest.update(json.dumps([track["grid_times_us"], track["source_times_us"]]).encode())
        for axis in STRATA:
            digest.update(track["conditions"][axis].tobytes())
        for flag in ("loki_2d", "road_2d", "waymo_2d"):
            digest.update(track["source_flags"][flag].tobytes())
    return digest.hexdigest()


def collect(experiments, output, prior):
    output.mkdir(parents=True, exist_ok=False)
    previous = read(prior)
    inventory, hashes = [], {}
    for experiment, name in SWEEPS.items():
        sweep_path = experiments/experiment/"runs"/name
        sweep = read(sweep_path/"sweep.json")
        assert sweep["status"] == "complete" and not sweep["failures"]
        assert sweep["seeds"] == list(range(5)) and sweep["epochs"] == 75 and sweep["patience"] == 8
        assert len(sweep["attempts"]) == {"E001": 80, "E001b": 20, "E001c": 30}[experiment]
        destination = output/"metadata"/experiment/name
        destination.mkdir(parents=True)
        shutil.copyfile(sweep_path/"sweep.json", destination/"sweep.json")
        for attempt in sweep["attempts"]:
            assert attempt["status"] == "complete" and attempt["returncode"] == 0
            path = sweep_path/attempt["name"]
            local = destination/attempt["name"]; local.mkdir()
            for file in [path/n for n in ("config.json", "status.json", "history.json", "provenance.json", "source-diff.patch")]+list(path.glob("*-test/metrics.json")):
                relative = file.relative_to(path)
                (local/relative).parent.mkdir(exist_ok=True)
                shutil.copyfile(file, local/relative)
                hashes[str(file.resolve())] = checksum(file)
                if experiment != "E001c":
                    assert previous["metadata_sha256"][str(file.resolve())] == checksum(file)
            inventory.append({"experiment": experiment, "remote_path": str(path.resolve()), "local_path": str(local.relative_to(output/"metadata"))})
    assert len(inventory) == 130
    write(output/"metadata/inventory.json", inventory)
    shutil.copyfile(prior, output/"prior-prediction-verification.json")
    evidence = {"old_metadata_unchanged": 700, "old_predictions_unchanged": 0, "primary_reconstructed": 0,
                "all_13_axes_reconstructed": 0, "identity_time_gt_conditions_checked": 0,
                "prediction_sha256": {}, "metadata_sha256": hashes}
    for path, digest in previous["prediction_sha256"].items():
        assert checksum(Path(path)) == digest
        evidence["old_predictions_unchanged"] += 1
    references, old_flags = {}, {}
    for target in ("loki", "road-waymo"):
        base = experiments/"E001b/runs"/SWEEPS["E001b"]/"loki-geometry-BiLSTM-seed0"/(target+"-test/predictions.npz")
        tracks = load_predictions(base)
        references[target] = signature(tracks)
        old_flags[target] = [(t["source_flags"]["bbox_valid"], t["source_flags"]["bbox_velocity_valid"]) for t in tracks]
    for item in inventory:
        if item["experiment"] != "E001c":
            continue
        path = Path(item["remote_path"]); config = read(path/"config.json")
        provenance = read(path/"provenance.json")
        source_manifest = experiments/"E001c/box-cache"/config["source"]/"manifest.json"
        assert checksum(source_manifest) == provenance["datasets"][config["source"]]["boxes"]["manifest_sha256"]
        assert read(source_manifest)["normalization"] == provenance["source_statistics"]
        for target in ("loki", "road-waymo"):
            prediction = path/(target+"-test/predictions.npz")
            tracks = load_predictions(prediction); expected = read(prediction.parent/"metrics.json")
            assert prediction_metrics(tracks, 4) == expected["primary"]
            assert signature(tracks) == references[target]
            for track, flags in zip(tracks, old_flags[target]):
                assert np.array_equal(track["source_flags"]["box2d_valid"], flags[0])
                assert np.array_equal(track["source_flags"]["box2d_velocity_valid"], flags[1])
            if config["settings"]["seed"] == 0:
                assert stratified_metrics(tracks, 4, STRATA) == expected["strata"]
                evidence["all_13_axes_reconstructed"] += 1
            evidence["primary_reconstructed"] += 1
            evidence["identity_time_gt_conditions_checked"] += 1
            evidence["prediction_sha256"][str(prediction.resolve())] = checksum(prediction)
        print(f"Audited {path.name}", flush=True)
    assert evidence["primary_reconstructed"] == 60 and evidence["all_13_axes_reconstructed"] == 12
    assert all(checksum(Path(path)) == digest for path, digest in hashes.items())
    assert all(checksum(Path(path)) == digest for path, digest in evidence["prediction_sha256"].items())
    evidence["source_metadata_and_new_predictions_unchanged_after"] = True
    write(output/"prediction-verification.json", evidence)


def analyze(root):
    inventory = read(root/"metadata/inventory.json")
    records, runs, sources, normalization, settings = {}, [], {}, {}, None
    metadata_hashes = read(root/"prediction-verification.json")["metadata_sha256"]
    populations = {}
    for item in inventory:
        path = root/"metadata"/item["local_path"]
        config, status, provenance, history = [read(path/n) for n in ("config.json", "status.json", "provenance.json", "history.json")]
        for file in [path/n for n in ("config.json", "status.json", "history.json", "provenance.json", "source-diff.patch")]+list(path.glob("*-test/metrics.json")):
            assert checksum(file) == metadata_hashes[item["remote_path"]+"/"+str(file.relative_to(path))]
        exp, source, feature, variant, seed = config["experiment"], config["source"], config["configuration"], config["variant"], config["settings"]["seed"]
        key = exp, source, feature, variant, seed
        assert status == {"status": "complete", "failures": [], "kind": "comparison"}
        assert not (path/"source-diff.patch").read_bytes() and config["kind"] == "comparison" and config["device"] == "cuda"
        assert config["classes"] == CLASSES
        shared = config["settings"] | {"seed": None}
        settings = shared if settings is None else settings
        assert shared == settings and shared["epochs"] == 75 and shared["patience"] == 8
        best, stale = None, 0
        for epoch, row in enumerate(history, 1):
            assert row["epoch"] == epoch
            best, stale, selected = select_epoch(best, row["validation_f1"], row["validation_ce"], epoch, stale)
            assert selected == row["selected"] and stale == row["non_improving_epochs"]
        assert best == provenance["selection"] and best["epoch"] == provenance["selected_checkpoint_epoch"]
        assert len(history) == provenance["completed_epochs"] <= 75
        assert (stale == 8 and provenance["stopping_reason"] == "patience") or (len(history) == 75 and provenance["stopping_reason"] == "budget")
        for dataset, ref in provenance["datasets"].items():
            comparable = {k: ref[k] for k in ("setup_sha256", "collection_sha256", "policy_sha256", "population", "support", "native_policy")}
            assert sources.setdefault(dataset, comparable) == comparable
        stats = provenance["source_statistics"]
        family = "K+T+R" if exp == "E001b" else "boxes" if exp == "E001c" else feature
        stats_key = stats["baseline"] if exp == "E001b" else stats
        assert normalization.setdefault((source, family), stats_key) == stats_key
        if exp == "E001c":
            assert variant == "B" and config["inputs"] == {"numeric_columns": list(COLUMNS[:INPUT_COLUMNS[feature]]), "flags": list(FLAGS)}
            assert config["model"]["input_dim"] == INPUT_COLUMNS[feature]+4
            assert "no ROAD fallback" in config["cache_policy"]["sources2d"]
        runs.append({"experiment": exp, "source": source, "configuration": feature, "model": "MLP" if variant == "A" else "BiLSTM", "seed": seed,
                     "revision": provenance["revision"], "selected_epoch": best["epoch"], "completed_epochs": len(history),
                     "stopping_reason": provenance["stopping_reason"], "parameters": provenance["parameters"], "runtime_seconds": provenance["elapsed_seconds"],
                     "wandb_url": provenance["wandb"]["url"], "remote_path": item["remote_path"], "local_metadata": item["local_path"]})
        for target in ("loki", "road-waymo"):
            result = read(path/(target+"-test/metrics.json"))
            assert result["split"] == "test" and key+(target,) not in records
            assert populations.setdefault(target, result["primary"]["denominators"]) == result["primary"]["denominators"]
            for weighting in ("raw", "track_weighted"):
                original = result["primary"][weighting]
                assert confusion_metrics(original["confusion"]) == original
            for slices in result["strata"].values():
                for score in slices.values():
                    assert confusion_metrics(score["track_weighted"]["confusion"], supported_only=True) == score["track_weighted"]
            records[key+(target,)] = result
    assert len(runs) == 130 and len(records) == 260
    for row in read(root/"prior-prediction-verification.json")["baseline_bbox_strata"]:
        records["E001", row["source"], "K+T+R", "B", row["seed"], row["target"]]["strata"].update(row["strata"])
    for model in MODELS:
        for source, target in DIRS:
            assert all((model[0], source, model[1], model[2], seed, target) in records for seed in range(5))
    verification = {"runs": len(runs), "test_cells": len(records), "populations": populations, "shared_settings": settings,
                    "shared_frozen_references": sources, "e001c_source_normalization_identical_across_configurations": True,
                    "stopping": {exp: {"selected_epoch_range": [min(r["selected_epoch"] for r in runs if r["experiment"] == exp), max(r["selected_epoch"] for r in runs if r["experiment"] == exp)],
                                      "completed_epoch_range": [min(r["completed_epochs"] for r in runs if r["experiment"] == exp), max(r["completed_epochs"] for r in runs if r["experiment"] == exp)],
                                      "budget_stops": sum(r["stopping_reason"] == "budget" for r in runs if r["experiment"] == exp),
                                      "revisions": sorted({r["revision"] for r in runs if r["experiment"] == exp})} for exp in SWEEPS}}
    csv_file(root, "runs.csv", runs); write(root/"verification.json", verification)
    return records


def report(root):
    data = analyze(root)
    def results(model, direction):
        return [data[model[0], direction[0], model[1], model[2], seed, direction[1]] for seed in range(5)]
    def values(model, direction, field="macro_f1", axis=None, label=None):
        return [(r["strata"][axis][label] if axis else r["primary"])["track_weighted"][field] for r in results(model, direction)]
    def base(model, direction):
        return {"experiment": model[0], "configuration": model[1], "model": title(model), "source": direction[0], "target": direction[1]}
    primary, classes, secondary, strata, confusion = [], [], [], [], []
    for model in MODELS:
        for direction in DIRS:
            records = results(model, direction); identifiers = base(model, direction)
            primary.append(identifiers | moments(values(model, direction)))
            row = identifiers.copy()
            for weighting in ("raw", "track_weighted"):
                matrices = np.array([r["primary"][weighting]["confusion"] for r in records])
                supports = matrices.sum(2, keepdims=True)
                normalized = np.divide(matrices, supports, out=np.zeros_like(matrices), where=supports != 0)
                confusion.append(identifiers | {"weighting": weighting, "mean_counts": matrices.mean(0).tolist(), "row_normalized_mean": normalized.mean(0).tolist(), "row_normalized_std": normalized.std(0, ddof=1).tolist()})
                for field in ("macro_f1", "accuracy", "support_weighted_f1"):
                    stats = moments([r["primary"][weighting][field] for r in records])
                    row[weighting+"_"+field] = stats
                for i in range(4):
                    classes.append(identifiers | {"weighting": weighting, "class": i,
                        "gt_frames": records[0]["primary"]["denominators"]["class_frames"][i],
                        **{field: moments([r["primary"][weighting][field][i] for r in records]) for field in ("precision", "recall", "f1")}})
            row.update({field: moments([r["primary"][field] for r in records], 1 if field == "track_ce" else 100) for field in ("track_ce", "mean_track_accuracy", "median_track_accuracy")})
            secondary.append(row)
            for axis, slices in records[0]["strata"].items():
                for label, first in slices.items():
                    assert all(r["strata"][axis][label]["denominators"] == first["denominators"] for r in records)
                    strata.append(identifiers | {"axis": axis, "bin": label, "denominators": first["denominators"],
                        "supported_classes": first["track_weighted"]["supported_classes"], "macro_f1": moments(values(model, direction, axis=axis, label=label)),
                        **{field: [moments([r["strata"][axis][label]["track_weighted"][field][i] for r in records]) for i in range(4)] for field in ("precision", "recall", "f1")}})
    paired_rows, slice_pairs = [], []
    for before, after in PAIRS:
        for direction in DIRS:
            row = {"before": title(before), "after": title(after), "source": direction[0], "target": direction[1]}
            paired_rows.append(row | paired(values(before, direction), values(after, direction)))
            for axis, bins in results(after, direction)[0]["strata"].items():
                for label, first in bins.items():
                    slice_pairs.append(row | {"axis": axis, "bin": label, "denominators": first["denominators"],
                        **paired(values(before, direction, axis=axis, label=label), values(after, direction, axis=axis, label=label))}) if axis in results(before, direction)[0]["strata"] else None
    for name, rows in (("summary", primary), ("per-class-summary", classes), ("secondary-summary", secondary), ("strata-summary", strata), ("confusion-summary", confusion), ("paired-differences", paired_rows), ("stratum-differences", slice_pairs)):
        write(root/(name+".json"), rows)
        csv_file(root, name+".csv", [{k:json.dumps(v) if isinstance(v, (dict, list)) else v for k,v in row.items()} for row in rows])
    tables = {"primary": table(["Model/input"]+NAMES, [[title(m)]+[fmt(values(m,d)) for d in DIRS] for m in MODELS])}
    tables["new-class-f1"] = table(["Direction", "BiLSTM input", "Moving", "Stopped", "Waiting", "Crossing"],
        [[name, m[1]]+[fmt([r["primary"]["track_weighted"]["f1"][i] for r in results(m,d)]) for i in range(4)] for d,name in zip(DIRS,NAMES) for m in NEW])
    tables["new-secondary"] = table(["Direction", "BiLSTM input", "Pooled macro-F1 (%)", "Track accuracy (%)", "Track CE"],
        [[name,m[1],fmt([r["primary"]["raw"]["macro_f1"] for r in results(m,d)]),fmt(values(m,d,"accuracy")),fmt([r["primary"]["track_ce"] for r in results(m,d)],1)] for d,name in zip(DIRS,NAMES) for m in NEW])
    tables["new-pairs"] = table(["Comparison"]+NAMES,
        [[title(after)+" − "+title(before)]+[fmt(np.array(values(after,d))-np.array(values(before,d)))+f" ({paired(values(before,d),values(after,d))['positive_seeds']}/5 ↑)" for d in DIRS] for before,after in PAIRS])
    focus = [("frame-bbox","missing"),("frame-bbox","valid"),("track-bbox","never"),("track-bbox","partial"),("transition","near"),("transition","steady")]
    rows=[]
    for direction,name in zip(DIRS,NAMES):
        for axis,label in focus:
            first=results(NEW[0],direction)[0]["strata"][axis][label];denom=first["denominators"]
            rows.append([name,axis+" "+label,f"{denom['groups']}/{denom['tracks']}/{denom['gt_frames']}; "+("".join("MSWC"[i] for i in first["track_weighted"]["supported_classes"]) or "—")]+[fmt(values(m,direction,axis=axis,label=label)) for m in NEW])
    tables["new-strata"]=table(["Direction","Slice","G/T/F; classes","boxes","+ 2D size vel","+ 2D/3D size vel"],rows)
    write(root/"tables.json", tables)
    (root/"tables.md").write_text("\n".join("## "+key+"\n\n"+value for key,value in tables.items()))
    plot(root, results, values)
    print(tables["primary"], flush=True);print(tables["new-pairs"], flush=True)


def plot(root, results, values):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plots=root/"plots";plots.mkdir(exist_ok=True)
    def save(fig,name):
        for extension in ("png","svg"):
            fig.savefig(plots/(name+"."+extension),dpi=160,bbox_inches="tight")
        plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(17,13),layout="constrained")
    for ax,direction,name in zip(axes.flat,DIRS,NAMES):
        for i,model in enumerate(MODELS):
            stats=moments(values(model,direction));ax.scatter(stats["seeds"],np.full(5,i),s=18,alpha=.65)
            ax.errorbar(stats["mean"],i,xerr=stats["std"],fmt="ks",capsize=3)
        ax.set(yticks=range(len(MODELS)),yticklabels=[title(m) for m in MODELS],title=name,xlabel="Track-weighted four-class macro-F1 (%)")
        ax.invert_yaxis();ax.grid(axis="x",alpha=.2)
    fig.suptitle("E001 / E001b / E001c · all five seeds; black square = mean ± sample SD")
    save(fig,"primary-f1")
    fig,axes=plt.subplots(2,2,figsize=(17,10),layout="constrained")
    for ax,direction,name in zip(axes.flat,DIRS,NAMES):
        for i,(before,after) in enumerate(PAIRS):
            score=paired(values(before,direction),values(after,direction));ax.scatter(score["seeds"],np.full(5,i),s=20,alpha=.65)
            ax.errorbar(score["mean"],i,xerr=score["std"],fmt="ks",capsize=3)
        ax.axvline(0,color="gray",lw=1);ax.set(yticks=range(len(PAIRS)),yticklabels=[a[1]+" − "+b[1] for b,a in PAIRS],title=name,xlabel="Same-seed macro-F1 difference (pp)");ax.invert_yaxis()
    fig.suptitle("E001c comparisons · input dimensions/initialization differ; 3D size velocities are redundant")
    save(fig,"e001c-differences")
    for direction,name in zip(DIRS,NAMES):
        fig,axes=plt.subplots(2,4,figsize=(17,9),layout="constrained")
        for ax,model in zip(axes.flat,[("E001","RAW","A")]+PLOT_MODELS):
            matrices=np.array([r["primary"]["track_weighted"]["confusion"] for r in results(model,direction)])
            normalized=100*(matrices/matrices.sum(2,keepdims=True)).mean(0)
            image=ax.imshow(normalized,vmin=0,vmax=100,cmap="Blues")
            for (i,j),value in np.ndenumerate(normalized):ax.text(j,i,f"{value:.1f}",ha="center",va="center",color="white" if value>55 else "black")
            ax.set(xticks=range(4),yticks=range(4),xticklabels=list("MSWC"),yticklabels=list("MSWC"),xlabel="Prediction",ylabel="GT",title=title(model),aspect="equal")
        fig.colorbar(image,ax=list(axes.flat),shrink=.7,label="Mean of seed-wise row-normalized percentages")
        fig.suptitle(name+" · five-seed track-weighted confusion comparison")
        save(fig,"confusions-"+name.lower().replace("→","-"))
        reference=results(PLOT_MODELS[1],direction)[0]["strata"]
        fig,axes=plt.subplots(5,3,figsize=(20,25),layout="constrained")
        models=[PLOT_MODELS[1],PLOT_MODELS[3]]+NEW
        for ax,(axis,bins) in zip(axes.flat,reference.items()):
            for offset,model in zip(np.linspace(-.28,.28,len(models)),models):
                points=[(i,moments(values(model,direction,axis=axis,label=label))) for i,label in enumerate(bins)]
                scored=[(i,s) for i,s in points if s["mean"] is not None]
                ax.errorbar([i+offset for i,s in scored],[s["mean"] for i,s in scored],yerr=[s["std"] for i,s in scored],fmt="o",capsize=2,label=model[1])
            labels=[]
            for label,score in bins.items():
                support=score["denominators"];codes="".join("MSWC"[i] for i in score["track_weighted"]["supported_classes"]) or "—"
                labels.append(f"{label}\nG/T={support['groups']}/{support['tracks']}\nF={support['gt_frames']} {codes}")
            ax.set(xticks=range(len(labels)),xticklabels=labels,xlim=(-.6,len(labels)-.4),ylim=(-3,103),title=axis,ylabel="Supported-class macro-F1 (%)");ax.tick_params(axis="x",labelsize=7);ax.grid(axis="y",alpha=.15)
        for ax in list(axes.flat)[len(reference):]:ax.axis("off")
        handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc="outside upper right",fontsize=9)
        fig.suptitle(name+" · mean ± sample SD across five seeds; empty slices unscored")
        save(fig,"strata-"+name.lower().replace("→","-"))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect",action="store_true")
    parser.add_argument("--experiments-root",type=Path,default=Path("outputs/experiments"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--prior-verification",type=Path)
    args=parser.parse_args()
    if args.collect:
        if args.prior_verification is None:parser.error("--collect requires --prior-verification")
        collect(args.experiments_root,args.output,args.prior_verification)
    else:report(args.output)


if __name__=="__main__":main()
