"""Extract frozen E002d source-model representations on their own validation split."""

import argparse
import json
from pathlib import Path

import torch

from pedestrian_behavior.data.preparation import checksum
from pedestrian_behavior.experiments import e001b, e001b_run, e002a_run, e002b, e002_transfer, e003b_run
from pedestrian_behavior.experiments.run import write_json
from pedestrian_behavior.training import seed_run


def extract(args):
    outputs_root, output = args.outputs_root.resolve(), args.output.resolve()
    expected_root = outputs_root / "experiments/E003b/embedding-diagnostic-20261105/input"
    if args.source not in e003b_run.SOURCES or output.exists() or not output.is_relative_to(expected_root):
        raise ValueError("Use one fresh own-validation cache under the E003b embedding-diagnostic input directory")
    if args.device not in ("cuda", "cpu") or (args.device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Requested extraction device is unavailable")

    source_assembly = e001b_run.assemble(args.source, "geometry", "B",
        e001_root=outputs_root / "experiments/E001", e001b_root=outputs_root / "experiments/E001b")
    normalization = source_assembly["normalization"]
    source_data, _ = e003b_run._baselines(source_assembly, args.source, normalization)
    source_val_ids = {entry["locator"] for entry in source_data["validation"].entries}
    frozen = e003b_run._load_source(outputs_root, args.source, "e002d", normalization, source_val_ids)

    setups, bbox_records = {}, {}
    for dataset in e003b_run.SOURCES:
        setup_path = outputs_root / "experiments/E001/data-setup" / dataset / "setup.json"
        collection = outputs_root / "experiments/E001/reader-preparation" / dataset
        extension = outputs_root / "experiments/E001b/bbox-native" / dataset
        setups[dataset] = (setup_path, json.loads(setup_path.read_text()), collection, extension)
        bbox_records[dataset] = e001b.load_extension(collection, setup_path, extension)

    reader = e002_transfer._Target(args.source, outputs_root, setups, bbox_records,
                                   args.probe, args.rgb_cache_root)
    try:
        supply = reader.supply({"input": "geometry+rgb+lidar"}, frozen["run_config"])
        catalog, points = reader.catalog, reader.points
        if source_data["validation"].entries and set(catalog.tracks) != {
                entry["locator"] for split in source_data.values() for entry in split.entries}:
            raise ValueError("Prepared feature catalog differs from the frozen source population")
        extraction = reader.extraction
        e003b_run.e003a_run._validate_rgb_preparation(
            args.rgb_preparation, args.source, catalog, extraction, args.rgb_cache_root)
        profile = e002a_run.validate_profile(frozen["run_config"]["profile"])
        dataset = e002a_run.FeatureDataset(source_data["validation"], supply, args.source, e003b_run.ARM)

        seed_run(0)
        model = e002b.build_model(e003b_run.ARM, e003b_run.VARIANT, profile,
            dropout=0.1, rgb_branch_dropout=0.5).to(args.device)
        model.load_state_dict(frozen["checkpoint"]["model_state"], strict=True)
        e003b_run._freeze_source(model)
        output.parent.mkdir(parents=True, exist_ok=True)
        records = e003b_run._extract(dataset, model, profile, torch.device(args.device),
                                     "own-validation", output.parent)
        prediction_path = Path(frozen["identity"]["validation_predictions"])
        e003b_run._check_source_predictions(e003b_run._source_tracks(records), prediction_path)
        identity = {"experiment": "E003b embedding diagnostic", "source": args.source,
            "dataset": args.source, "split": "validation", "tracks": len(records),
            "checkpoint": frozen["identity"], "source_normalization": normalization,
            "profile": profile, "target_catalog": catalog.identity, "target_points": points.identity,
            "rgb_preparation_sha256": checksum(args.rgb_preparation),
            "validation_predictions_sha256": checksum(prediction_path),
            "extractor_sha256": checksum(Path(__file__)),
            "e003b_run_sha256": checksum(Path(e003b_run.__file__))}
        e003b_run._save_cache(output, records, identity)
        e003b_run._load_cache(output, identity)
        write_json(output.parent / f"{args.source}-extraction.json", {
            "cache": output.name, "cache_sha256": checksum(output), "identity": identity,
            "source_prediction_check": "passed"})
    finally:
        reader.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("road-waymo", "loki"), required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), required=True)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe", type=Path, default=Path("outputs/experiments/E002a/native-probe-07"))
    parser.add_argument("--rgb-cache-root", type=Path, default=Path("outputs/experiments/E002a/rgb-cache"))
    parser.add_argument("--rgb-preparation", type=Path,
        default=Path("outputs/experiments/E002a/rgb-cache-preparation-03/cache.json"))
    extract(parser.parse_args())


if __name__ == "__main__":
    main()
