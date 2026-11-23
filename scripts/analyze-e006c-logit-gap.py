"""Compare selective-labeling scores on the saved E006c predictions."""

import argparse
import json
from pathlib import Path

import numpy as np

from pedestrian_behavior.evaluation import load_predictions
from pedestrian_behavior.experiments.e004a_analysis import _choose_threshold, _curve, _metrics, _population


def populations(path):
    tracks = load_predictions(path)
    softmax = _population(tracks)
    logits = np.concatenate([row["logits"][row["gt_valid"]] for row in tracks]).astype(np.float64)
    top_two = np.sort(logits, axis=1)[:, -2:]
    gap = top_two[:, 1] - top_two[:, 0]
    return {"max_softmax": softmax, "logit_gap": softmax | {"scores": gap}}


def measure(development, assessment):
    threshold, reject_all = _choose_threshold(_curve(development), .05)

    def accepted(population):
        scores = population["scores"]
        return np.zeros_like(scores, dtype=bool) if reject_all else scores >= threshold

    curve = _curve(assessment)
    fixed_coverage = {}
    for goal in (.30, .60):
        index = int(np.searchsorted(curve["coverage"], goal))
        fixed_coverage[str(goal)] = {key: float(curve[key][index]) for key in ("coverage", "error")}
    return {"threshold": threshold, "reject_all": reject_all,
            "development": _metrics(development, accepted(development)),
            "assessment": _metrics(assessment, accepted(assessment)),
            "assessment_at_fixed_coverage": fixed_coverage}


def compare(run, predictions, sources):
    original = json.loads((run / "summary.json").read_text())
    keys = {(source, 0, None) for source in sources} | {
        (source, budget, draw) for source in sources for budget in (25, 50, 100, 250) for draw in range(5)}
    actual = {(row["source"], row["budget"], row["draw"]) for row in original["results"]}
    if (original["status"] != "complete" or original["fit_count"] != len(keys) - len(sources)
            or len(original["results"]) != len(keys) or actual != keys):
        raise ValueError("Expected the complete E006c comparison")
    rows = []
    for row in original["results"]:
        folder = predictions / row["source"] / Path(row["fit_dir"]).name
        development = populations(folder / "development-predictions.npz")
        assessment = populations(folder / "assessment-predictions.npz")
        scores = {key: measure(development[key], assessment[key]) for key in development}
        baseline = next(item for item in row["coverage"]["targets"] if item["error_target"] == .05)
        replay = scores["max_softmax"]
        if (replay["threshold"] != baseline["threshold"] or replay["reject_all"] != baseline["reject_all"]
                or any(replay[split][key] != baseline[original_split][key]
                       for split, original_split in (("development", "calibration"), ("assessment", "assessment"))
                       for key in ("coverage", "error", "accepted_frames"))
                or development["max_softmax"]["track_count"] != 194
                or assessment["max_softmax"]["track_count"] != 469):
            raise ValueError(f"Saved max-softmax evidence did not replay for {folder}")
        rows.append({key: row[key] for key in ("source", "budget", "draw", "macro_f1")} | {"scores": scores})
    return {"source_run": str(run), "calibration_error_target": .05, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--loki-run", type=Path, help="Completed LOKI-only extension")
    parser.add_argument("--predictions", type=Path, help="Directory containing the saved source prediction trees")
    parser.add_argument("--output", required=True, type=Path, help="Fresh JSON file")
    args = parser.parse_args()
    result = compare(args.run, args.predictions or args.run, ("road-waymo", "joint"))
    if args.loki_run:
        result["loki_source_run"] = str(args.loki_run)
        result["rows"].extend(compare(args.loki_run, args.predictions or args.loki_run, ("loki",))["rows"])
    with args.output.open("x") as stream:
        json.dump(result, stream, allow_nan=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
