"""Run the agreed E001 repetitions in separate, bounded concurrent processes."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .e001 import FEATURE_SETS
from .e001_run import write_json


def attempts(seeds):
    return [{"source": source, "configuration": features, "variant": variant, "seed": seed,
             "name": f"{source}-{features}-{'MLP' if variant == 'A' else 'BiLSTM'}-seed{seed}"}
            for seed in seeds for source in ("road-waymo", "loki")
            for features in FEATURE_SETS for variant in ("A", "B")]


def attempt_command(attempt, output, device):
    return [sys.executable, "scripts/train-e001.py", "--source", attempt["source"],
            "--configuration", attempt["configuration"], "--variant", attempt["variant"],
            "--seed", str(attempt["seed"]), "--epochs", "75", "--patience", "8",
            "--device", device, "--output", str(output/attempt["name"])]


def run_sweep(output, jobs=2, seeds=(0, 1, 2, 3, 4), device="cuda"):
    if jobs < 1 or not seeds or len(set(seeds)) != len(seeds) or any(not 0 <= s < 2**32 for s in seeds):
        raise ValueError("Positive jobs and distinct seeds in [0, 2**32) are required")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/"logs").mkdir()
    plan = attempts(seeds)
    state = {"status": "incomplete", "started_utc": datetime.now(timezone.utc).isoformat(),
             "jobs": jobs, "device": device, "epochs": 75, "patience": 8, "seeds": list(seeds),
             "cpu_threads_per_process": 2, "attempts": plan, "failures": []}
    environment = os.environ | {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                                "OPENBLAS_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"}
    queued, active = iter(plan), {}
    write_json(output/"sweep.json", state)
    try:
        while True:
            while len(active) < jobs:
                row = next(queued, None)
                if row is None:
                    break
                row.update(status="starting", command=attempt_command(row, output, device))
                with (output/"logs"/(row["name"]+".log")).open("wb") as log:
                    process = subprocess.Popen(row["command"], stdout=log, stderr=subprocess.STDOUT, env=environment)
                active[row["name"]] = (process, row)
                row["status"] = "running"
                print("Started", row["name"], flush=True)
                write_json(output/"sweep.json", state)
            if not active:
                break
            finished = [(name, process, row) for name, (process, row) in active.items() if process.poll() is not None]
            for name, process, row in finished:
                del active[name]
                row.update(returncode=process.returncode, status="complete" if process.returncode == 0 else "failed")
                print(row["status"].capitalize(), name, flush=True)
            if finished:
                write_json(output/"sweep.json", state)
            failed = [row["name"] for _, process, row in finished if process.returncode != 0]
            if failed:
                raise RuntimeError("Attempt failed; see logs: "+", ".join(failed))
            if not finished:
                time.sleep(.2)
        state["status"] = "complete"
        return state
    except BaseException as error:
        state["status"] = "failed"
        state["failures"].append({"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        for process, row in active.values():
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            row.update(status="interrupted", returncode=process.returncode)
        for row in plan:
            row.setdefault("status", "not-started")
            if row["status"] == "starting":
                row["status"] = "failed"
        state["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output/"sweep.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments/E001/runs")/("seeds-"+timestamp))
    parser.add_argument("--jobs", type=int, default=2, help="Maximum simultaneous attempts on the selected device")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    run_sweep(**vars(parser.parse_args()))
