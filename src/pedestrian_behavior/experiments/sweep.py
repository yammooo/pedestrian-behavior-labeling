"""Bounded independent attempt processes, fresh outputs and failure containment."""

from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import time

from .run import write_json


def run_sweep(output, plan, command, jobs=2, seeds=(0, 1, 2, 3, 4), device="cuda"):
    if jobs < 1 or not seeds or len(set(seeds)) != len(seeds) or any(not 0 <= s < 2**32 for s in seeds):
        raise ValueError("Positive jobs and distinct seeds in [0, 2**32) are required")
    names = [row["name"] for row in plan]
    if not names or len(set(names)) != len(names) or any(not n or Path(n).name != n or n in (".", "..") for n in names):
        raise ValueError("Nonempty attempts with unique directory names are required")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/"logs").mkdir()
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
                row.update(status="starting", command=command(row, output, device))
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

