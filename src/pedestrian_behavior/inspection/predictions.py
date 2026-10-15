"""Local prediction review server: existing results in, separate review notes out."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import threading
from urllib.parse import parse_qs, urlparse
import uuid

from .prediction_context import ContextCache
from .prediction_index import PredictionIndex, write_json


def arguments(params):
    return {"runs": params.get("runs", "").split(","), "dataset": params.get("dataset", "loki"),
            "metric": params.get("metric", "accuracy"), "axis": params.get("axis", ""),
            "value": int(params.get("value", 0)), "scope": params.get("scope", "matching"),
            "confusion": params["focus"].split(",") if params.get("focus") else None,
            "sort": params.get("sort", "0"),
            "descending": None if not params.get("descending") else params["descending"] == "true",
            "search": params.get("search", ""), "minimum": int(params.get("minimum", 0)),
            "page": int(params.get("page", 0))}


class Viewer:
    def __init__(self, outputs, cache, axes, **native_roots):
        self.index = PredictionIndex(outputs, cache, axes)
        self.context = ContextCache(outputs, cache, **native_roots)
        self.cache = Path(cache).resolve()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="native-preview")
        self.jobs, self.futures = {}, {}
        self.lock = threading.RLock()
        self.notes_path = self.cache/"reviews.json"

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)

    def media(self, run, dataset, locator):
        track = self.index.load(run, dataset)["tracks"][locator]
        key = hashlib.sha256(json.dumps([dataset, locator, track["archive"]]).encode()).hexdigest()
        with self.lock:
            if key not in self.jobs or self.jobs[key]["status"] == "cancelled":
                for old, future in self.futures.items():
                    if future.cancel():
                        self.jobs[old] = {"status": "cancelled"}
                self.jobs[key] = {"status": "queued", "done": 0, "total": len(track["targets"])}
                self.futures[key] = self.executor.submit(self._prepare, key, dataset, track)
            return {"job": key, **self.jobs[key]}

    def _prepare(self, key, dataset, track):
        def progress(done, total):
            with self.lock:
                self.jobs[key].update(done=done, total=total)
        try:
            with self.lock:
                self.jobs[key]["status"] = "preparing"
            result = self.context.prepare(dataset, track["archive"], track["locator"], progress, expected=track)
            with self.lock:
                self.jobs[key].update(status="ready", result=result)
        except Exception as error:
            with self.lock:
                self.jobs[key].update(status="unavailable", error=f"{type(error).__name__}: {error}")

    def notes(self):
        with self.lock:
            return json.loads(self.notes_path.read_text()) if self.notes_path.exists() else []

    def save_case(self, body):
        state = body["state"]
        selected = self.index.detail(state["runs"].split(","), state["dataset"], state["locator"])
        if body["hashes"] != [m["sha256"] for m in selected["models"]]:
            raise ValueError("Predictions changed; reload the track before saving")
        frame = int(state["frame"])
        if not 0 <= frame < len(selected["targets"]):
            raise ValueError("Invalid review frame")
        self.index.query(**arguments(state))
        note = str(body.get("note", ""))
        if len(note) > 4000:
            raise ValueError("Review note exceeds 4000 characters")
        record = {"id": uuid.uuid4().hex, "saved_utc": datetime.now(timezone.utc).isoformat(),
                  "state": state, "hashes": body["hashes"], "source_time_us": selected["source_times_us"][frame],
                  "grid_time_us": selected["grid_times_us"][frame], "reason": body.get("reason", ""), "note": note}
        with self.lock:
            records = self.notes()
            records.append(record)
            write_json(self.notes_path, records)
        return record


def handler(viewer):
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, value, content_type="application/json"):
            data = json.dumps(value, allow_nan=False).encode() if content_type == "application/json" else value
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store" if content_type == "application/json" else "private, max-age=3600")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            parsed = urlparse(self.path)
            params = {k: values[-1] for k, values in parse_qs(parsed.query, keep_blank_values=True).items()}
            try:
                if parsed.path == "/":
                    self.send(200, Path(__file__).with_name("predictions.html").read_bytes(), "text/html; charset=utf-8")
                elif parsed.path == "/viewer.js":
                    self.send(200, Path(__file__).with_name("predictions.js").read_bytes(), "text/javascript; charset=utf-8")
                elif parsed.path == "/api/runs":
                    self.send(200, viewer.index.catalog())
                elif parsed.path == "/api/tracks":
                    with viewer.index.lock:
                        self.send(200, viewer.index.query(**arguments(params)))
                elif parsed.path == "/api/track":
                    with viewer.index.lock:
                        self.send(200, viewer.index.detail(params["runs"].split(","), params["dataset"], params["locator"]))
                elif parsed.path == "/api/media":
                    with viewer.index.lock:
                        self.send(200, viewer.media(params["run"], params["dataset"], params["locator"]))
                elif parsed.path == "/api/job":
                    with viewer.lock:
                        self.send(200, viewer.jobs[params["id"]])
                elif parsed.path == "/api/reviews":
                    self.send(200, viewer.notes())
                elif parsed.path == "/media":
                    key, image = params["key"], params["image"]
                    if not re.fullmatch(r"[a-f0-9]{64}", key) or not re.fullmatch(r"\d{4,}-(rgb|rgb-road|bev|crop)\.jpg", image):
                        raise ValueError("Invalid media address")
                    self.send(200, (viewer.cache/"context"/key/image).read_bytes(), "image/jpeg")
                else:
                    self.send(404, {"error": "Unknown route"})
            except (ValueError, KeyError, TypeError, OSError) as error:
                self.send(400, {"error": str(error)})

        def do_POST(self):
            try:
                if self.path != "/api/reviews":
                    self.send(404, {"error": "Unknown route"})
                    return
                origin = self.headers.get("Origin")
                if origin and urlparse(origin).netloc != self.headers.get("Host"):
                    raise ValueError("Cross-origin writes are not allowed")
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("Expected application/json")
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 32768:
                    raise ValueError("Invalid review size")
                body = json.loads(self.rfile.read(length))
                with viewer.index.lock:
                    self.send(201, viewer.save_case(body))
            except (ValueError, KeyError, TypeError, OSError) as error:
                self.send(400, {"error": str(error)})

        def log_message(self, format, *args):
            if args and str(args[1]) not in ("200", "201"):
                super().log_message(format, *args)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--cache", type=Path, default=Path("outputs/inspection/prediction-viewer"))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--loki-root", type=Path)
    parser.add_argument("--road-index", type=Path)
    parser.add_argument("--waymo-root", type=Path)
    args = parser.parse_args()
    # Experiment policy is wired here; the index and renderer receive ordinary data.
    from pedestrian_behavior.experiments.e001b_run import STRATA
    viewer = Viewer(args.outputs_root, args.cache, STRATA, loki_root=args.loki_root,
                    road_index=args.road_index, waymo_root=args.waymo_root)
    if viewer.cache.is_relative_to(viewer.index.outputs/"experiments"):
        parser.error("Keep viewer caches/notes outside experiment directories")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(viewer))
    print(f"Prediction viewer: http://127.0.0.1:{server.server_port} (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        viewer.close()


if __name__ == "__main__":
    main()
