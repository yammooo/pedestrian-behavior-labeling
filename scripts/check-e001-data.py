"""Write E001/E001b data/model/training/evaluation expected/actual acceptance."""

import argparse
import html
import json
import os
from pathlib import Path
import unittest

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
os.environ["E001_ACCEPTANCE_OUTPUT"] = str(args.output / "checks")
suite = unittest.defaultTestLoader.discover("tests", pattern="test_e001*.py")
result = unittest.TextTestRunner(verbosity=2).run(suite)
checks = [check for path in sorted((args.output / "checks").glob("*.json")) for check in json.loads(path.read_text())]
rows = "".join("<tr><td>" + html.escape(c["check"]) + "</td><td><pre>" + html.escape(json.dumps(c["expected"], indent=2))
               + "</pre></td><td><pre>" + html.escape(json.dumps(c["actual"], indent=2)) + "</pre></td></tr>" for c in checks)
(args.output / "index.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>E001/E001b data/model/training/evaluation acceptance</title>'
    '<style>body{font:16px system-ui;margin:2rem}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:.6rem;text-align:left;vertical-align:top}pre{white-space:pre-wrap;max-width:38rem}</style>'
    f'<h1>E001/E001b data/model/training/evaluation acceptance</h1><p>{result.testsRun} scenarios; {len(result.skipped)} skipped; {len(checks)} expected/actual checks; '
    f'{"PASS" if result.wasSuccessful() else "FAIL"}.</p><table><tr><th>Check</th><th>Hand-written expected</th><th>Actual</th></tr>{rows}</table></html>')
if not result.wasSuccessful():
    raise SystemExit(1)
print(args.output / "index.html")
