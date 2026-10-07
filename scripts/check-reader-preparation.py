"""Run inspectable end-to-end acceptance cases and preserve their saved-track evidence."""

import argparse
import json
import os
from pathlib import Path
import unittest

from pedestrian_behavior.inspection.prepared_tracks import write_inspector

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
fixtures = args.output / "fixtures"
os.environ["PREPARATION_ACCEPTANCE_OUTPUT"] = str(fixtures)


class Result(unittest.TextTestResult):
    records = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.records.append({"test": test.id(), "status": "passed"})

    def addFailure(self, test, error):
        super().addFailure(test, error)
        self.records.append({"test": test.id(), "status": "failed", "error": self._exc_info_to_string(error, test)})

    def addError(self, test, error):
        super().addError(test, error)
        self.records.append({"test": test.id(), "status": "error", "error": self._exc_info_to_string(error, test)})


suite = unittest.defaultTestLoader.discover("tests", pattern="test_track_preparation.py")
result = unittest.TextTestRunner(verbosity=2, resultclass=Result).run(suite)
(args.output / "checks.json").write_text(json.dumps(result.records, indent=2) + "\n")
cases = []
for path in sorted(fixtures.glob("*.npz")):
    expected = json.loads(path.with_name(path.stem + "-expected.json").read_text())
    cases.append({"name": path.stem, "path": path, "reason": "Synthetic native-format acceptance fixture", "expected": expected})
actual = write_inspector(args.output / "index.html", cases)
if not result.wasSuccessful() or any(not c["pass"] for case in actual for c in case["checks"]):
    raise SystemExit(1)
print(f"{len(result.records)} acceptance scenarios; {len(cases)} saved/reloaded fixture views: {args.output/'index.html'}")
