"""Run one E001d matched-capacity MLP attempt."""

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from pedestrian_behavior.experiments.e001d_run import main

if __name__ == "__main__":
    main()
