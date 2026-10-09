"""Run a single E001 attempt with recorded seed and training budget."""
import os

# cuBLAS reads this before its first handle is created.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from pedestrian_behavior.experiments.e001_run import main

if __name__ == "__main__":
    main()
