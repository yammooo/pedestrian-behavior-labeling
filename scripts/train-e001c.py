"""Run one E001c BiLSTM attempt from the verified native box cache."""
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from pedestrian_behavior.experiments.e001c_run import main

if __name__ == "__main__":
    main()
