"""Run one E001b BiLSTM attempt using the verified native-only bbox extension."""
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from pedestrian_behavior.experiments.e001b_run import main

if __name__ == "__main__":
    main()
