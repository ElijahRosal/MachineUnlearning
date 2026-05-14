from __future__ import annotations

from pathlib import Path

from machine_unlearning.experiment import run_experiment


if __name__ == "__main__":
    output = Path("artifacts")
    summary = run_experiment(output)
    print(summary)
