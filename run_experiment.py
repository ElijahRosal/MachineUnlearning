from __future__ import annotations

from pathlib import Path

from machine_unlearning.experiment import run_experiment


if __name__ == "__main__":
    # Keep all generated demo outputs in one folder so they are easy to find.
    output = Path("artifacts")
    summary = run_experiment(output)

    # Printing the summary makes the command-line demo self-contained.
    print(summary)
