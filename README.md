# Machine Unlearning Final Project

This project implements a small machine unlearning pipeline in PyTorch. It trains a baseline classifier, retrains after removing selected data, and runs an approximate unlearning pass for comparison. The main experiment now uses a harder class-removal scenario so the unlearning effect is visible.

## Dependencies

- Python 3.10+ recommended
- PyTorch
- Matplotlib
- Jupyter / nbconvert if you want to run the notebook workflow

The project was developed in a local `torch_cpu` conda environment, but any Python environment with the dependencies installed should work.

Activate your preferred environment before running the commands below.

## How to Run

From the project root, run the main experiment and write results to `artifacts/results.json`:

```powershell
python run_experiment.py
```

Run the notebook demo:

```powershell
python -m nbconvert --to notebook --execute notebooks/experiment_notebook.ipynb --inplace
```

## Main Code Locations

- Training loop and evaluation: `machine_unlearning/train.py`
- Model definition: `machine_unlearning/model.py`
- Data generation and splitting: `machine_unlearning/data.py`
- Unlearning logic: `machine_unlearning/unlearning.py`
- End-to-end experiment runner: `machine_unlearning/experiment.py` and `run_experiment.py`
- Notebook demo / presentation workflow: `notebooks/experiment_notebook.ipynb`

The main experiment writes:

- Raw metrics: `artifacts/results.json`
- Performance visualization: `artifacts/accuracy_comparison.png`
- Efficiency visualization: `artifacts/runtime_comparison.png`

## Project Summary

This is a theoretical project focused on implementation, training, experiments, and results. It compares full retraining with an approximate retain-and-scrub unlearning approach on a synthetic Gaussian-mixture dataset. The default demo removes all examples from one class, then reports accuracy, loss, runtime, speedup, forgetting gap, and removed-set accuracy drop.
