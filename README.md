# Optimizers by hand

AdamW and Lion written from scratch, weight animations of how neurons learn under each, and a Lion "dead-zone" variant tracked with MLflow.

Part of a series of repos from my 2026 machine-learning sabbatical: a designer learning ML by building models from scratch and running experiments.

## What's here

- `optimizers/optimizers.ipynb`, `lion_deadzone.py`: the optimizers and the variant
- `mlflow/mlflow_lion_experiment.ipynb`: 18 tracked runs on EMNIST letters
- `mlflow/mlflow_serve.ipynb`: serving a small CNN with MLflow

## Running it

`pip install mlflow` for the MLflow notebooks.

## Results

The dead-zone variant gave no gain (negative result).

**Code:** written by hand in Jupyter notebooks, with a chatbot as a tutor for questions and plotting helpers.

**Requirements:** Python 3.11+, PyTorch 2.x, torchvision, matplotlib, a CUDA GPU for anything beyond MNIST.
