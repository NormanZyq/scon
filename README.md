# Source Code of SCon

Source code of submission *From Scores to Interventions: Score-Conditioned Gating for Message Passing in Node Anomaly Detection*

## Overview
WIP. 

Currently we provide the running commands for quick run.

After cloning the repository, create a directory `data` first by 

```
mkdir data
```

<!-- ## Architecture

## Requirements
- Language: Python 3.10
- Dependencies: 
- Hardware: 

## Installation
WIP -->

## How to Run
Clone the repository.

An example of running the code:
```bash
# Single run
python src/train.py experiment=disney
```

```bash
# Multi-run
python src/multi_train.py experiment=disney
```

## Project Structure
```text
.
├── configs/            # Configuration files
├── notebooks/          # Jupyter notebooks
├── scripts/            # Helper scripts
├── src/                # Source code
│   ├── data/           # Data loading / processing
│   ├── models/         # Model definitions
│   ├── utils/          # Utility functions
│   ├── train.py        # Training entrypoint
│   ├── eval.py         # Evaluation entrypoint
│   └── multi_train.py  # Multi-run training
├── tests/              # Tests
├── requirements.txt    # Pip dependencies
├── environment.yaml    # Conda environment
├── pyproject.toml      # Tooling configuration
├── setup.py            # Package setup
└── Makefile            # Common commands
```

## Citation
Not available.
