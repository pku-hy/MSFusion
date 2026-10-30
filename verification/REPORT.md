# Execution validation

Date: October 30, 2026. Executed by Codex on YJY.
Working directory: `/home/sangibb/hy/msfusion/source_20261030/MSFusion_paper`.

## Environment

Linux, Python 3.12, PyTorch 2.5.1+cu121 (CPU execution), NumPy 1.26.4, pandas 2.3.3, scikit-learn 1.7.2, and Matplotlib 3.9.2.
Checks used an existing Anaconda environment. Installation in a clean environment and GPU training have not been verified.

## Real-data integration

The original project's `dataset/ETT-small/ETTh1.csv` contains 17,420 rows and seven channels. The fixed ETT benchmark splits contain 8,640/2,880/2,880 time points.

```bash
python train.py --check-only
python train.py --smoke --device cpu --batch-size 128 --run-id quickstart_real_ETTh1
python train.py --evaluate --device cpu --batch-size 128 --run-id quickstart_real_ETTh1
```

Completed checks include one epoch over all training windows for each stage, validation-based checkpoint selection, saving all three checkpoints, evaluation on all 2,785 test windows, and independent checkpoint reloading. Reloaded metrics matched the original evaluation exactly.

| Stage | MSE | MAE |
| --- | --- | --- |
| I | 0.4075781107 | 0.4204682410 |
| II | 0.4111363888 | 0.4243333042 |
| III | 0.4084532857 | 0.4223066568 |

These are short integration checks, not converged training results or paper reproduction. See `smoke_training.log` and `smoke_evaluation.log` in this directory.

## Data formats and model checks

`python verification/check_loaders.py` generates 1,200 rows with three channels using a fixed random seed. Both custom CSV and headerless Solar data complete three-stage training, checkpoint saving, and testing. The checks passed. Synthetic data do not establish accuracy on real Weather, Electricity, Traffic, or Solar datasets.

`python smoke_model.py` passed checks for scale output lengths, preservation of pre-fusion predictions, recursive fusion, and finite backward gradients.

Real ETTm and other real datasets, GPU execution, clean dependency installation, and full benchmark accuracy remain unverified.
