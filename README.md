# MSFusion

MSFusion is a long-term multivariate time series forecasting model with scale-aligned supervision and coarse-to-fine prediction fusion. Training includes scale-specific predictor learning, fusion learning with frozen predictors, and joint fine-tuning.

## Installation

Use Python 3.10-3.12. Clone the repository and install the dependencies:

```bash
git clone https://github.com/pku-hy/MSFusion.git
cd MSFusion
python -m venv .venv
```

Activate the environment:

```bash
# Linux / macOS
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
python -m pip install -r requirements.txt
```

The model runs on CPU or CUDA. For GPU training, install a PyTorch build compatible with your GPU driver.

## Dataset preparation

Place the dataset files under `dataset/`. Only the dataset selected for a run is required. Datasets and pretrained weights are not included.

```text
dataset/
|-- ETT-small/
|   |-- ETTh1.csv
|   |-- ETTh2.csv
|   |-- ETTm1.csv
|   `-- ETTm2.csv
|-- weather/weather.csv
|-- electricity/electricity.csv
|-- traffic/traffic.csv
`-- solar/solar_AL.txt
```

| Dataset option | File relative to dataset/ | Sampling interval |
| --- | --- | --- |
| ETTh1 / ETTh2 | ETT-small/ETTh1.csv / ETTh2.csv | 1 hour |
| ETTm1 / ETTm2 | ETT-small/ETTm1.csv / ETTm2.csv | 15 minutes |
| Weather | weather/weather.csv | 10 minutes |
| Electricity | electricity/electricity.csv | 1 hour |
| Traffic | traffic/traffic.csv | 1 hour |
| Solar | solar/solar_AL.txt | 10 minutes |

CSV files must start with a `date` column followed by numeric feature columns. Every numeric column is predicted. Timestamps must be unique, increasing, and aligned to a regular sampling interval. Resolve missing or non-finite values before training.

```csv
date,sensor_1,sensor_2,OT
2020-01-01 00:00:00,1.2,3.4,5.6
2020-01-01 01:00:00,1.3,3.5,5.8
```

Solar uses a headerless, comma-separated numeric matrix without a date column.

ETT uses fixed 12/4/4-month benchmark splits: ETTh needs at least 14,400 rows and ETTm at least 57,600 rows. Other datasets use chronological 70%/10%/20% training/validation/test splits. Standardization is fitted on the training split only. Validation and test input windows may include preceding historical observations, while their targets remain inside the corresponding split.

## Ready-to-run scripts

After installing the dependencies and preparing the data, run a dataset script:

```bash
bash scripts/ETTh1.sh
bash scripts/ETTh2.sh
bash scripts/ETTm1.sh
bash scripts/ETTm2.sh
bash scripts/Weather.sh
bash scripts/Electricity.sh
bash scripts/Traffic.sh
bash scripts/Solar.sh
```

Each script runs all four prediction horizons (96, 192, 336, and 720) sequentially. To run only selected horizons, pass them as arguments:

```bash
bash scripts/ETTh1.sh 96
bash scripts/Weather.sh 96 192
```

Run the complete eight-dataset benchmark (32 training runs):

```bash
bash scripts/all.sh
```

Configure the environment before running a script:

```bash
DEVICE=cuda GPU=0 BATCH_SIZE=32 bash scripts/ETTh1.sh
DATASET_DIR=/path/to/dataset DEVICE=cpu bash scripts/Weather.sh 96
PYTHON=/path/to/python EPOCHS=100 bash scripts/all.sh
```

| Environment variable | Default | Purpose |
| --- | --- | --- |
| PYTHON | python | Python executable from the installed environment |
| DATASET_DIR | Repository dataset/ | Dataset root directory |
| DEVICE | auto | auto, cpu, or cuda |
| GPU | 0 | GPU index |
| BATCH_SIZE | 16 | Batch size |
| EPOCHS | 100 | Total three-stage epoch budget |
| SEQ_LEN | 336 | Input history length |
| LOG_DIR | Repository logs/ | Console log directory |
| RUN_TAG | Timestamp and process ID | Shared experiment tag |

Scripts print each Run ID and save its output to `logs/<run_id>.log`. Runs execute sequentially and stop if a command fails. Every horizon receives a distinct Run ID. Use a new RUN_TAG for repeated experiments to preserve existing outputs. The scripts locate the repository relative to their own location and can be invoked from another working directory. Use Bash on Linux/macOS or Git Bash/WSL on Windows; the Python commands below also work directly in PowerShell.

## Training with Python

After installing dependencies and placing ETTh1 in the default location, run:

```bash
python train.py
```

The launcher detects the channel count, validates the data, trains all three stages, selects checkpoints using validation loss, and evaluates the test split.

Select another dataset, prediction horizon, or device:

```bash
python train.py --dataset Weather --pred-len 192 --device cuda --gpu 0 --run-id weather_192
python train.py --dataset ETTh1 --device cpu --run-id etth1_96
```

Use a different data directory or file:

```bash
python train.py --dataset Weather --dataset-dir /path/to/dataset
python train.py --dataset ETTh1 --file /path/to/ETTh1.csv
```

Quote paths containing spaces, for example `--file "D:/Time Series/ETTh1.csv"`.

### Custom datasets

Prepare a CSV in the format above and specify the sampling frequency:

```bash
python train.py --dataset custom --file ./dataset/my_data.csv --freq h --run-id my_model
```

The number of channels is inferred automatically. For shorter datasets, reduce the input length, prediction horizon, or batch size. The training split must contain complete input and forecast windows, and each validation/test target split must contain at least one forecast horizon.

### Options

| Option | Default | Description |
| --- | --- | --- |
| --dataset | ETTh1 | Dataset preset, or custom |
| --seq-len | 336 | Input history length |
| --pred-len | 96 | Forecast length: 96, 192, 336, or 720 |
| --epochs | 100 | Total budget across three stages; minimum 3 |
| --batch-size | 16 | Training and evaluation batch size |
| --device | auto | auto, cpu, or cuda |
| --gpu | 0 | GPU index visible to the process |
| --run-id | Dataset name and timestamp | Output identifier; use a new ID for each training run |

Defaults include factor-two average pooling, two downsampling layers, predictor dimension 16, AdamW, learning rate 0.001, random seed 2021, and auxiliary loss coefficient 0.1. A 100-epoch budget is split into 33/33/34 epochs with early-stopping patience of 5/3/3. For additional model arguments, see `python run.py --help`; `run.py` implements the same training pipeline used by `train.py`.

## Results and checkpoint evaluation

Outputs are saved relative to the repository root:

```text
checkpoints/<setting>/
    config.json
    checkpoint_stage1_best.pth
    checkpoint_stage2_best.pth
    checkpoint_stage3_best.pth
test_results/<setting>/
    stage1/metrics.json
    stage2/metrics.json
    stage3/metrics.json
```

`config.json` records the run arguments. Metrics files contain MSE and MAE in standardized value space; lower values are better. Stage I evaluates the independent original-resolution prediction. Stages II and III evaluate the fused prediction. The Stage III checkpoint is the final model.

Reload checkpoints using the original dataset, Run ID, input length, and forecast horizon:

```bash
python train.py --dataset ETTh1 --evaluate --device cpu --run-id etth1_96
python train.py --dataset Weather --pred-len 192 --evaluate --run-id weather_192
```

Evaluation updates the corresponding metrics files. New training runs should use distinct Run IDs to preserve previous checkpoints. The lower-level `run.py --plot` option saves sample prediction PDFs.

## Troubleshooting

- **Dataset not found:** Check the path or specify it using --file.
- **Insufficient data:** Reduce window lengths or batch size. Use custom for arbitrary CSV datasets rather than the fixed ETT split.
- **CUDA unavailable:** Select --device cpu or install a compatible PyTorch GPU build.
- **Out of memory:** Reduce --batch-size, especially for datasets with many variables.
- **Invalid data:** Ensure timestamps are ordered and unique, feature columns are numeric, and all values are finite. Align sampling intervals before running.
- **Checkpoint mismatch:** Reuse the original dataset, Run ID, and model configuration.

## Repository structure

| Path | Purpose |
| --- | --- |
| train.py | Dataset presets, input validation, and training launcher |
| scripts/ | Dataset scripts and the complete benchmark runner |
| run.py | Model configuration and training execution |
| models/ms_fusion.py | Scale-specific predictors and recursive fusion |
| layers/ | Embedding and normalization layers |
| exp/ | Three-stage training, checkpoint selection, and evaluation |
| data_provider/ | Dataset loading, splits, and sliding windows |
| utils/ | Forecasting metrics, time features, and plotting |
| requirements.txt | Runtime dependencies |

The training and checkpoint-reloading workflow has been exercised on real ETTh1 data using CPU execution. Full accuracy reproduction across all datasets and horizons, GPU execution, and installation in a clean environment remain unverified.
