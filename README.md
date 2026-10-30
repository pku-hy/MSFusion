# MSFusion

MSFusion performs long-term multivariate time series forecasting using scale-aligned supervision and coarse-to-fine prediction fusion. Training proceeds through three stages: scale-specific predictor training, fusion training with frozen predictors, and joint fine-tuning.

**Install the dependencies, place your data in `dataset/`, and run `python train.py`.** The default configuration runs ETTh1, detects the number of channels, and completes training, validation-based checkpoint selection, and testing. Other supported datasets and custom CSV files require no model code changes.

The complete three-stage workflow and checkpoint reloading were verified on real ETTh1 data on October 30, 2026. These short runs verify execution; they do not establish reproduction of the paper's reported accuracy.

## 1. Set up the environment

Python 3.10-3.12 is recommended. Clone the repository and enter its root directory:

```bash
git clone https://github.com/pku-hy/MSFusion.git
cd MSFusion
python -m venv .venv
```

Activate the environment on Linux or macOS:

```bash
source .venv/bin/activate
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The main dependencies are PyTorch, NumPy, pandas, scikit-learn, and Matplotlib. Running this package does not require the original server, pretrained weights, another project, or `sktime`.

The entire workflow can run on CPU. For GPU execution, install a PyTorch build compatible with your machine and driver. Selecting `--device cuda` checks CUDA availability.

The versions used for the integration checks are listed in `requirements-tested.txt`. They were tested in an existing environment; installation in a completely new virtual environment has not yet been verified.

## 2. Prepare the data

Datasets are not included. Place the files in the following locations, or use `--file` to specify another location. Only the dataset needed for the current run is required.

```text
MSFusion/
|-- train.py
|-- run.py
|-- requirements.txt
|-- dataset/
|   |-- ETT-small/
|   |   |-- ETTh1.csv
|   |   |-- ETTh2.csv
|   |   |-- ETTm1.csv
|   |   `-- ETTm2.csv
|   |-- weather/weather.csv
|   |-- electricity/electricity.csv
|   |-- traffic/traffic.csv
|   `-- solar/solar_AL.txt
|-- models/
|-- layers/
|-- data_provider/
`-- exp/
```

| `--dataset` | Default path relative to dataset/ | Format | Default frequency |
| --- | --- | --- | --- |
| ETTh1 / ETTh2 | ETT-small/ETTh1.csv / ETTh2.csv | CSV with a date column | 1 hour |
| ETTm1 / ETTm2 | ETT-small/ETTm1.csv / ETTm2.csv | CSV with a date column | 15 minutes |
| Weather | weather/weather.csv | CSV with a date column | 10 minutes |
| Electricity | electricity/electricity.csv | CSV with a date column | 1 hour |
| Traffic | traffic/traffic.csv | CSV with a date column | 1 hour |
| Solar | solar/solar_AL.txt | Headerless, comma-separated numeric matrix | 10 minutes |
| custom | Specify with --file | CSV with a date column | Specify with --freq |

For CSV files, the first column must be named `date`. All remaining columns must contain the numeric variables to forecast:

```csv
date,sensor_1,sensor_2,OT
2020-01-01 00:00:00,1.2,3.4,5.6
2020-01-01 01:00:00,1.3,3.5,5.8
```

Timestamps must be strictly increasing and unique. Align observations to a regular sampling interval before running the model, and resolve missing or non-finite values. The launcher checks numeric values and timestamp order; it does not fill time gaps or interpolate observations.

ETT uses fixed benchmark splits of 12/4/4 months. ETTh requires at least 14,400 rows, and ETTm requires at least 57,600 rows. Other CSV datasets and Solar use chronological splits of 70% training, 10% validation, and 20% testing. Standardization statistics are fitted only on the training split. Validation and test inputs may use the preceding split's trailing history, while their forecast targets remain within the corresponding split.

## 3. Check the setup and start a run

Check the dataset and training imports first:

```bash
python train.py --check-only
```

After `PASS: dataset and training imports`, run a short integration check. The `--smoke` option trains for one epoch per stage on the complete data splits, then evaluates all test windows:

```bash
python train.py --smoke --device cpu --batch-size 128 --run-id first_smoke
```

Start a regular ETTh1 run with a 96-step forecast horizon:

```bash
python train.py
```

Select another dataset, forecast horizon, or device:

```bash
python train.py --dataset Weather --pred-len 192 --device cuda --gpu 0 --run-id weather_192
python train.py --dataset ETTm2 --pred-len 720 --device cpu --run-id ettm2_720
```

Use data stored elsewhere:

```bash
python train.py --dataset ETTh1 --file /path/to/ETTh1.csv
python train.py --dataset Weather --dataset-dir /path/to/dataset
```

Quote paths containing spaces on Windows, for example `--file "D:/Time Series/ETTh1.csv"`.

## 4. Use your own dataset

Convert your data to the CSV format above and specify its file and actual sampling frequency. The launcher infers the channel count and forecasts every numeric column.

```bash
python train.py --dataset custom --file ./dataset/my_data.csv --freq h --run-id my_first_run
```

Add `--check-only` to validate the input, then use `--smoke --device cpu` to check training. For shorter datasets, reduce `--seq-len`, `--pred-len`, or `--batch-size`. The training split must accommodate the input and forecast windows; each validation and test target split must contain at least one forecast horizon.

## 5. Common options

| Option | Default | Description |
| --- | --- | --- |
| --dataset | ETTh1 | Dataset preset |
| --seq-len | 336 | Historical input length |
| --pred-len | 96 | One of 96, 192, 336, or 720 |
| --epochs | 100 | Stage budgets of 33/33/34 epochs, with validation-based early stopping |
| --batch-size | 16 | Reduce this for limited memory |
| --device | auto | auto, cpu, or cuda |
| --gpu | 0 | GPU index visible to the current process |
| --run-id | Dataset name and timestamp | Use a distinct ID for each training run |
| --smoke | Disabled | One epoch per stage to check execution |

The preset uses factor-two average pooling, two downsampling layers, AdamW, a learning rate of 0.001, random seed 2021, an auxiliary loss coefficient of 0.1, and predictor dimension 16. Early-stopping patience is 5/3/3 for the three stages. Settings not explicitly specified in the paper use the defaults of this source package.

For detailed model options, run `python run.py --help`. The `train.py` launcher provides convenient presets, while `run.py` exposes lower-level controls. This package supports long-term multivariate forecasting.

## 6. Inspect results and reload checkpoints

Each run prints its Run ID. Outputs are written relative to the repository root:

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

The `config.json` file records the actual run arguments. Each `metrics.json` file contains MSE and MAE; lower values are better. Metrics are computed in the standardized value space using training-split statistics. Stage I evaluates the independent prediction at the original resolution; Stages II and III evaluate the fused prediction. Use the Stage III checkpoint as the final model.

Reload checkpoints for evaluation using the same dataset, window lengths, and Run ID:

```bash
python train.py --evaluate --device cpu --batch-size 128 --run-id first_smoke
python train.py --dataset Weather --pred-len 192 --evaluate --run-id weather_192
```

Evaluation updates the corresponding metrics files. Use a new Run ID for each new training run to avoid overwriting checkpoints. Plots are disabled by default; the lower-level `run.py --plot` option saves sample prediction PDFs.

## 7. Troubleshooting

- **Dataset not found:** Check the filename and directory, or provide the full path with `--file`.
- **Splits are too short or contain no valid windows:** Check dataset completeness and reduce the window lengths or batch size. Use `custom` for arbitrary short CSV files; ETT uses fixed benchmark splits.
- **CUDA is unavailable or the driver is incompatible:** Check the workflow with `--device cpu`, then configure a compatible GPU environment.
- **GPU memory is exhausted:** Reduce the batch size. Memory usage grows with the number of variables.
- **Invalid timestamps or values:** Ensure the first column is date and all other columns contain finite numeric values. Resolve missing values, duplicate timestamps, and irregular sampling first.
- **Missing or incompatible checkpoints:** Reuse the original dataset, Run ID, input length, and forecast horizon for evaluation.

## 8. Validation and source layout

The complete ETTh1 splits were used to verify one training epoch per stage, validation-based checkpoint selection, all three checkpoint files, and evaluation after independent checkpoint reloading. Synthetic data were also used to check the custom CSV and Solar workflows. See the [validation report](verification/REPORT.md) for commands, environment details, and results. Full accuracy reproduction across all eight real datasets and forecast horizons remains to be completed.

| File or directory | Purpose |
| --- | --- |
| train.py | Dataset presets, channel inference, input checks, and convenient execution |
| run.py | Detailed model and training arguments |
| models/ms_fusion.py | Scale-specific predictions and recursive fusion |
| exp/exp_long_term_forecasting.py | Three-stage training, validation, checkpoints, and testing |
| data_provider/ | Data splits, standardization, and sliding windows |
| smoke_model.py | Model forward and backward checks without a dataset |
| verification/check_loaders.py | Synthetic CSV and Solar integration checks |
| docs/SOURCE_AUDIT.md | Source provenance and method-alignment changes |

This package was prepared from `/home/yhe/MS_fusion/`. Changes include pre-fusion auxiliary supervision, consistent training and validation losses, and correct stage-specific evaluation. These changes can affect historical results. The `SOURCE_MANIFEST.json` file records hashes of the original source snapshot, rather than hashes of the current files.
