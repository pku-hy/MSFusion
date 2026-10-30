# Source audit and provenance

Prepared on October 30, 2026, from `/home/yhe/MS_fusion/` on the original MSFusion server. The original project directory was preserved. The local source snapshot was archived as `source_review/msfusion_source_review_20261030.tar.gz` outside this repository. `SOURCE_MANIFEST.json` records SHA256 hashes of the original Python and shell files, not the current edited files.

## Scope

The package corresponds to *MSFusion: Scale-Aligned Supervision and Coarse-to-Fine Prediction Fusion for Long-Term Time Series Forecasting*. It retains the historical predictors, RevIN, data loading, and three-stage training framework while aligning auxiliary supervision and evaluation with the described method.

The repository contains source code, documentation, and short-run validation logs. Datasets, trained weights, and third-party baseline implementations are not included. Current setup instructions and validation status are maintained in the root README and `verification/REPORT.md`.

## Method alignment

| Paper component | Implementation | Change or retained behavior |
| --- | --- | --- |
| Matching input and target pyramids | models/ms_fusion.py and exp/exp_long_term_forecasting.py | Retain average pooling for both inputs and targets |
| Scale-specific prediction and normalization | models/ms_fusion.py | Retain historical parameter names and predictor structure |
| Recursive fusion, Eq. 24 | MoE_fusion_Block | Historical class name; the module uses learned projections without a routing mechanism |
| Auxiliary losses, Eqs. 27-29 | Model.forward, vali, and train_stagewise | Preserve pre-fusion predictions; use the same geometric weights and auxiliary coefficient in training and validation |
| Stage I/II freezing | train_stagewise | Include normalization parameters in Stage I; train only moe_fusion in Stage II |
| Stage-specific evaluation, Table V | test | Remove duplicate evaluation definitions, select the correct forward stage, and save separate stage metrics |
| Paper configuration | run.py and train.py | Default input length 336, two downsampling layers, factor two, and learning rate 0.001 |

The historical Stage III code applied auxiliary losses to the already fused prediction list. Its validation path used unweighted scale losses with a default coefficient of zero. The prepared version applies supervision to pre-fusion predictions and uses consistent weighted objectives for training and validation.

The historical Stage I parameter selection left normalization parameters frozen. They are included in Stage I training in this package. Per-epoch test-loss reporting was removed; test evaluation is performed after training. These changes can alter checkpoint selection and final results. They do not retroactively modify historical weights or recorded experiments.

## Running the package

Use `train.py` for dataset presets, automatic channel inference, and input validation. The lower-level `run.py` entrypoint supports multivariate M forecasting with matching input and output dimensions and average pooling. Hyperparameters not explicitly specified in the paper use the prepared package's defaults; they should not be assumed to match every historical result.

The default total budget is 100 epochs split across three stages, with validation-based early stopping. The default auxiliary coefficient of 0.1 comes from the historical implementation; it is not claimed to be uniquely optimal. Stage-specific test metrics are saved in `test_results/<setting>/stageN/metrics.json`.

Historical shell scripts remain in the original source snapshot. They include output deletion and hard-coded multi-GPU execution and are not provided as the current run interface. Unrelated classification and anomaly-detection data loaders were removed from the forecasting import path, eliminating the sktime dependency.

## Validation and remaining work

Model shape, pre-fusion output, recursive fusion, and CPU backward checks passed. Real ETTh1 three-stage short training and independent checkpoint evaluation also passed. Synthetic custom CSV and Solar integration checks passed. Details and logs are in `verification/REPORT.md`.

Remaining reproduction work includes checking the provenance of the 32 historical dataset-horizon results, identifying the configurations behind the paper tables, recording data hashes and a fully validated environment, and running the complete benchmark suite. GPU execution and installation in a clean environment remain unverified. Third-party code provenance and licenses should be checked before assigning a repository-wide license; this package does not infer or add such a license.
