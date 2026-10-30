# MSFusion：快速开始

MSFusion 通过尺度对齐监督和由粗到细的预测融合完成长期多变量时间序列预测。训练依次执行尺度预测器训练、冻结预测器的融合训练、联合微调三个阶段。

**安装依赖后，把数据放进 `dataset/`，即可执行 `python train.py`。** 默认运行 ETTh1，自动识别通道数并完成训练、验证选模和测试。选择其他数据集或接入自己的CSV也无需修改模型代码。

2026-10-30 已用真实 ETTh1 数据完成三阶段短训练和检查点重载测试。这确认了代码能运行，尚不代表复现了论文表格的精度。

## 1. 安装环境

建议 Python 3.10–3.12。以下命令在解压后的 `MSFusion_paper` 文件夹中执行。

```bash
python -m venv .venv
```

Linux/macOS 激活：

```bash
source .venv/bin/activate
```

Windows PowerShell 激活：

```powershell
.\.venv\Scripts\Activate.ps1
```

安装依赖：

```bash
python -m pip install -r requirements.txt
```

主要依赖为 PyTorch、NumPy、pandas、scikit-learn、Matplotlib。运行不需要原服务器路径、预训练权重、其他项目或 `sktime`。
CPU 可以完成整个流程。使用 GPU 时，应事先安装适合本机驱动的 PyTorch；`--device cuda` 会检查 CUDA 是否可用。
实际通过验证的依赖版本见 `requirements-tested.txt`。该版本列表来自已有环境，尚未在全新虚拟环境中逐项重装验证。

## 2. 放入数据

代码包不含数据。把已有数据复制到以下位置，也可以用 `--file` 指定其他位置。只需准备当前要运行的数据集。

```text
MSFusion_paper/
├── train.py
├── run.py
├── requirements.txt
├── dataset/
│   ├── ETT-small/
│   │   ├── ETTh1.csv
│   │   ├── ETTh2.csv
│   │   ├── ETTm1.csv
│   │   └── ETTm2.csv
│   ├── weather/weather.csv
│   ├── electricity/electricity.csv
│   ├── traffic/traffic.csv
│   └── solar/solar_AL.txt
├── models/
├── layers/
├── data_provider/
└── exp/
```

| `--dataset` | 默认文件（相对 dataset） | 格式 | 预设频率 |
| --- | --- | --- | --- |
| ETTh1 / ETTh2 | ETT-small/ETTh1.csv / ETTh2.csv | 带 date 列的 CSV | 1小时 |
| ETTm1 / ETTm2 | ETT-small/ETTm1.csv / ETTm2.csv | 带 date 列的 CSV | 15分钟 |
| Weather | weather/weather.csv | 带 date 列的 CSV | 10分钟 |
| Electricity | electricity/electricity.csv | 带 date 列的 CSV | 1小时 |
| Traffic | traffic/traffic.csv | 带 date 列的 CSV | 1小时 |
| Solar | solar/solar_AL.txt | 无表头、逗号分隔数值矩阵 | 10分钟 |
| custom | 用 `--file` 指定 | 带 date 列的 CSV | 用 `--freq` 指定 |

CSV 第一列必须名为 `date`，其余列均为待预测数值变量：

```csv
date,sensor_1,sensor_2,OT
2020-01-01 00:00:00,1.2,3.4,5.6
2020-01-01 01:00:00,1.3,3.5,5.8
```

日期应严格递增、无重复，数据应提前按固定频率对齐，不含缺失值和无穷值。程序检查数值和时间顺序，但不会自动补齐时间缺口或插值。

ETT 使用固定12/4/4个月基准划分，ETTh至少14400行，ETTm至少57600行。其他CSV与Solar按时间顺序划分70%训练、10%验证、20%测试。归一化统计仅在训练段拟合。验证和测试可使用前一段末尾历史作为输入，预测目标仍位于对应划分中。

## 3. 检查并开始运行

首次使用先检查文件和依赖：

```bash
python train.py --check-only
```

看到 `PASS: dataset and training imports` 后，执行短训练。`--smoke` 在完整数据划分上每阶段训练1轮，共3轮，然后评价全部测试窗口：

```bash
python train.py --smoke --device cpu --batch-size 128 --run-id first_smoke
```

正式训练 ETTh1，默认预测未来96步：

```bash
python train.py
```

其他数据集和预测长度：

```bash
python train.py --dataset Weather --pred-len 192 --device cuda --gpu 0 --run-id weather_192
python train.py --dataset ETTm2 --pred-len 720 --device cpu --run-id ettm2_720
```

数据位于其他位置时：

```bash
python train.py --dataset ETTh1 --file /path/to/ETTh1.csv
python train.py --dataset Weather --dataset-dir /path/to/dataset
```

Windows路径含空格时加引号，例如 `--file "D:/Time Series/ETTh1.csv"`。

## 4. 使用自己的数据

将数据转换为前述CSV格式，指定文件和实际采样频率即可。通道数自动推断，所有数值列都会被预测。

```bash
python train.py --dataset custom --file ./dataset/my_data.csv --freq h --run-id my_first_run
```

可先加 `--check-only`，再用 `--smoke --device cpu` 检查训练。数据较短时减小 `--seq-len`、`--pred-len` 或 `--batch-size`。训练段须容纳完整输入加预测窗口；验证、测试目标段各须至少有一个预测长度。

## 5. 常用参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--dataset` | ETTh1 | 数据集预设 |
| `--seq-len` | 336 | 输入历史长度 |
| `--pred-len` | 96 | 可选96、192、336、720 |
| `--epochs` | 100 | 三阶段预算33/33/34轮，验证早停 |
| `--batch-size` | 16 | 小显存时调低 |
| `--device` | auto | auto、cpu、cuda |
| `--gpu` | 0 | 当前进程可见GPU编号 |
| `--run-id` | 数据集名加时间戳 | 不同训练使用不同ID |
| `--smoke` | 关闭 | 三阶段各1轮，仅确认流程 |

预设使用因子2平均池化、两层降采样、AdamW、学习率0.001、随机种子2021、联合辅助损失系数0.1、预测器维度16。阶段早停耐心值为5/3/3。未由论文明确指定的参数使用整理版默认值。

更详细的模型参数见 `python run.py --help`。`train.py` 提供便捷预设，`run.py` 提供底层控制。本包支持长期多变量预测；不提供分类、异常检测等其他任务入口。

## 6. 查看结果与重新评价

运行会打印 Run ID。输出位于代码根目录：

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

`config.json` 保存实际运行参数；每个 `metrics.json` 保存 MSE、MAE，越小越好。指标在训练段统计标准化后的尺度上计算。Stage I评价原分辨率独立预测，Stage II/III评价融合预测。最终模型使用Stage III检查点。

用相同的数据、窗口和Run ID重新加载检查点评价：

```bash
python train.py --evaluate --device cpu --batch-size 128 --run-id first_smoke
python train.py --dataset Weather --pred-len 192 --evaluate --run-id weather_192
```

重新评价会更新对应指标文件。新训练应使用新的Run ID，避免覆盖旧权重。默认不生成图，底层 `run.py --plot` 可保存样例预测PDF。

## 7. 常见问题

- **Dataset not found**：核对文件名和目录，或用 `--file` 指定完整路径。
- **划分太短/没有有效窗口**：检查数据是否完整，减小窗口或batch size；任意短CSV请使用custom，不能套用ETT固定划分。
- **CUDA不可用/驱动不匹配**：先用 `--device cpu` 检查流程，再配置本机GPU环境。
- **显存不足**：减小batch size，多变量数据开销随通道数增加。
- **日期或数值格式错误**：首列须为date，其余均为有限数值；提前处理缺失、重复和不规则时间间隔。
- **检查点找不到/参数结构不匹配**：评价时沿用原数据集、Run ID、输入长度和预测长度。

## 8. 验证与源码导航

已验证真实ETTh1完整划分上的三阶段短训练、三个最佳检查点保存及独立重载评价。另用合成数据验证custom CSV和Solar格式的三阶段流程。详细环境、命令与结果见 [验证记录](verification/REPORT.md)。尚未对八个真实数据集及全部预测长度完成全量复现。

| 文件 | 用途 |
| --- | --- |
| train.py | 数据预设、通道识别、输入检查、便捷启动 |
| run.py | 详细模型与训练参数 |
| models/ms_fusion.py | 尺度预测与递归融合 |
| exp/exp_long_term_forecasting.py | 三阶段训练、验证、检查点与测试 |
| data_provider/ | 数据划分、标准化、滑窗 |
| smoke_model.py | 无需数据的模型前向/反向检查 |
| verification/check_loaders.py | 合成CSV/Solar三阶段集成检查 |
| docs/SOURCE_AUDIT.md | 原始源码来源及方法修正记录 |

源码整理自 `/home/yhe/MS_fusion/`。历史修正包括融合前辅助监督、训练/验证损失一致性和阶段正确评价；这些变化可能影响旧实验数值。`SOURCE_MANIFEST.json` 记录原始源码哈希，不能作为当前整理版文件哈希使用。
