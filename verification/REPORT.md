# 运行验证

日期：2026-10-30。执行者：Codex，YJY。
工作目录：`/home/sangibb/hy/msfusion/source_20261030/MSFusion_paper`。

## 环境

Linux，Python 3.12，PyTorch 2.5.1+cu121（CPU执行），NumPy 1.26.4、pandas 2.3.3、scikit-learn 1.7.2、Matplotlib 3.9.2。
使用既有Anaconda环境，未进行全新环境安装验证，未验证GPU训练。

## 真实数据端到端

数据来自原项目 `dataset/ETT-small/ETTh1.csv`：17420行、7通道，ETT固定划分8640/2880/2880。

```bash
python train.py --check-only
python train.py --smoke --device cpu --batch-size 128 --run-id quickstart_real_ETTh1
python train.py --evaluate --device cpu --batch-size 128 --run-id quickstart_real_ETTh1
```

已完成：全部训练窗口的三个阶段（各1轮）、验证选模、三个检查点保存、完整测试集2785窗口评价、独立进程重载评价。

| 阶段 | MSE | MAE |
| --- | --- | --- |
| I | 0.4075781107 | 0.4204682410 |
| II | 0.4111363888 | 0.4243333042 |
| III | 0.4084532857 | 0.4223066568 |

这是短训练流程验证，不能当作正式收敛结果或论文复现。日志见本目录 `smoke_training.log` 和 `smoke_evaluation.log`。

## 数据接入与模型检查

`python verification/check_loaders.py`：固定种子生成1200行3通道合成数据；custom CSV与无表头Solar各执行三阶段训练、保存检查点、测试。通过。合成数据不代表真实Weather/Electricity/Traffic/Solar精度验证。

`python smoke_model.py`：尺度输出长度、融合前输出保留、递归融合和反向梯度有限性检查通过。

当前限制：真实ETTm、其他真实数据集、GPU、全新依赖环境与正式全量精度尚未验证。
