# MSFusion paper source package

> 本文件为第一轮源码核对历史记录。最新安装、支持范围、运行命令和验证状态以根目录README及verification/REPORT.md为准。后续已精简为论文长期预测加载器并增加train.py入口。

整理日期：2026-10-30。来源：kangbo (`10.129.161.221`)，`/home/yhe/MS_fusion/`。
原目录未修改。原始代码另存本地 `source_review/`，归档为 `source_review/msfusion_source_review_20261030.tar.gz`。
`SOURCE_MANIFEST.json` 记录原始 Python/shell 文件 SHA256；整理后文件不应与原始哈希混淆。

## 范围与状态

对应论文 MSFusion: Scale-Aligned Supervision and Coarse-to-Fine Prediction Fusion for Long-Term Time Series Forecasting。
保留原预测器、RevIN、数据加载与训练框架；修正论文公式与代码不一致处。
本包是待验证的论文方法整理版，不是已经复现论文 Table II/V 数值的冻结发布版。
未打包数据、权重、实验结果或第三方基线，不自动启动全量训练。

## 安装与运行

建议独立 Python 3.10+ 环境，按硬件安装合适 PyTorch，再 `pip install -r requirements.txt`。
依赖尚未冻结版本，最终需以实际通过端到端训练的环境导出锁定版本。
从本目录执行，数据从原服务器已有数据集复制或只读链接到 `dataset/`。

```bash
python run.py --task_name long_term_forecast --is_training 1 --model_id ETTh1_paper_96 --model ms_fusion --data ETTh1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --features M --seq_len 336 --label_len 0 --pred_len 96 --c_in 7 --c_out 7 --down_sampling_layers 2 --down_sampling_window 2 --down_sampling_method avg --learning_rate 0.001 --scale_reg_weight 0.1 --num_workers 0
```

其余预测长度：192、336、720。四个 ETT、Weather、Electricity、Traffic、Solar 的加载器保留；数据路径与通道数须按实际数据配置。
当前仅支持 M 多变量同维预测、平均池化；CLI 会拒绝其他设置。
论文没有明确指定的超参数使用历史代码默认值，不能视作历史结果对应配置。
默认总训练预算100轮，按三阶段拆分，验证早停；默认 lambda=0.1 是原代码值，不代表论文已证明其唯一取值。
训练使用验证损失选模，测试结果分 stage 保存于 `test_results/<setting>/stageN/metrics.json`。
历史 shell 脚本仅保留于原始快照，包含删除输出与硬编码多GPU行为，不纳入整理版运行入口。

## 论文与源码对应及修正

| 论文 | 位置 | 整理说明 |
| --- | --- | --- |
| 输入/目标同算子降采样 | models/ms_fusion.py 与 exp/exp_long_term_forecasting.py | 保留平均池化 |
| 每尺度预测和归一化 | models/ms_fusion.py | 保留历史参数命名与结构 |
| Eq.24 递归融合 | MoE_fusion_Block | 原名称为历史命名，实际为固定投影，无路由器 |
| Eq.27–29 辅助损失 | Model.forward、vali、train_stagewise | 返回融合前列表；训练和验证统一2^-s权重、同一lambda |
| Stage I/II 冻结 | train_stagewise | Stage I 包括归一化参数；Stage II 仅训练 moe_fusion |
| Table V阶段评价 | test | 移除重复定义，按当前stage前向；分stage输出指标 |
| 论文默认配置 | run.py | 输入336、两层、因子2、学习率0.001 |

原代码Stage III对已融合列表施加辅助损失，验证却用无权重且默认系数0的损失；这会影响选模。
原Stage I未解冻归一化参数；整理版纳入训练，进一步意味着结果可能与旧实验不同。
原训练逐轮打印测试损失；整理版取消，测试在训练结束后执行。
修正不会追溯改变旧权重或旧实验结果。

## 待完成的复现证据

1. 完整环境依赖检查、真实数据三阶段短训练和各阶段检查点评价。
2. 核对32组历史日志/检查点与论文表格来源，确认论文实际使用哪套配置与版本。
3. 冻结运行配置、随机种子、环境版本与数据哈希，再进行全量复现。
4. 核对第三方继承代码许可证后再公开发布；本次不推断或新增授权。

## 本次验证
YJY已有Anaconda环境：smoke_model.py通过尺度形状、融合前输出、递归融合和CPU反向传播检查；run.py --help通过。未进行真实数据训练或GPU验证。
