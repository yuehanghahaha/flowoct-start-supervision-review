# exp028 → exp040：只比较TF32和卷积算法设置

用户希望把此前A100版本修改后放回同一A100测试。本版基线是用户当时运行的`exp028_20261028_transpro_retention200e`，不是当前A800上的exp035。旧exp028及正在运行的exp035均不修改。

## 正式训练代码唯一执行改动

只修改`experiment.py`中的`precision()`及`runtime()`：前者配置以下开关，后者补记实际运行标志。详细差异保存在`provenance/backend_runtime.patch`。

| 项目 | exp028 | exp040默认 |
| --- | --- | --- |
| 张量及优化器状态 | float32 | float32 |
| AMP/梯度缩放 | 关闭 | 关闭 |
| 一般matmul TF32 | False | False |
| cuDNN卷积TF32 | False | **True** |
| cuDNN benchmark | False | **True** |
| cuDNN deterministic | True | **False** |

这与exp035已经记录的计算开关一致。TF32降低部分乘法的有效精度，因此本版不能称为“严格关闭TF32的FP32”。卷积算法选择可能改变浮点累加顺序，不承诺与exp028逐位一致。用户本次明确要求测试这两项，作为项目严格FP32默认规则的本实验例外。

## 保持与exp028逐字节相同的训练/评价实现

`run.py`、`network.py`、`dataset.py`、`metrics.py`、`evaluation.py`、`retention.py`、全部65份官方source文件、两份教师权重、固定患者成员表保持相同。配置只增加实验身份、上述开关和精度说明。

因此保留exp028原有：200轮、140训练患者/10验证/50测试、batch1、256立方、64通道、原损失及系数、教师eval、原LR曲线、种子2026、串行读缓存、每步梯度检查、每轮4窗口完整验证及8项指标、每轮latest及双槽best留存。没有为本次测速改成exp035的4worker/单裁块MAE/每5轮续训断点，也没有减少分辨率或训练预算。

父实验中的论文/官方代码差异仍存在，原协议快照见`provenance/parent_PROTOCOL.md`。本版用途是比较速度，不重新宣称exp028已等同作者实现。

## 新增测速工具

`benchmark_speed.py`及`BENCHMARK.sh`：原宽度、256立方、合成输入，依次启动四个独立子进程。每个进程重新用种子2026初始化相同模型、教师、Adam及输入，默认预热2次再测5次完整G/D更新。

| 组名 | cuDNN TF32 | benchmark | deterministic |
| --- | --- | --- | --- |
| baseline | False | False | True |
| tf32_only | True | False | True |
| algorithms_only | False | True | False |
| combined | True | True | False |

一般matmul TF32在所有组都为False，AMP在所有组都关闭。算法选择组同时改变benchmark/deterministic，与本次讨论的策略一致，不能从该组单独区分这两个开关的贡献。

每次`model.step`前后同步GPU，计时包括exp028原有的梯度扫描和优化器更新；不包含模型装载、预热、数据加载、整轮验证或断点写盘。记录各步、均值、中位数、显存和实际设置。`baseline均值/当前组均值`才是该次同卡的测速倍数；baseline失败时不输出倍数。它不是整轮加速比或临床准确率对照，也不据此选择模型或患者划分。

测试只写`logs/`与`verification/speed_benchmarks/`，不读取患者数据、不改旧断点、不生成正式训练权重；子进程退出后释放模型与三维合成数组。日志与小JSON/CSV作为测速证据保留。不下载服务器结果，不启动远程训练。

## 核查边界

本地仅做CPU合成、原始代码完整性、参数/梯度及输出流程检查；没有在本机6GB GPU执行原宽度更新，不把本地检查称为A100验收。速度以用户上传后在A100生成的`summary.csv`和原始逐步时间为准。旧源码、配置和正式证据保留。
