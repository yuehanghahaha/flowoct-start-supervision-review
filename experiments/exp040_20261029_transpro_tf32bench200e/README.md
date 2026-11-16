# exp040：把原A100版exp028只调整TF32/卷积设置，回到A100对照测速

目录已包含独立源码、两份教师权重、配置、测速工具及验证记录。**请上传整个本目录到A100的`~/experience/`，与exp028并列；不要覆盖原目录。** 按项目规则直接同步目录，不默认制作ZIP/tar。

本版仅把cuDNN TF32改True、benchmark改True、deterministic改False；matmul TF32仍False、AMP仍关闭。exp028原有训练/验证/保存方式保持不变。逐项清单见[CHANGES.md](CHANGES.md)。

## 先测开关带来的速度差异

激活你之前已经安装好依赖的`lyh_transpro`环境：

```bash
conda activate lyh_transpro
cd ~/experience/exp040_20261029_transpro_tf32bench200e
bash BENCHMARK.sh
```

此命令**不需要prepare**：不读数据集，依次测旧设置、仅TF32、仅算法选择、两者组合。每组都是原64通道、256立方、同一随机种子及合成输入，预热2次后测5次完整G/D更新；四组依次运行，不同时占用GPU。首组/首步可能较慢，首次算法测速和优化器初始化不计入稳定步耗时。

结果目录：

```text
verification/speed_benchmarks/时间戳/
  plan.json
  baseline.json
  tf32_only.json
  algorithms_only.json
  combined.json
  summary.csv
  summary.json
```

`summary.csv`中`mean_seconds`是每次G/D更新的平均秒数，`speedup_vs_baseline`是与同次旧设置相比的加速倍数，例如4表示该组单步平均耗时为旧设置的四分之一。各步原始耗时、预热耗时、实际TF32/cuDNN设置和显存都另存JSON。

默认测试不包含数据加载、验证八项指标和保存文件。它回答TF32/算法设置如何影响**训练更新速度**，不能直接当作整轮加速倍数。

仅测旧设置与组合设置可运行：

```bash
bash BENCHMARK.sh --profiles baseline combined
```

需要更多稳定样本可运行`bash BENCHMARK.sh --steps 10`。选择其他已分配可见GPU用`--device cuda:1`；默认cuda:0，保持平台CUDA_VISIBLE_DEVICES。为便于比较，请在GPU没有其他训练任务占用时测。终端断开会中断测速，可将这些命令放入tmux会话。

## 如需按完整轮次对比

四组测速仅用临时合成模型，不产生正式训练断点。若还要运行本版完整200轮，先绑定原A100缓存并验收：

```bash
python -B -u run.py prepare
python -B -u run.py check
```

沿用exp028已成功的自动路径定位，不重建大型缓存，也不照抄当前不可见的`/home/...`数据路径。check成功后，在新的tmux会话中执行：

```bash
bash START.sh
```

本版从同一初始种子重新训练200轮；原尺寸、原损失、每轮4窗口完整验证、8项指标及每轮可续训latest均保留，后续空间预算仍约19.5 GiB。不能把测速命令的5次更新当作缩短正式训练预算，也不要把exp028的checkpoints/binding或验收记录复制来绕过本版检查。

本版有意保留较重的exp028验证和梯度扫描，因此即使训练单步显著提速，整轮速度仍可能与A800上exp035不同。

## 查看与回传

实时测速日志：

```bash
tail -n 30 -F ~/experience/exp040_20261029_transpro_tf32bench200e/logs/benchmark_console.log
```

测试完成后，把对应时间戳目录中的`summary.csv`、`summary.json`和四组JSON同步回来，即可定量区分两项设置的影响。没有自动上传结果或修改正在运行的exp035。

本机只有6GB GPU，本地验证以CPU合成及代码完整性为限。**本助手未上传本版、未执行本版A100测速、未启动本版正式训练。** 实际测试结果以你在A100运行后生成的记录为准。

本地17项检查通过：10项继承数值/流程、4项留存恢复、3项开关范围/源码差异/加速倍数有效性检查。运行内核`network.py`等6个受保护模块以及65份官方文件、教师权重保持父版指纹，细节见`verification/delivery.json`。
