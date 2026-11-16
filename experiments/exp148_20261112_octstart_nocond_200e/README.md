# exp148 OCT 起点单通道流匹配

用固定验证集检验移除额外 OCT 输入对 exp017 的浅层血管投影和整卷重建的影响。训练从同一种子随机初始化，裁块内以 OCT 为 `x0`、配对 OCTA 为 `x1`，均匀采样 `t ∈ (0,1)`。将 `x_t = (1−t)x0 + t x1` 与 `t` 输入 3D U-Net，监督为所有体素上的 `MSE(v(x_t,t), x1−x0)`。推理从整卷 OCT 开始，每一步输入当前状态及时间。

| 项目 | exp017 | exp148 |
| --- | --- | --- |
| 起点和速度目标 | OCT、OCTA−OCT | 同一配对路径 |
| 速度网络 | `v(x_t,t,OCT)`，2,097,521 参数 | `v(x_t,t)`，2,097,089 参数 |
| 主干和初始化 | 四尺度 base16，种子 2026 | 同主干。沿用同种子初始化的第一通道卷积核及其余参数 |
| 训练监督 | 均匀时间采样，速度 MSE | 同一速度 MSE；额外 L1、投影损失、精确 t=0 更新均为 0 |
| 裁块和优化 | ZYX 256×64×64、batch 1、AdamW 2e−4、weight decay 0、梯度范数上限 1 | 相同，保留原病例顺序及每轮病例、裁块、时间采样规则 |
| 训练预算 | 150 轮 × 200 步 | 200 轮 × 200 步 |
| CUDA 数值 | FP32、TF32 关闭 | BF16 模型 autocast、TF32 允许、cuDNN benchmark 开启 |
| 验证 | 固定 10 人、每 5 轮、Euler 8 步 | 相同，FP32 整卷状态及 Hann 速度融合在 GPU 上完成 |
| 主选模 | ILM–OPL 最大投影 SSIM | 相同，同分保留先达到的轮次 |

模型参数、优化器状态、输入数据、速度 MSE 和 Euler 状态用 float32。模型前向允许 BF16 autocast，矩阵乘法和 cuDNN 的 TF32 标志分别记录。梯度缩放关闭，cuDNN 不强制确定性，PyTorch 确定性算法约束关闭。缓存用既有 Z256 float32 NPY mmap，固定 /255 归一化。首次读取时核验对应源 NPZ 和实际使用的 NPY 哈希。

同一 A800、训练病例 10301、同一完整裁块交错计时。模型前向、反向和 AdamW 更新的中位耗时为 FP32 允许 TF32 82.1 ms、BF16 autocast 73.6 ms。两档的损失与梯度均有限，初始速度相对 L2 差异为 2.20%。测速样本在 GPU 上，正式用时还受共享算力和数据读取影响。完整记录见 `verification/precision_benchmark.json`。

验证使用 ZYX 256×304×304、50% 重叠的正 Hann 权重，全卷速度融合后再更新状态，最后裁剪到 [0,1]。逐患者记录完整 3D、304 张 [Z,X] B-scan、ILM–OPL 最大投影、OPL–BM 最大投影、全深度均值及最大投影、三个层间均值投影的 MAE、PSNR、SSIM。标准指标用原始浮点数据和 range 1，SSIM 使用 win7 与样本协方差，患者等权汇总。参考 GT 范围和区域误差单列。浅层投影用标注 native Z640 层界映射到 Z256，标注仅用于验证计算。

首要比较采用两组各自原规则选出的验证最优轮次，读取同轮全部指标与投影。exp017 使用 E5–E150，exp148 使用 E5–E200。可用 E5–E150 的指标历史做等窗口补充对照。历史比较同时包含精度、后端和训练预算差异，不能单独识别条件通道的因果效应。

## 服务器运行

`deployment.json` 已绑定当前服务器固定 140 训练、10 验证、50 测试及已核验缓存。当前任务范围为训练和验证，入口仅加载训练及验证影像。独立会话内用同一条命令启动或续跑。

```bash
cd /docsystem/niuke/lvyuehang/Flow_oct2octa/experience/exp148_20261112_octstart_nocond_200e
bash launch.sh
tail -f logs/train_console.log
```

`Ctrl+C` 可退出上面的日志查看。需要停止实验时，在 `tmux attach -t exp148_octstart_nocond` 进入训练会话后按 `Ctrl+C`。启动入口默认使用服务器现有 PyTorch 2.7.0+cu126 环境，可在新部署前通过 `EXP148_PYTHON` 指定等价环境。训练中断从最近提交轮次恢复。已写入待验证断点的完整训练轮次会直接续跑剩余验证病例。其余未提交的训练轮次重新跑该轮。普通训练时保留 `checkpoints/latest.pt` 与 `checkpoints/best.pt`，提交后的临时文件清理。

## 验证证据和回传

- `verification/` 包含单通道、速度目标、真实裁块测速、真实训练整卷和启动记录。
- `runs/train/history.json` 记录每轮训练及所有验证指标，`training_complete.json` 写入时表示 200 轮及验证全部提交。
- `reports/validation/` 包含每轮汇总及逐病例指标、原浮点二维投影、逐 B-scan 指标和固定病例图。整卷数组在计算完指标后释放。
- 分析时回传 `runs/train/history.json`、`runs/train/training_complete.json`、`reports/validation/` 和 `verification/` 即可。再推理需要保留对应权重、数据与环境。
