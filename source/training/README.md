# 实验 001：3DFM + 起点监督，100 轮

创建日期：2026-10-24。代码和结果均归属于本目录；正式输出为 `runs/train/`。这是从头训练的独立实验。

## 方法

沿用原体素域 3D U-Net 基线，输入当前状态、时间和原始 OCT 条件。设配对 OCT 为 `c`、OCTA 为 `y`：

```text
x_t = (1-t)c + t*y
u = y-c
L = MSE(v_theta(x_t, t, c), u)
```

每轮预先安排、随机打散 100 次精确 `t=0` 和 100 次 `t∈(0,1)` 的更新，优化期望目标 `0.5*L_start + 0.5*L_uniform_FM`。在起点，`L_start = MSE(c + v_theta(c,0,c), y)`，会直接训练网络从纯 OCT 迈出第一步。**起点仍是 OCT，本实验没有另加一个可学习的起点生成网络。** 每次更新只有一次网络前向和反向，AMP 溢出会重试相同数据及时间，不占成功更新预算。

这沿用此前单块起点诊断的设计。此前训练块上的改善不能推断本次完整数据实验的泛化效果；50% 是待验证的实验比例。

## 固定设置

| 项目 | 设置 |
| --- | --- |
| 数据 | OCTA500 common_v1，3M，原固定患者划分 140/10/50 |
| 空间与强度 | ZYX，Z=256，固定除以 255，配对随机裁块 |
| 体块 | 256×64×64，batch=1 |
| 主干 | 原 3D U-Net，基础通道 16，约 210 万参数 |
| 训练预算 | 100 轮，每轮 200 次成功更新，共 20,000 次 |
| 时间预算 | 10,000 次明确起点；10,000 次随机时间 |
| 优化 | AdamW，lr=2e-4，weight_decay=0，梯度裁剪 1，CUDA AMP |
| 验证 | 每 5 轮，10 个验证病例的完整预处理体，FP32，Euler8 |
| 推理 | 从 OCT 出发，50% 重叠滑窗，逐步融合速度，最终截断到 [0,1] |
| 最优模型 | 验证患者宏平均 ILM–OPL 投影 SSIM 最大值 |
| 保存 | 每轮 latest；每 10 轮完整断点及截至当时的累计最优，共 10 组 |

一个“轮次”指 200 次随机有放回病例裁块更新，不表示每个训练病例恰好遍历一次。模型与训练损失不含小波、额外投影、对比或重建分支；包内 `wavelets.py` 是复用模块的依赖，当前配置始终为 `voxel`。11 份层标注仅用于验证和固定训练块的诊断评价，不进入训练损失或生成输入。测试集不参与选轮。

## 服务器使用

上传位置：`/docsystem/niuke/lvyuehang/Flow_oct2octa/experience/exp001_20261024_3dfm_start_100e`，服务器为 SSH 配置中的 `eias-member.eitech.edu.cn`。

```bash
cd /docsystem/niuke/lvyuehang/Flow_oct2octa/experience/exp001_20261024_3dfm_start_100e
bash launch.sh check       # 数据、预算、正式体块的两次 GPU 更新、Euler8、保存重载
bash launch.sh train       # 从头开始 100 轮，已有输出会拒绝覆盖
bash launch.sh resume      # 从最新完整轮恢复
bash launch.sh status      # 查看已完成轮数、更新数及最佳轮
```

默认使用原 `lyh_fm3d` 环境和可见 GPU0。可用 `CUDA_VISIBLE_DEVICES=1 bash launch.sh check` 选择另一张卡；`PYTHON` 可指定解释器。正式入口要求 CUDA，不会自动转 CPU。

后台启动示例：

```bash
mkdir -p runs
nohup bash launch.sh train > runs/train_console.log 2>&1 < /dev/null &
tail -f runs/train_console.log
```

迁移到其他数据位置时，检查、首次训练及恢复均传入相同的 `--manifest /实际位置/split_3M_z256.csv`，或设置 `FM3D_MANIFEST`。科学参数在 `experiment.json`；修改已交付实验的科学设置时新建下一个编号目录。没有重划分或重新转换服务器数据。

100 轮全部完成后执行 `bash launch.sh test`，固定 `best.pt` 与 Euler8，在 50 个测试病例上输出 FP32 的体积 MAE、PSNR 以及预测 NPZ 到 `runs/test_best/`。该基础测试入口不计算分层 SSIM、LPIPS、FID 或血管拓扑；这些不能由本实验的验证投影指标替代。

## 文件与恢复

- `experiment.json`：本实验冻结配置。
- `fm3d/`：本地独立代码副本，不依赖另一个项目的 Python 路径。
- `tools/`：原数据去重分组的回归测试依赖；正式训练入口不会调用数据转换或重划分。
- `provenance.json`：代码来源、原文件指纹及本次改动。
- `verification/`：语法检查、功能测试、部署校验和 GPU 短程检查；这里的权重不属于正式训练。
- `runs/train/steps.jsonl`：每次成功更新的时间、分组、损失及梯度。
- `runs/train/history.jsonl`：每轮两类损失、诊断、验证指标与显存峰值。
- `runs/train/latest.pt`：最新完整轮的模型、优化器、AMP 和历史最优状态。
- `runs/train/best.pt`：截至当前的验证最优模型。
- `runs/train/epoch_0010.pt`、`best_through_0010.pt` 等：整十轮断点及累计最优快照。
- `runs/train/diagnostics/`：固定训练块与验证块的起点误差和真正从 OCT 出发的预测。

恢复以完整轮为边界；未完成轮的日志会回滚并重做该轮。更改时间比例、数据、源代码或评价协议会拒绝续训。

本次任务交付代码、上传和运行检查；完整 100 轮训练的启动状态以 `DEPLOYMENT.md` 和 `bash launch.sh status` 为准。
