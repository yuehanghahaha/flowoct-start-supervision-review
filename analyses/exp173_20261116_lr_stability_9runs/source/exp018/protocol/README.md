# 实验018：3DFM＋起点＋Z投影约束，200轮

这是从头独立训练的正式科研实验。固定配置为`experiment.json`，数学、参考指标、选模、数据坐标和恢复协议详见[PROTOCOL.md](PROTOCOL.md)。父实验为exp001。本次统一FP32，不接入旧AMP权重，不导入先验/残差大网络。

同批四组：exp016起点200轮；exp017均匀FM150轮；exp018起点＋全图Z投影200轮；exp019起点＋全图及非大血管区域额外投影200轮。每轮200次成功更新，seed2026，相同初始化与病例/裁块随机序列。

正式设置：原base16约210万参数，256×64×64体块，batch1，固定lr=2e-4、AdamW、weight_decay=0、clip1。AMP、GradScaler、TF32全部关闭。原固定140/10/50划分，固定/255、ZYX，完整预处理视野256×304×304。

第3、4组采用同一ILM–OPL沿Z最大投影，辅助项只作用于精确t=0的单步估计OCT+v，复用同次前向。第3组全图L1；第4组额外增加非大血管区域平均L1，没有三类等权，也没有额外边缘损失。它区别于历史全深度均值投影，不能混称完全相同。

## A100正式训练

```bash
conda activate lyh_flowoct
cd /workspace/experience/exp018_20261027_3dfm_start_zproj_200e
python -u run.py train
```

入口直接开始正式预算，没有先跑短测的门槛。它优先只读继承已部署exp014/013/011/004的实际数据manifest路径；不存在时默认`/workspace/flowoct_work/data/OCTA500_v1/split_3M_z256.csv`及其`labels_native/`。可在第一次运行时显式绑定：

```bash
python run.py bind --data-root /实际数据目录 --manifest /实际清单.csv --labels-root /实际数据目录/labels_native
python -u run.py train
```

`bind`只处理路径、固定分组和输入文件，不运行模型、显存短测或测试图像。中断后用`python -u run.py resume`；进度用`python run.py status`。已有正式输出不覆盖。

## 四组并行

四个目录一起放到`/workspace/experience/`，在任意一组目录执行：

```bash
python launch_four.py start
python launch_four.py status
```

启动器使用当前环境的Python和CUDA_VISIBLE_DEVICES，为四组创建独立tmux会话：`fm3d_exp016`至`fm3d_exp019`。已运行或已完成的组保留。可用`--only 1 3`只启动指定组，编号为本批1–4。

```bash
tmux attach -t fm3d_exp018
tail -f logs/formal_*.log
```

没有自动减小体块、轮数、精度或每任务20GB分片，也不停止已有其他任务。四开逻辑独立，但共享算力，不能保证更快；真实OOM将记录显存并停止相应任务。

## 完整输出与正式测试

- `checkpoints/`：initial、latest、best、best_reference_psnr，每10轮和累计best快照；四组均保留150轮前缀。
- `runs/train/`：逐步、逐轮和状态；`logs/`：控制台、启动和故障。
- `predictions/validation|test/`：**所有完整256×304×304的float32预测**。
- `reports/validation|test/`：3D MAE/PSNR/SSIM、单独命名的官方GT范围参考列、分层投影/PCC/区域诊断、完整病例图、CSV和HTML。
- `verification/`：源文件、输入、精度、环境和最终测试冻结记录；`fm3d/`是逐字节继承的父模块。

主best统一选择验证集ILM–OPL SSIM最大值；另存参考范围体积PSNR最佳。不能测试后跨指标挑best。完成预算后显式执行：

```bash
python run.py freeze --checkpoint best.pt
python -u run.py test
```

不自动跑测试，也不使用测试决定超参数。若预先安排等150轮比较，可明确冻结`best_through_0150.pt`。每次验证保留完整预测，不因资源而省略。恢复以latest完整轮为边界，未提交日志保留为中断证据。

## 当前交付状态

本地科研代码已完成；只做静态语法、继承指纹和配置核对，没有运行短程替代训练。A100的SSH登录返回认证失败，本次尚未成功上传，也尚未启动这四组。四组直接目录上传脚本在实验016的`UPLOAD.ps1`，上传不自动训练。完整部署说明见实验016的`README_FOUR.md`。
