# exp162 / exp163 A100 配对训练交付

上传两个完整目录到服务器 `/workspace/experience/` 下。它们可独立保存训练与评价，套件入口会找到同级的另一组。

| 目录 | 训练损失 |
| --- | --- |
| exp162_20261114_startproj_control_a100_200e | 速度 MSE，精确 t=0 加 0.1×ILM–OPL 最大投影 L1 |
| exp163_20261114_startproj_layermean_a100_200e | A 的损失，再加三张层内均值投影 L1，各系数0.1 |

两组均从头200轮，每轮200更新，固定140/10/50患者。均值范围为ILM–OPL、OPL–BM和ILM–BM。训练输入、网络、时间日程、初始种子、裁块和数值配置配对。选择best时，两组都取浅层最大投影和三张均值投影SSIM的等权平均。与历史exp018比较时，需保留新旧精度和选模规则差异。

参数/数据/优化器/损失/ODE均float32，TF32允许，AMP/scaler关闭，cuDNN benchmark开启。完整科学定义、边界、空射线和留存方案见 [PROTOCOL.md](PROTOCOL.md)。

1. 将本机以下两个目录直接上传到服务器，保持同级结构。

```text
D:/data/code/keyan_shiyan/Flow_oct2octa/experience/exp162_20261114_startproj_control_a100_200e
D:/data/code/keyan_shiyan/Flow_oct2octa/experience/exp163_20261114_startproj_layermean_a100_200e

/workspace/experience/exp162_20261114_startproj_control_a100_200e
/workspace/experience/exp163_20261114_startproj_layermean_a100_200e
```

可以使用已有文件上传界面。两个目录各自包含代码、配置、固定成员表、150人训练/验证标注与单独存放的50人测试标注。源卷和缓存继续读取A100已有目录。

2. 在已分配A100的终端使用现有conda环境。训练环境为 lyh_flowoct，路径 /workspace/.conda/envs/lyh_flowoct。tmux环境为 lyh_tmux，路径 /workspace/.conda/envs/lyh_tmux。conda根目录是 /opt/conda。

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate /workspace/.conda/envs/lyh_flowoct
cd /workspace/experience/exp162_20261114_startproj_control_a100_200e

python -c "import sys,torch,numpy,scipy,skimage,PIL,matplotlib; print(sys.executable); print(torch.__version__)"
conda run -n lyh_tmux tmux -V
```

这两个环境由用户2026年11月15日提供的A100环境列表确认。训练直接使用现有 lyh_flowoct 的Python和依赖，运行验收会记录实际版本。requirements.txt保留本地检查所用版本作为参考。启动器使用当前激活环境的 sys.executable 和绝对前缀，并在tmux中重新激活该前缀。

3. 核验上传，并检查当前Python能看到分配的GPU。

```bash
conda activate /workspace/.conda/envs/lyh_flowoct
cd /workspace/experience/exp162_20261114_startproj_control_a100_200e
python -B suite.py verify
python -c "import sys,torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
nvidia-smi
```

`cuda:0`表示当前可见GPU中的第0张。启动器继承分配终端的 `CUDA_VISIBLE_DEVICES`，实际设备写入运行记录。

4. 准备输入并分别验收两组。准备过程会先后运行，便于记录每组原尺寸更新、完整Euler8和梯度诊断。

```bash
bash SETUP_A100.sh
```

默认路径如下。

| 项目 | A100默认路径 |
| --- | --- |
| 数据根 | /workspace/flowoct_work/data/OCTA500_v1 |
| 原Z256卷 | 数据根下 z256/3M/病例ID.npz |
| 固定清单 | /workspace/experience/exp004_20261025_a100_prior_direct_residualfm_100e100e/data/manifest.csv |
| 训练/验证缓存 | /workspace/experience/exp004_20261025_a100_prior_direct_residualfm_100e100e/cache/train_validation |
| 标签 | 每个实验自己的annotations和test_annotations |

挂载位置不同，在首次准备时传入真实路径。

```bash
bash SETUP_A100.sh \
  --data-root /actual/OCTA500_v1 \
  --manifest /actual/data/manifest.csv \
  --cache-root /actual/cache/train_validation
```

原清单中的旧Windows/A800路径会按冻结病例ID映射到指定数据根，保留原清单SHA和缓存绑定证明。准备核对全部150人训练/验证缓存与原NPZ的 `/255` 逐值一致性。真实验收只使用第一位训练患者，测试输入访问在代码中单独限制。

看到 `Both experiments prepared and accepted. Formal training has not started.` 表示两组验收完成。记录在各自 `verification/preparation.json`、`preflight.json`、`full_preflight.json`，其中有各项损失的加权梯度、实测显存与推理时间。

5. 显式启动200轮训练。默认创建两个独立tmux会话，在当前分配的可见GPU上并行运行。

```bash
conda activate /workspace/.conda/envs/lyh_flowoct
cd /workspace/experience/exp162_20261114_startproj_control_a100_200e
bash START_TMUX.sh
bash STATUS.sh
```

启动器在每个会话中执行 `source` 和 `conda activate`，再调用本次激活的训练Python。创建会话后的真实训练步以 `runs/train/status.json` 中的step/updates为准。日志路径会打印到终端，也记录在 `verification/latest_launch.json`。

6. 查看、进入和离开tmux。

```bash
conda run -n lyh_tmux tmux ls
conda run -n lyh_tmux tmux attach -t exp162_layerproj
```

查看B组时执行下面的命令。

```bash
conda run -n lyh_tmux tmux attach -t exp163_layerproj
```

在tmux内按 `Ctrl+B`，松开后按 `D`，即可离开会话并保持训练。重新SSH登录后执行 `source /opt/conda/etc/profile.d/conda.sh` 与 `conda activate /workspace/.conda/envs/lyh_flowoct`，然后可以继续用 `STATUS.sh` 查看两组状态。

7. 中断后恢复。训练进程退出、tmux会话消失后，再执行同一启动命令。

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate /workspace/.conda/envs/lyh_flowoct
cd /workspace/experience/exp162_20261114_startproj_control_a100_200e
bash STATUS.sh
bash START_TMUX.sh
```

已有会话会显示正在运行。已存在initial检查点的组自动使用resume，完成200轮的组直接显示完成。恢复包含AdamW与RNG，完整训练轮已结束但验证中断时会先补完验证。轮内中断会重跑未提交轮。

也可在自己的tmux终端中直接运行单组，训练与恢复命令分别为下面两条。

```bash
python -B -u run.py train
python -B -u run.py resume
```

8. 查看验证结果和A/B同窗报告。

```bash
python -B suite.py status
python -B suite.py report
```

两组都有完整验证best后，会生成A组目录下 `reports/comparison_validation/index.html`。其中包含九类对象的全部MAE/PSNR/SSIM、最优加粗、SSIM柱状图和同患者分层图。各组原始报告在 `reports/validation/epoch_轮次_权重指纹/index.html`。

9. 两组200轮预算完成后，显式执行50人正式测试与比较。

```bash
python -B -u suite.py test
```

这个命令依次对A/B执行 `run.py freeze`、`run.py test`，然后生成 `reports/comparison_test/index.html`。测试时各组保留50份完整float32三维预测，所有指标和图像使用各自验证best。需要保持SSH断开后继续测试时，可以单独创建测试会话。

```bash
conda run -n lyh_tmux tmux new -s exp162_163_test
source /opt/conda/etc/profile.d/conda.sh
conda activate /workspace/.conda/envs/lyh_flowoct
cd /workspace/experience/exp162_20261114_startproj_control_a100_200e
python -B -u suite.py test
```

10. 回传时直接同步两组的 `reports/`、`runs/`、`logs/`、`verification/`、`provenance/` 和配置文件。模型检查点与正式三维预测保留在服务器，按需要单独同步。每组完整三维测试原始载荷约4.4GiB，验证期间只保留二维证据。

本次本地交付的实际检查见 `verification/LOCAL_DELIVERY.json`。上传、目标A100实测和正式训练状态以服务器执行后的记录为准。
