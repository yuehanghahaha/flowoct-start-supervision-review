# exp035：官方代码基线＋三项论文依据补丁，200轮

本版响应用户2026-10-29的明确要求：基于公开代码，仅修改有论文直接依据的内容；服务器运行和最终报告所需补充逐项列明。父实验exp033、旧exp026/028及其结果保持原样，本版从新初始化训练。

**先看[修改与补充清单](CHANGES_CHECKLIST.md)。** 清单逐项给出原代码、本版行为、论文公式/章节、补充原因和影响。代码差异原件见`provenance/model_paper.patch`及`provenance/train_integration.diff`。

三项科学改动全部位于官方`source/models/TransPro_model.py`：

| 修改 | 论文依据 |
| --- | --- |
| LSGAN改为稳定实现的对数minimax | §3.1式(1)/(4)，§3.4式(13) |
| 投影L1从5改10，其余保持体10/HCG5/VPG5 | 式(7)/(12) |
| VPG由softmax概率改为原始logits L1 | §3.3式(10)/(11) |

作者网络、优化顺序、D损失0.5、LR调度及其调用时机、教师/验证BatchNorm行为、每轮单裁块MAE验证、4worker、no_flip、默认精度标志均保留。不加逐参数扫描、确定性卷积或训练中八项评价。官方best/latest及编号模型保存行为也保留，完整八项和4窗口全视野报告在200轮后单独生成。

## 当前服务器

SSH：`eias-member.eitech.edu.cn:42060`，用户root；现场实际分配GPU为**A800 80GB**。实验目录：

```text
/docsystem/niuke/lvyuehang/Flow_oct2octa/experience/exp035_20261029_transpro_paperpatch200e
```

复用现有环境Python：`/docsystem/niuke/lvyuehang/conda_envs/lyh_flowoct/bin/python`；torch2.5.1+cu121、torchvision0.20.1、torchio0.20.4。缺少的torchsummary/dominate仅装到本实验runtime_deps，不替换其他运行环境。TF32保持作者/框架默认并记录实际值，不能称为严格关闭TF32的FP32。

复用现有150人只读NPY缓存及固定140/10/50清单；不拷贝大体积数据。公开GT_Capillary的200个原始BMP随本目录同步，训练不使用该掩膜作新监督；prepare只打开150人训练/验证标注，测试50人标注到冻结后才解码使用。

## 上传、准备、检查

直接同步本目录，不打ZIP/tar、不覆盖旧实验。Windows可运行本目录`UPLOAD.ps1`，默认采用已配置的42060入口。

服务器进入本目录后：

```bash
bash SETUP_SERVER.sh
```

脚本只补依赖、执行prepare和原宽度256立方合成GPU检查，**不开始200轮训练**。详细结果见`verification/gpu_check.json`，成功标志为：

```text
Paper-patched TransPro GPU check passed. Formal training has NOT started.
```

当前服务器的缓存/数据位置已经写入`experiment.json`。路径变更可在首次prepare显式传`--cache`、`--manifest`、`--data-root`、`--labels-root`；既有binding不同则拒绝混用，不静默替换数据。

采用作者编号模型保存后的最坏未来空间预算193.21 GiB（最多200套G/D/D_proj、best/latest、resume更新临时副本、最终60份完整预测，含10 GiB预留）。实际占用取决于best改善次数，现场检查按tensor大小计算；这不是论文规定磁盘需求。实验所在共享盘`df -h`实测约198T空闲，空间检查已通过；旧磁盘392 GiB门槛不再用于本版。

## 单独启动训练

确认检查通过后，在服务器tmux中新建/进入会话，再启动训练：

```bash
tmux new-session -A -s transpro_exp035
cd /docsystem/niuke/lvyuehang/Flow_oct2octa/experience/exp035_20261029_transpro_paperpatch200e
bash START.sh
```

START使用已指定的Python，无需在root的默认Python中重新装torch。若tmux不在PATH，使用已有工具环境的tmux；不要因启动界面问题重新创建或替换训练环境。

实时日志另开终端查看：

```bash
tail -n 50 -F /docsystem/niuke/lvyuehang/Flow_oct2octa/experience/exp035_20261029_transpro_paperpatch200e/logs/train_console.log
```

离开tmux保持训练：Ctrl+B，松开，再按D。状态查询：

```bash
/docsystem/niuke/lvyuehang/conda_envs/lyh_flowoct/bin/python -B run.py status
```

作者原每轮验证MAE与新增`[timing]`分别记录；时间分为训练、验证及包含保存的整轮总时间。作者LR调用在当前PyTorch下可能导致200轮内基本不降，本版按“其他内容不动”保留，不擅自修正。

续训仍执行`bash START.sh`。完整恢复点为第1轮和每5轮末，包含G/D/教师BN、Adam、scheduler、RNG和边界best；其后轮次重新执行。为保留旧编号结果，重跑保存到新子目录。只允许本版同一协议恢复，不能接exp026/028/033权重后称为本版从头复现。

## 训练后的八项评价及图

200轮完成后自动为所选best生成完整10人验证报告；如报告中断可恢复：

```bash
/docsystem/niuke/lvyuehang/conda_envs/lyh_flowoct/bin/python -B -u run.py evaluate
```

验证报告完整后冻结，再测试固定50人：

```bash
/docsystem/niuke/lvyuehang/conda_envs/lyh_flowoct/bin/python -B run.py freeze
/docsystem/niuke/lvyuehang/conda_envs/lyh_flowoct/bin/python -B -u run.py test
```

`results/{validation,test}/epoch_NNN/`保存完整float32预测、逐B-scan/患者8项指标、论文差表及对照图、每人OCT/GT/预测/误差投影图和浮点投影。训练中不保存每轮完整3D预测。最终指标的[0,1]范围、SSIM参数、拼接和公开掩膜来源等未知细节见清单E1–E7，不能声称已核实作者完全相同。

## 状态与核查

11项本地合成/完整性检查通过。**本版已经同步至42060服务器，prepare通过，原宽度256立方GPU前向/反向/优化器更新通过；尚未启动正式训练。** A800峰值已分配显存约22.85 GiB。实际cuDNN TF32允许、benchmark=True、deterministic=False；完整记录在`verification/gpu_check.json`，环境包版本见`verification/server_environment.json`。上传与检查成功不等于已正式开训。

所有修改、必要补充及未公开的选择均在[CHANGES_CHECKLIST.md](CHANGES_CHECKLIST.md)列出。本版只按论文明确公式修改三项，不将补充工程代码或原库未公开评价细节包装成论文要求。
