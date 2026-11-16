# XOCT部署与训练启动交付记录

核查时刻：2026-11-02 22:59:31（北京时间）。本记录是启动快照，后续实时进度以服务器 `runs/status.json` 和完整epoch记录为准。

首轮闭环验收补充（23:05:06）：140位不同训练患者各一次更新全部完成，10例作者原规则验证通过；完整 `latest.pt` 为864,459,523字节，训练已进入第2/300轮并推进至总170次更新。稳定更新中位约2.16秒。证据见 `verification/first_epoch_acceptance.json` 及 `runs/epochs/epoch_001.json`；这不表示300轮预算或正式测试完成。

| 阶段 | 状态与证据 |
| --- | --- |
| 上传 | 122个源码、配置及溯源文件逐文件SHA256核对一致，作者104个文件保持固定提交原始字节；`verification/delivery.json` |
| 独立环境 | Python3.11.18，PyTorch2.7.0+cu126，torchvision0.22.0+cu126；109项作者锁定版本全匹配，pip check无冲突；`environment_pins.json`、`pip_freeze.txt`、`pip_check.txt` |
| 数据 | 固定140训练／10验证／50测试；只转换150训练/验证病例，450个派生文件全部哈希通过；原始NPZ、原始层标注与转换身份留档；未打开测试影像 |
| GPU验收 | 原尺寸256×96×96、batch1的D/G更新成功，损失/参数/梯度有限，float32且AMP/GradScaler/TF32关闭；训练allocated峰值34.07GiB |
| 恢复验收 | 5个网络、两个Adam、scheduler、Python/NumPy/PyTorch/CUDA RNG保存/重读并继续更新通过 |
| 整卷验收 | 一位训练患者10301，完整256×304×304、原作者16窗口；单例推理及全部指标约19.64秒，allocated峰值8.47GiB；不保存中间三维预测 |
| 清理 | 验收临时断点删除；安装缓存/下载文件共5,667,706,595字节清理，保留源代码、环境、VGG权重、数据及科学证据；未打交付压缩包 |
| 正式训练 | 已于22:55:40记录启动，独立tmux `xoct_exp059`、PID286885；核查时第1/300轮，实际成功更新32次，稳定更新中位约2.14秒，损失有限；未完成完整训练预算 |
| 正式测试 | 未启动；测试不参与训练、验收或选模 |

服务器目录：`/docsystem/niuke/lvyuehang/Flow_oct2octa/experience/exp059_20261102_xoct_paperlr_300e`。

原代码与论文差异及必要补充见 `provenance/protocol_decisions.json`。明确保留：论文学习率1e-4；原作者LSGAN、网络、VGG预处理及scheduler调用行为；最终第300轮作为预先固定主权重，原规则201–300轮best作为补充。固定项目140/10/50划分、严格FP32、层边界栅格化、完整断点和整卷补充评价均单列为项目补充，不能宣称论文同划分、同预处理表格复现。论文三维Perp.的公开实现仍缺失，不编造该列。

首次更新含算法选择，验收162.08秒、正式159.52秒；稳定约2.14秒仅是计算更新计时，不是包含全部数据读取、验证、落盘的全程速度，不据此承诺完成日期。正式训练重新同种子初始化，未使用GPU验收权重，也未加载作者已训练XOCT权重。

上传、验收、启动和持续成功更新分别见 `verification/delivery.json`、`preflight.json`、`training_launch.json`、`training_observation.json`。临时/安装文件清理见 `cleanup.json`、`installation_cleanup.json`。本地三个生成的Python字节码缓存删除被自动审批拒绝（返回blocked by policy），已保留，记录见 `verification/local_cleanup.json`；不影响服务器训练或科学证据。
