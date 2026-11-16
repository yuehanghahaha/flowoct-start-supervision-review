# exp163 单实验训练启动记录

北京时间启动：2026-11-15T15:44:39.590668+08:00。最近确认：2026-11-15T15:49:47.236783+08:00。

仅运行 exp163；未启动 exp162 对照或正式测试。使用用户已有 slot1 的 NVIDIA A100 80GB PCIe，单张 GPU。没有申请、释放或变更 GPU 资源，没有创建或修改 Conda 环境。

当前状态：validating，第 5 轮，累计 1000 次正式更新。已确认前 4 轮提交并写入 latest.pt；若正在第 5 轮验证，trained_pending_validation.pt 保存已完成训练、待提交验证的第 5 轮状态。独立 tmux 会话 exp163_layerproj 正在运行，启动 SSH 已退出。

预算保留为 200 轮 × 200 次更新；源码、experiment.json、数据划分、损失、精度和选模规则均未改。float32 张量，matmul/cuDNN TF32 允许，AMP/scaler 关闭，cuDNN benchmark=True、deterministic=False。每 5 轮原分辨率验证，按原定四项投影 SSIM 平均选 best。默认不保存中间验证三维预测。

训练环境：lyh_flowoct；后台工具：lyh_tmux。实际依赖：{"torch": "2.5.1+cu121", "numpy": "1.26.4", "scipy": "1.15.3", "scikit-image": "0.25.2", "Pillow": "12.3.0", "matplotlib": "3.10.1"}。SciPy/Pillow 与 requirements.txt 的固定版本有差异，但与保留的本地计算验收环境一致；matplotlib 与本地记录略有差异。未为此安装或覆盖依赖；实际版本已冻结在运行记录和检查点内。

检查：627 个交付文件完整性通过；140 训练与 10 验证病例逐值核对通过，未打开测试数组或解码测试标签；pip check 通过。原合成检查 checks.py:117 因不同 float32 求和顺序产生 1 ULP（5.960464477539063e-08）差异而失败，原失败日志保留。独立补充审计确认两种求和均在浮点误差界内且全部参数梯度逐位相同，仅在内存中替换该验证断言，未修改交付文件。采用上述验证补充后的 9 组合成/续训回归全部通过。原版真实训练病例全尺寸验收通过，Euler8 为 33.349 秒，显存峰值约 2.987 GiB。细节见 REQUEST.md、loss_grouping_diagnostic.json、synthetic_acceptance.json、preflight.json。

服务器实验目录：/home/lvyuehang/experience/exp163_20261114_startproj_layermean_a100_200e；容器目录：/workspace/experience/exp163_20261114_startproj_layermean_a100_200e。

训练日志：/workspace/experience/exp163_20261114_startproj_layermean_a100_200e/logs/train_20261115_074436_759644.log。

在用户自己的已分配 slot1 内查看状态：

```bash
cd /workspace/experience/exp163_20261114_startproj_layermean_a100_200e
/workspace/.conda/envs/lyh_flowoct/bin/python -B run.py status
tail -f /workspace/experience/exp163_20261114_startproj_layermean_a100_200e/logs/train_20261115_074436_759644.log
```

不要执行默认的双组 SETUP_A100.sh / START_TMUX.sh 来重复启动；本次通过交付代码 suite.start([common.ROOT], "lyh_tmux") 仅启动本实验。训练完成后不会自动运行正式测试，也不会自动申请或释放 GPU。

临时验收数据和临时检查点已由原检查自动清理；保留失败、诊断、完整验收和启动证据。运行中使用的临时文件保留至进程退出。未创建 ZIP/tar 归档。
