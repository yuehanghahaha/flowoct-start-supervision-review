# 结果回传说明

本实验完成固定验证 best 的全部 50 位测试患者后，会自动生成 `analysis_return/` 并写入 `test_complete.json`。在服务器运行 `python return_results.py verify` 或在本机下载后执行同一命令，可核对全部回传文件。

这里保存同一个选定权重的完整 27 项三指标、所有患者与 B-scan 数值、训练和验证轨迹、同窗投影与固定切片，以及重现所需的输入、权重、求解器、源码和环境身份。测试的完整三维预测和检查点在服务器实验目录，具体文件路径与哈希登记在 `server_volumes.json`。

数据下载完成后，保留实验父目录名 `exp172_20261116_fullvolume_rms_mamba_bf16_200e` 与 `analysis_return/`。可在本机运行本实验的 `DOWNLOAD.ps1` 按目录下载并自动校验。

本实验使用BF16网络autocast，float32 ODE与损失；冻结训练集RMS统计随结果提供。diagnostics目录是冻结best后的10名验证患者条件错配/全时间探针，不参与选模。
