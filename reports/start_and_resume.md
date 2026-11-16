# exp172 已启动

记录日期：2026年11月16日。最近核实：2026-11-16T16:28:40+08:00。

上传、验收与启动分别完成：30份冻结源码校验通过；CPU科学恒等式和恢复检查通过；
两名训练患者完整分辨率、BF16物理B2的25次前后向与Heun16推理验收通过；
正式训练于2026年11月16日16:19:10（UTC+8）开始。未使用测试病例进行准备或检查。

实际38,109,512参数；整卷256×304×304；12层384宽轴向Mamba与三尺度CNN；
所有时间的归一化速度/终点体素/浅深MIP监督。计划200×200=40,000更新，没有缩短。

精度：网络BF16 autocast；权重/AdamW/EMA/Haar/路径/ODE/损失float32；
TF32 matmul/cuDNN均开，benchmark开，deterministic关，无GradScaler。
实际物理batch=2、累积1、选择性重计算；稳态中位4.1876秒/更新。
不重计算的B2因显存不足未被采用。正式记录至100更新峰值：
分配24.919GiB、预留26.346GiB。

真实暂停于第57更新，检查点610,343,903字节，包含AdamW/EMA/CPU与CUDA RNG/未完成轮次。
恢复后继续至第100更新并再次保存，源码/输入/统计签名全部一致。
当前第1轮，已保存100更新；完整验证/正式测试均未到开始时间，不能评价模型质量。

旧exp131保持停止，latest为E115/23000更新，原best E115；没有改动或删除旧实验。

服务器：/home/HPC_USER/experience/exp172_20261116_fullvolume_rms_mamba_bf16_200e
容器：/workspace/experience/exp172_20261116_fullvolume_rms_mamba_bf16_200e
已申请slot2，环境lyh_flowoct；tmux工具来自lyh_tmux，会话exp172_fullvolume_rms_mamba_bf16。
操作入口：bash /workspace/experience/exp172_20261116_fullvolume_rms_mamba_bf16_200e/manage.sh status
支持pause/resume/log/attach。没有申请、续租、释放GPU或变更环境依赖。
租期结束后须由用户申请有效资源，再用resume继续；每20更新及完整轮/安全暂停均提交latest。

本地DOWNLOAD.ps1下载完成后的analysis_return，-Volumes附加完整50人三维结果。
每5轮整卷10人验证；200轮预算完成冻结原best后自动测试50人、运行固定验证条件诊断并导出。
完整27指标口径、所有网络宽深、损失、时间与来源见README.md、PROVENANCE.md。

证据：verification/training_launch.json、gpu_restart.json、full_gpu.json、correctness.json、
training_normalization.json、server_mapping.json；服务器runs/status.json、checkpoints/latest.pt及runs/a100/。


### 2026年11月16日16:36 首轮完整提交

已核验第1轮全部200次成功更新，记录连续1–200，均为两名不同患者的完整B2；
共100位患者精确t=0、其余300位U(0,1)，全部损失与梯度范数有限。
首轮平均训练损失1.450292；该值不是正式生成效果评价。
首轮张量分配/缓存预留峰值为24.919/
26.346GiB。
截至2026-11-16T16:36:12+08:00，已提交第1轮，正在第2轮，状态记录第201次更新。
GPU设备单次采样为NVIDIA A100 80GB PCIe, 81920 MiB, 24383 MiB, 100 %；100%为瞬时样点，不是整个训练的平均。
正式验证需到第5轮，正式测试仍未开始。
当前分配核查约剩22小时20分钟，按4.19秒/步，40,000次纯更新约46.5小时，
另有验证/测试开销；本次没有续租、重新申请或缩短预算，后续须在有效分配中继续。
证据：verification/epoch_001.json、first_epoch.json、handoff_status.json。
本次笔记更新日期：2026年11月16日。

**2026年11月16日16:42:41交接核实：** exp172仍在训练，第1轮已提交，正在第2轮，累计300次更新；exp131状态为stopped。来源verification/final_status.json及服务器runs/status.json。
