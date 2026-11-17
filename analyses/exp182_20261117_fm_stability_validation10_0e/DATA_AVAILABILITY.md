# 共享数据范围

共享50条完成的模型病例记录、十位验证患者的全部A–F数值、15,200张B-scan指标、直方图计数、全256深度统计、同时间轨迹和速度统计、GT校正敏感性、七幅诊断图、完整协议、源代码与来源哈希。

图像包括十人聚合曲线，以及预先固定首例10302的五组同色标有符号误差和四个FM权重九状态二维预览，选择规则在诊断代码中固定。

完整三维OCT/OCTA、360个三维中间状态、GT分割、权重及原始二维数组保留于实验持久化存储。仓库提供身份记录，重跑图像级推理需要这些外部依赖。复算报告数值使用共享JSON、CSV与NumPy即可。

G/H由独立运行产生，本次共享范围为已完成A–F。父仓库exp173/174提供历史波动与二维证据背景。

source/af_core_records.json中的files字段为原实验文件身份。verification/original_af_summary_audit.json保留首次汇总时阶段表哈希。本次per_patient_standard.csv筛选A–F处理方式，verification/af_summary_audit.json记录筛选后的5,400项核对结果。
