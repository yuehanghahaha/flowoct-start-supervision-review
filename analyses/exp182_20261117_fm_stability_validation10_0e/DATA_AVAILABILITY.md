# 共享数据范围

共享50条完成的模型病例记录、十位验证患者的全部A–F数值、15,200张B-scan指标、直方图计数、全256深度统计、同时间轨迹和速度统计、GT校正敏感性、七幅诊断图、完整协议、源代码与来源哈希。

图像包括十人聚合曲线，以及预先固定首例10302的五组同色标有符号误差和四个FM权重九状态二维预览，选择规则在诊断代码中固定。

完整三维OCT/OCTA、360个三维中间状态、GT分割、权重及原始二维数组保留于实验持久化存储。仓库提供身份记录，重跑图像级推理需要这些外部依赖。复算报告数值使用共享JSON、CSV与NumPy即可。

2026年11月18日补充完整G/H：320条求解器记录、840条扰动及对照、40条组提交，10,440行逐患者标准指标、1,044行均值、72组扰动统计、864组求解器配对差、两幅新图及离线核对脚本。原A–F表继续保留；G/H表位于analysis/gh。父仓库exp173/174提供历史波动与二维证据背景。

source/af_core_records.json中的files字段为原实验文件身份。verification/original_af_summary_audit.json保留首次汇总时阶段表哈希。本次per_patient_standard.csv筛选A–F处理方式，verification/af_summary_audit.json记录筛选后的5,400项核对结果。


G/H数值来源为 `source/gh_records.json`，整份源快照与冻结身份的SHA256在 `analysis/gh/verification.json`。其中record_sha256及组提交files保存服务器原记录文件身份；本仓库提供聚合JSON，不将这些哈希误称为拆分后重排JSON的文件哈希。共享源和筛选转换见 `provenance/gh_source_manifest.json`。G/H不含逐张B-scan明细，包含原评价中完整304张B-scan的卷内平均指标。

`implementation/gh_analysis_original.py`保留原实验中的汇总与绘图实现及其路径假设；公开快照的离线数值复核入口为 `verify_gh_data.py`。重建原始图像级推理仍需要外部权重、OCT/OCTA和标注；本次没有上传这些三维数组或重新训练。
