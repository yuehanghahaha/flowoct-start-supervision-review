# OCT→OCTA · A–F强度漂移与生成轨迹验证

本分支新增2026年11月17日完成的exp182 A–F十人验证，供独立分析者检查强度漂移、逐步生成误差和teacher/free速度差异。

**从 [A–F报告与数据入口](analyses/exp182_20261117_fm_stability_validation10_0e/README.md) 开始。**

- [十人结论与完整指标](analyses/exp182_20261117_fm_stability_validation10_0e/analysis/af_conclusions.md)
- [给分析者的阅读说明](analyses/exp182_20261117_fm_stability_validation10_0e/REVIEW_GUIDE.md)
- [逐患者指标与GT校正](analyses/exp182_20261117_fm_stability_validation10_0e/analysis/per_patient_standard.csv)
- [逐步轨迹数据](analyses/exp182_20261117_fm_stability_validation10_0e/analysis/trajectory.csv)
- [Teacher/free速度误差](analyses/exp182_20261117_fm_stability_validation10_0e/analysis/velocity_teacher_free.csv)
- [指标与区域定义](analyses/exp182_20261117_fm_stability_validation10_0e/METRIC_PROTOCOL.md)

四个exp163权重E90、E120、E130、E190及XOCT E213，共50个完成的模型病例。完整3D、逐B-scan和七投影均保留MAE、PSNR、SSIM。E125权重缺失，本次E90为可运行低谷，E190和XOCT E213保留原选模身份。

E90的3D SSIM为0.364562，逐病例GT均值与方差校正后为0.849173，10/10患者改善。共享数据同时提供背景偏差、候选血管区域、截断前后范围和自由轨迹变化。

```text
python verify_snapshot.py
cd analyses/exp182_20261117_fm_stability_validation10_0e
python -B analyze_af.py
```

当前分支为 `exp182-af-diagnostics-review-20261117`。原有9组科学代码和exp173/174分析继续作为背景，入口为 [历史审阅说明](HISTORICAL_REVIEW.md)。

范围见 [A–F数据说明](analyses/exp182_20261117_fm_stability_validation10_0e/DATA_AVAILABILITY.md)，全分支文件身份在SNAPSHOT_MANIFEST.json。
