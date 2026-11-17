# exp182 · A–F强度漂移与FM轨迹诊断

2026年11月17日完成的固定十人验证，四个exp163权重及XOCT E213共50个模型病例，新训练0轮。

- [十人结论与完整指标](analysis/af_conclusions.md)
- [给独立分析者的阅读说明](REVIEW_GUIDE.md)
- [指标和区域定义](METRIC_PROTOCOL.md)
- [数据字段与留存范围](DATA_AVAILABILITY.md)

## 快速复算

在本目录使用已有NumPy环境运行：

```text
python -B analyze_af.py
```

脚本读取 `source/af_core_records.json`，核对冻结模型与患者身份，以及 `analysis/per_patient_standard.csv` 中5,400个标准值，重新生成 `analysis/af_summary.json` 和 `analysis/af_conclusions.md`。本次发布已经实际复算并核对两个输出逐字节一致。

## A–F数据入口

| 实验 | 文件 | 内容 |
|---|---|---|
| A | analysis/intensity_distributions.csv | 100行，五组×十人×截断前后，均值、标准差、分位数和越界率 |
| A | analysis/gt_region_bias.csv | 1,300行，固定GT区域体素数、均值、偏差和MAE |
| B | analysis/depth_profiles.csv | 25,600行，逐深度强度、偏差、MAE |
| C | analysis/figures/signed_error_first.png | 预先固定首例10302，五组同色标有符号误差 |
| D | analysis/per_patient_standard.csv | 1,800行，原输出和三组GT校正的九类对象三指标 |
| D | analysis/oracle_volume_delta.csv | GT校正相对原输出的3D指标变化 |
| E | analysis/trajectory.csv | 360行，四组×十人×九状态，同时间理想与自由状态误差 |
| F | analysis/velocity_teacher_free.csv | 320行，四组×十人×八节点，速度误差、内积和递推残差 |
| 标准指标 | analysis/bscan_per_slice.csv | 15,200行，每组每人全部304张B-scan的指标 |
| 汇总 | analysis/standard_means.csv | 180行，患者等权的完整3D、B-scan与七投影三指标 |
| 补充 | analysis/topology.csv | 二维投影Dice、clDice与FAZ面积诊断 |
| 全量数值 | source/af_core_records.json | 50条完成记录、GT分布、直方图计数及原文件哈希 |

`implementation/`保留实际诊断代码。原训练代码位于本分支 `../../experiments/exp163_20261114_startproj_layermean_a100_200e/` 与XOCT目录，原权重、输入和源码身份在 `verification/frozen.json`。

E90为仍可运行的低谷权重，E125权重缺失。E190为exp163原四视图best，XOCT E213为其原规则best。GT校正使用逐病例真值，用于敏感性分析。G/H的另行运行进度在原实验记录中维护。
