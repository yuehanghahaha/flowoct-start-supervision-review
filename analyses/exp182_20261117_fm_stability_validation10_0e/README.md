# exp182 · A–H强度漂移与FM轨迹诊断

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

E90为仍可运行的低谷权重，E125权重缺失。E190为exp163原四视图best，XOCT E213为其原规则best。GT校正使用逐病例真值，用于敏感性分析。G/H于2026年11月18日00:06完成，结果、逐患者数据和对照图已补充于下方入口。


## G/H完整结果（2026年11月18日补充）

四个exp163权重×固定十位验证患者，共320项求解器评价、840项扰动与无扰动对照。A–H最终报告校验2,220份实验文件。原A–F数值与图保留原快照。

- [G/H完整分析与对照图](analysis/gh/report.md)
- [G/H逐患者标准指标：10,440行](analysis/gh/per_patient_standard.csv)
- [患者等权的九对象三指标：1,044行](analysis/gh/standard_metrics.csv)
- [G原始逐患者扰动传播：840行](analysis/gh/perturbation_recovery.csv)
- [G原始/截断后72组统计](analysis/gh/recovery_groups.csv)
- [H逐患者求解器结果与运行信息：320行](analysis/gh/solver_runtime.csv)
- [H全部32种权重与求解器组合](analysis/gh/solver_means.csv)
- [H相对Euler8的864组配对差](analysis/gh/solver_paired_deltas.csv)
- [原始1,160条分支记录、40条组提交及哈希](source/gh_records.json)

在本目录使用已有NumPy环境运行 `python -B verify_gh_data.py`，独立核对31,320项逐患者标准值、3,132项患者均值、扰动传播和求解器配对统计。该脚本只读取数据，不加载模型。原A–F复算命令仍有效。

G从理想路径三个节点开始，与匹配无扰动轨迹比较；原始和最终[0,1]截断后结果分别记录。H保留原Euler8参考，Heun4/8分别为8/16次速度求值且包含t=1查询。完整方法和解释边界见报告及METRIC_PROTOCOL.md。
