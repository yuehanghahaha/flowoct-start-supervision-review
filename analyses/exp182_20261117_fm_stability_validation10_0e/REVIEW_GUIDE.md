# 给分析者的阅读说明

本次研究固定权重下的强度漂移与训练、推理轨迹差异。数据为固定验证患者10302、10319、10338、10357、10396、10422、10447、10456、10495、10500。

## 阅读顺序

1. 阅读 `analysis/af_conclusions.md`，查看两个十人统计图 `intensity_depth.png` 和 `trajectory.png`。
2. 在 `analysis/per_patient_standard.csv` 配对查看同病例原输出与三种GT强度校正，核对全部九类评价对象。
3. 用 `gt_region_bias.csv` 与 `depth_profiles.csv` 定位信号抬高区域，查看截断前后两个stage。
4. 对齐 `trajectory.csv` 与 `velocity_teacher_free.csv` 的t，检查状态偏差、速度偏差、方向内积和递推。
5. 查看 `implementation/measure.py` 与 `implementation/trajectory.py`，核对分层、候选区域、截断、拼接和时间节点。
6. 运行 `python -B analyze_af.py`，从50条冻结记录复算结论与标准指标。

## 希望重点分析的问题

- E90背景正偏与较低血管对比度，如何解释3D SSIM变化和GT校正敏感性？
- E90在理想路径上已有正向速度均值误差，理想与自由路径差异各贡献了什么？
- 40条自由轨迹的误差与速度误差内积持续为正，对实际纠错方向有何含义？
- 较高3D SSIM权重仍有大量截断前负值，应怎样结合背景、投影与微血管指标解释？
- 指标、区域定义或实现中是否有影响判断的具体问题？
- 下一项最小对照应隔离哪个机制，才能解释改变后的结果？

请将反馈对应到具体病例、字段或代码行，说明判断来自哪项观测。E90、E120、E130为诊断检查点，E190和XOCT E213保留原best身份。GT强度校正使用了真值信息，原网络指标使用unadjusted行。
