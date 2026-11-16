# 建议如何审阅

本分支提供一个正在训练的研究方案与可追溯的早期测量。请把已经测得的结论、静态代码推断和仍需实验的假设分开。

## 可以直接转发的分析请求

请审阅这个完整3D OCT→OCTA流匹配实验，重点分析网络和训练目标是否合理、训练为何约4.19秒/步，以及怎样利用A100 80GB减少额外计算。请引用具体文件或函数，核对公式和张量尺寸，并给出按收益/风险排序的建议。

请回答：

1. 逐带RMS归一化、解析高斯参考速度、残差目标、全时间端点估计是否数学一致？真实目标不是高斯时，应如何理解该解析分量的作用和局限？
2. 在精确t=0及其他t，网络是否仍可能依赖容易拟合的项而忽略患者OCT条件？当前设计能排除什么，不能排除什么？需要哪些具有区分力的诊断？
3. CNN多尺度融合与12层共享双向轴向Mamba是否提供所需的局部/全局建模？轴线重排有没有混合患者或不同空间轴？零初始化门控和输出头有什么学习阶段影响？
4. 高分辨率CNN、Mamba扫描、FFN、全时间辅助损失、反向和激活重计算，哪些最值得做分段计时？请不要把GPU利用率样点当作Tensor Core效率。
5. 当前只试过完全不重计算B2和较大范围的选择性重计算。怎样用80GB内存减少重复前向，同时保持同样的参数、有效B2、分辨率、损失和数值协议？应怎样验证正确性和提速？
6. 200×200更新在140人、B2下相当于约571次遍历；这个预算如何评价？请区分单步效率、训练预算、收敛速度和最终质量，不能把减小预算当作相同实验的无代价加速。
7. E5/E10/E15原best轨迹及完整27指标能说明什么？为何体积、B-scan、最大投影、均值投影可能出现不同走势？EMA=.9995、2000步热身和未完成训练有哪些解释边界？
8. 来源清单中哪些是真正复用的实现，哪些只是结构思路，哪些是本项目新增选择？请指出未充分支持的创新性或机制解释。

请首先给出有证据的主要问题及测量方案；对于需要改动算法、数据、时间采样或科学预算的建议，明确说明改了什么和可能牺牲什么。不要预先声称某项优化必然提速几倍或改善指标。

## 阅读顺序

| 要分析的问题 | 入口 |
| --- | --- |
| 实际速度与预算 | [耗时报告](reports/training_time.md)、[原统计](evidence/timing_status_20261116_2035.json)、[计算汇总](evidence/timing_summary_20261116_2035.json) |
| 网络、轴线重排、激活重计算 | [fm3d/model.py](source/fm3d/model.py)，特别是AxialBlock.axial与WaveletVelocity.block |
| 路径、速度、终点、MIP | [fm3d/flow.py](source/fm3d/flow.py) |
| RMS统计及数据隔离 | [normalization.py](source/normalization.py)、[dataset.py](source/dataset.py) |
| 时刻与患者采样 | dataset.schedule；每轮100个精确起点患者、300个均匀时刻患者 |
| 优化、计时、恢复 | [train.py](source/train.py)中的optimization_step、prefetched、train_epoch、snapshot |
| 验证生成与指标 | [report.py](source/report.py)、[metrics.py](source/metrics.py)、[projection.py](source/fm3d/projection.py) |
| 实际A100验收 | [full_gpu.json](evidence/full_gpu.json)、[GPU采样](evidence/timing_gpu_20261116_2035.json) |
| 真实暂停/恢复 | [gpu_restart.json](evidence/gpu_restart.json)、[启动记录](reports/start_and_resume.md) |
| 早期验证全部指标 | [CSV](evidence/validation_metrics.csv)、原统计JSON的validations字段；同一个E15原best用于全部九组指标 |
| 具体参数和来源 | [experiment.json](source/experiment.json)、[PROVENANCE.md](source/PROVENANCE.md) |

## 已知证据边界

- 快照只到2026年11月16日20:35；服务器训练继续，不能把本分支当作实时监控。
- 正式50人测试尚未开始；当前三次评价都是固定10人验证，不能称为最终测试结果。
- 耗时记录是完整更新范围及轮次间隔，没有算子级或CUDA内核级profiling。
- 约1.68%是单步计时范围外的轮内额外开销；不能称为纯I/O占比。
- GPU三个短样点100%/94%/100%，不代表持续平均利用率或已达到理论峰值算力。
- 关闭重计算B2发生OOM；这并没有测出每一种更细重计算策略的内存/速度。
- 全时间监督是每个抽样时间都应用损失；训练每步只抽一个t，不是每步执行Heun16。
- 理论一致性、可运行性、条件利用及生成质量是不同结论；工程验收没有证明研究假设成立。
- 代码/配置/划分的数学部分保留原字节；行政路径替换及原始指纹逐项记录，未发布原始三维图像或模型权重。
