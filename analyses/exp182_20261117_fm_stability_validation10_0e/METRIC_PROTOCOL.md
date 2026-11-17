# 数据、指标与区域定义

## 固定比较

OCTA-500 3M原140/10/50划分，本次读取固定10位验证患者。完整ZYX为256×304×304，影像固定除255，评价data_range为1，逐病例计算后患者等权平均。float32参数和ODE状态，matmul与cuDNN允许TF32，AMP和scaler关闭，cuDNN benchmark开启，deterministic关闭。

FM原Euler8在全局状态推进，256×64×64滑窗，50%重叠，81窗，正Hann速度融合。每步保持有符号状态，终点clamp到[0,1]。GPU融合与原CPU融合在首例逐体素一致。XOCT保留原96横向窗、4×4网格及均匀拼接，使用本次数值设置。历史XOCT训练的TF32关闭状态在原配置中记录。

## 标准指标

- `volume`为完整三维体积，SSIM采用三维窗口。
- `bscan2d`覆盖全部304张[Z,X]切片，先每张计算、卷内平均，再患者等权。
- `ILM_OPL`为主浅层最大投影，`OPL_BM`为深层最大投影。
- `FULL`为全深度均值，`ZMAX`为全深度最大。
- `ILM_OPL_MEAN`、`OPL_BM_MEAN`、`ILM_BM_MEAN`为各层带均值。
- 每类对象均有MAE、PSNR和SSIM。SSIM为uniform win7、样本协方差，切片和投影采用二维窗口。PSNR为−10log10(max(MSE,1e−12))。
- 原native640六层边界为MATLAB一基坐标，按重采样体素中心选取层带。最大投影闭区间，ILM–OPL均值上界半开，其余沿原实现。
- 投影为XY方向，体素层界为ZYX。实现见 `implementation/measure.py` 与 `implementation/metrics.py`。

## A–D的GT区域

二维GT血管与大血管掩膜合并后提取骨架，在骨架处用距离变换估计直径，再分配给最近骨架点对应的血管像素。直径阈值固定4像素。分别统计浅层对应射线和GT峰值50%以上的候选体素，区域定义固定。

`retina_exterior`为GT分层定义的ILM–BM层带之外，`background_ILM_OPL`为GT二维无血管射线对应浅层体素。FAZ分别使用官方GT_FAZ与GT_FAZ3D，三维标注核对方向后最近邻映射到Z256，并限定于ILM–BM内。

GT校正从未截断终点计算，分别校正全局均值、全局均值与标准差、逐深度平面均值，最后clamp到[0,1]，用于逐病例GT敏感性诊断。

## E–F的时间与公式

`x_gt(t)=(1−t)X+tY`，目标速度为`Y−X`。E的step0–8对应t=0,.125,…,1，状态误差比较相同时间理想插值。F的step0–7为Euler更新节点。t=0时teacher与free共用同次前向，其余在各自状态查询同一权重。

`e_next=e+dt*(v_free−v_gt)`，MSE变化应为`2*dt*mean(e*(v_free−v_gt))+dt^2*velocity_MSE`。`error_velocity_inner`为该整卷内积均值，`mse_balance_residual`为实测变化减去代数预测。原始误差保留符号，截断后诊断另列clipped字段。

二维Dice与clDice使用GT主投影Otsu阈值。FAZ面积取GT FAZ重心所在封闭无血管连通区，开放或无效时为空，空值保留在表格中。
