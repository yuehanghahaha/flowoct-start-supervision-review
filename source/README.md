# exp172：整卷 RMS 流匹配与 CNN–轴向 Mamba（BF16）

建立日期：2026年11月16日。父实验 exp131；从头训练，不加载其权重。
本实验使用单张已分配 A100 80GB slot2，不申请、释放或续租资源；旧 exp131 保留并不再续跑。

## 科学目标与改动

在整个时间区间学习真实 OCTA 速度残差，改善原实验的高频信号尺度、局部覆盖和条件学习。
这是待验证的组合研究方案；效果与方法新颖性均没有被启动验收证明，不承诺优于现有方法。

相对 exp131：全部八个 Haar 子带整卷训练；训练集逐带 RMS 归一化；显式解析高斯参考速度；
更深更宽的三尺度 CNN 与 12 个交替轴 Mamba；每次两名不同患者；25% 患者精确 t=0；
所有抽样时间同时具有速度、体素终点和浅/深层 MIP 监督。

草案原为 float32 张量、TF32 允许、AMP 关闭。按用户本次明确指令，正式版改为 BF16 autocast：
网络卷积/线性/Mamba 采用 BF16 混合精度；参数、AdamW、EMA、Haar、路径状态、解析速度、
ODE 与损失归约维持 float32；BF16 不使用 GradScaler。它不是强制纯 FP32，也不是全模型 BF16 权重。
matmul/cuDNN TF32=true；cuDNN benchmark=true、deterministic=false，不强制确定性算法。

## 完整配置

| 项目 | 固定设置 |
| --- | --- |
| 数据 | OCTA-500 3M；沿用 membership.json 的 140/10/50 患者划分 |
| 输入 | OCT 与 OCTA，ZYX=256×304×304；存储强度除以255，无逐图拉伸 |
| Haar | 一层正交3D Haar；8带均为128×152×152；含 LLL 和七个高频 |
| OCT编码器 | 8→32→64→96；三尺度各1个深度可分离残差 CNN 块 |
| 状态编码器 | 48/96/128通道；2/2/1个时间条件 CNN 块 |
| 全局网格 | 32×38×38=46,208个位置；Linear128→384，加三维中心位置线性编码 |
| Mamba | 深12、宽384；Z/Y/X交替4轮，每块一轴双向共享权重，共24次扫描 |
| Mamba内部 | 官方v1，d_state=16，d_conv=4，expand=2，dt_rank=24；不混合患者或不同行 |
| 全局FFN | 384→1536→384；非仿射LayerNorm eps=1e-6；AdaLN零门控时间调制 |
| 解码器 | 128/96/48通道，1/2/3个时间条件 CNN 块；三线性上采样与同尺度跳连 |
| CNN块 | depthwise3³→channel LayerNorm→1³扩至2C→SiLU→1³回C+残差 |
| 时间 | 128维sin/cos(t×1000)→384→384；局部shift/scale，全局六组shift/scale/gate |
| 输出 | 八带归一化残差速度；48→8零初始化卷积；时间调制零初始化，Mamba默认初始化保留 |
| 参数 | 实例化实测38,109,512；全局12块36,389,376；单个生成网络，无判别器 |
| batch | 有效batch=2整卷，优先物理B2；仅B2均无法保留8GiB余量时用B1×2累积 |
| epoch | 200轮×200次成功更新=40,000次；每轮400患者访问，共80,000访问 |
| 患者采样 | 每轮140患者两次独立排列，再第三次排列前120人；两人一组、不同患者 |
| 时间采样 | 每轮100组各随机一人t=0；其余300患者t~U(0,1)，独立噪声 |
| 优化器 | AdamW betas=(.9,.999)，eps=1e-8，wd=.01；bias/norm/官方_no_weight_decay不衰减 |
| LR | 2,000更新线性热身到1e-4，之后余弦到1e-5；梯度全局范数裁剪1 |
| EMA | .9995，每次完整优化更新一次 |
| 初始化 | seed=12920261111；从头训练，正式模型不继承验收时的诊断更新 |
| 推理 | OCT特征只编码一次；独立固定患者噪声；float32状态Heun16步/32次网络调用 |
| 验证 | 每5轮，固定10人，完整256×304×304；同一EMA权重计算全部27指标 |
| 选模 | 患者宏平均ILM–OPL max投影SSIM最大，平分保留较早轮次 |
| 正式测试 | 完成200轮与最终验证、冻结原best后，对50人各一次；支持病例边界继续 |
| 检查点 | 每20次更新、每完整轮与安全暂停保存latest；best单独保留 |
| 验证留存 | 保留27指标、每B-scan指标和有用2D图；不落盘中间验证3D预测 |
| 测试留存 | 保存50份完整float32 3D prediction.npy，另留全部2D图与逐患者指标 |

## 路径与损失

仅以140名训练患者估计逐带 RMS s_b，体素RMS d、ILM–OPL和OPL–BM最大投影RMS m_s/m_d；
float64累加，尺度下限1/255，不中心化。记录每个统计输入SHA256、有效射线数与完整成员。
不读取验证/测试以估计尺度，也不把GT层标注输入生成网络；GT层仅用于训练损失/评价。

z1=W(y)/s，z0~N(0,I)，zt=(1−t)z0+t z1，u=z1−z0。
q=(1−t)²+t²，v_ref=(2t−1)zt/q；网络输入zt/√q，输出r；
r*=u−v_ref=((1−t)z1−t z0)/q；总速度v=v_ref+r。
终点 yhat=W^-1(s[zt+(1−t)v])。训练终点不裁剪。

每位患者的损失为：
1/8 MSE(r_LLL,r*_LLL)+7/8 MSE(r_high,r*_high)
+min(update/1000,1)×[.10 L1(yhat,y)/d
+.20(.75 L1(MIP_ILM_OPL(yhat),MIP_ILM_OPL(y))/m_s
+.25 L1(MIP_OPL_BM(yhat),MIP_OPL_BM(y))/m_d)]。

各项在所有采样时间计算，包括精确t=0；不在t接近1时除以1−t。
MIP的空射线排除，采用native640闭区间、与父实验相同的体素中心几何。
B1×2时每个患者损失乘1/2，两人完成后统一裁剪、step和EMA，不能在中途提交暂停状态。

## 启动、暂停与继续

在平台已分配的slot2中，以HPC_USER进入，再运行：
    bash /workspace/experience/exp172_20261116_fullvolume_rms_mamba_bf16_200e/manage.sh start
其他操作使用 pause / resume / status / log / attach / check。
manage.sh复用已有lyh_flowoct Python和lyh_tmux中的tmux；不创建或更新环境。
后台会话 exp172_fullvolume_rms_mamba_bf16；断开SSH不停止训练。
pause在完整优化更新或已完成的评价病例边界提交；resume核验源码、输入与归一化身份。

流水线分开记录source_check、CPU正确性、训练统计、A100检查、正式training、
正式test、冻结best条件诊断、analysis_return导出。GPU检查只用两名训练患者，
每个候选模式预热5次、稳态20次，三类损失权重全部开启。
先测试无激活重计算B2，若超过实测设备总量−8GiB或72GiB上限，再尝试selective/all；
只有B2全不通过才尝试B1×2。不能因显存/租期缩小分辨率或预算。
验证未通过则退出并记录错误，不开始正式训练。

latest含model/EMA/optimizer/CPU与CUDA RNG、轮次/更新、未完轮记录、待验证状态及best。
同种子每步独立噪声和每轮独立患者/时间计划，恢复时跳过已提交更新。
latest只保证可恢复到最后一次成功原子提交，不保证在平台强制回收时保留未提交更新。

## 评价与下载

沿用父实验逐字节相同的metrics.py、projection.py。
9组各MAE/PSNR/SSIM：完整3D；逐B-scan2D；主ILM–OPL max；
OPL–BM max、full mean/max、ILM–OPL mean、OPL–BM mean、ILM–BM mean。
强度范围[0,1]、data_range=1，均匀7窗口SSIM，样本协方差；
B-scan逐Y的ZX切片先整卷平均，再等权患者平均；体积SSIM为3D，投影为2D。
它们不冒充TransPro的血管加权V指标。投影几何与SSIM其余细节见metrics.PROTOCOL。

训练预算完成且best冻结后：固定10人验证OCT循环错配、相同噪声/目标/权重；
另作t=0/.125/.25/.5/.75/.875/.9375逐带速度与终点误差探针。
这是诊断，不能用于更改训练预算、选模或正式测试参数。
保留诊断27指标、匹配与错配投影和逐带探针，不保留诊断3D预测。

使用本地 DOWNLOAD.ps1 下载analysis_return目录；-Volumes附加下载正式50人3D预测。
DOWNLOAD_PROGRESS.ps1用于阶段日志/指标，不把缺少的中间验证3D文件视作错误。
所有交付直接同步目录，无默认ZIP/tar。源码、权重、固定划分和配置使已保留权重输出可重新生成。

## 来源与证据

逐项列于PROVENANCE.md及provenance.json。验收证据位于verification/；
运行环境、实际batch模式、尺度、日志和状态分别在runs/environment/、runs/execution.json、
runs/normalization.json、runs/a100/与runs/status.json。
本地上传和源码校验通过不等于正式训练已经完成成功更新；实时状态以记录为准。
