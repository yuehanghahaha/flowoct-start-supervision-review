# 来源、固定实现与本项目选择

这是独立实现的局部优先流匹配项目。没有读取、复制或导入旧项目模型代码。问题背景来自用户对exp172的讨论，不继承其权重、源码、配置或数据绑定。

| 资料 | 核验来源 | 实际借鉴与区别 |
| --- | --- | --- |
| Flow Matching for Generative Modeling | [论文](https://arxiv.org/abs/2210.02747) | 直线条件路径与速度回归；本项目添加训练RMS、解析高斯参考速度、局部无偏FM及血管辅助目标，不宣称是作者官方训练配置 |
| HRNet | [官方实现](https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/0bbb2880446ddff2d78f8dd7e8c4c610151d5a51/lib/models/seg_hrnet.py) | 借鉴持续高分辨率并行分支与跨尺度融合思路；独立实现3D、逐体素LayerNorm、时间调制和局部生成器，没有复制原2D网络或BatchNorm结构 |
| MedNeXt | [官方blocks](https://github.com/MIC-DKFZ/MedNeXt/blob/0b78ed869fbd1cc2fd38754d2f8519f1b72d43ba/nnunet_mednext/network_architecture/mednextv1/blocks.py)、[论文](https://arxiv.org/abs/2303.09975) | 借鉴depthwise空间混合与通道扩展；本项目并行异形/空洞核、普通3³卷积、HR融合和深宽均自行设计，不是MedNeXt官方模型 |
| Mamba | [官方v2.2.4](https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/modules/mamba_simple.py) | GPU阶段直接调用官方v1算子；只用96宽3块，小型全卷上下文。保留dt特殊初始化与A_log/D不衰减标记。CPU参考递推独立实现，仅用于小空间工程验证与参数计数，禁止CUDA替换 |
| clDice | [论文](https://openaccess.thecvf.com/content/CVPR2021/papers/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.pdf)、[官方软骨架](https://github.com/jocpae/clDice/blob/47d31a6cc4a8101b1ffe8052994821961e57af9f/cldice_loss/pytorch/soft_skeleton.py) | 独立重写pooling软骨架与中心线精度/敏感度组合；本项目只监督匹配的二维毛细投影，GT骨架预计算、预测骨架保留halo，不能宣称三维拓扑保证 |
| Skeleton Recall | [官方仓库](https://github.com/MIC-DKFZ/Skeleton-Recall/tree/c7bac3c72ca7bb4f3d891235f051f0c4d07d7229)、[论文](https://arxiv.org/abs/2404.03010) | 参考将GT结构预计算、结合区域监督的思路；本项目没有复制其nnU-Net训练器或另外叠加其原Skeleton Recall公式，不能称为该损失的复现 |
| OCTA-500 | [数据集作者论文](https://arxiv.org/abs/2012.07261) | 核对标签来源、投影对象与二维/三维边界。采用既有划分导入机制，不声称用户下载版本已检查，也不把模型加工的毛细标签称为完整人工三维标注 |
| PyTorch | [checkpoint](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/utils/checkpoint.py) | 使用非重入重计算、BF16 autocast和原子状态保存；实际速度/内存仍需完整尺寸A100验收 |

读取的参考文件版本与链接另见reference_commits.json。没有将第三方框架整仓复制入本项目；第三方依赖保留其原作者版权与许可。公开审阅不等于本次替用户新增开源许可证。

## 本项目未发表的选择

以下都是本次设计：三尺度64/128/256、3次HR融合、局部精修4块；完整深度局部列；逐体素范数；96宽3块全局Mamba；训练专用冻结二维监督模型；生成投影的类别/中心线/强度联合目标；FAZ/背景与表观宽度权重；均匀无偏FM加结构偏置辅助采样；原选模加权MAE；200自然遍历、14,000更新、Heun16及所有数值权重。

它们是待检验的组合假设，不是已证明的新颖贡献或最佳参数。不是把几篇分割论文的指标直接当作生成质量证据。标准指标、分割诊断、管径表观单位和标签偏差应分开报告。
