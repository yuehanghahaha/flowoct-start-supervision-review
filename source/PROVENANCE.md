# 来源与原创改动

记录日期：2026年11月16日。网络组合、归一化参考速度、全时刻辅助目标与本实验训练预算是本项目选择；
不宣称是下列作者发布的方法或已证明的新颖贡献。所有外部运行库保留原许可证，不复制其CUDA实现。

| 来源 | 核验链接/固定版本 | 借鉴与区别 |
| --- | --- | --- |
| Flow Matching for Generative Modeling | https://arxiv.org/abs/2210.02747 ; https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/path/affine.py | 直线路径与条件速度回归；本实验使用配对OCT条件、训练集逐带尺度和解析速度残差 |
| Mamba官方代码 | https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/modules/mamba_simple.py | 直接调用官方Mamba v1；保留A_log、D的float32及不衰减标记、dt初始化；BF16 autocast验收 |
| PyTorch AMP | https://pytorch.org/docs/2.5/amp.html | autocast仅包围网络；状态与损失显式float32；不使用BF16 GradScaler |
| SegMamba | https://arxiv.org/abs/2401.13560 ; https://github.com/ge-xing/SegMamba/tree/cff35970e0c542ad940b5701267f1ac888298b06 | 借鉴医学3D中的卷积与状态空间组合；没有复制其BiMamba或其分割头/损失 |
| MedNeXt | https://arxiv.org/abs/2303.09975 ; https://github.com/MIC-DKFZ/MedNeXt/blob/0b78ed869fbd1cc2fd38754d2f8519f1b72d43ba/nnunet_mednext/network_architecture/mednextv1/blocks.py | 借鉴depthwise卷积与通道扩展的3D残差块；本实验通道、深度、shift/scale、Haar输入及目标自行设置 |
| DiT | https://arxiv.org/abs/2212.09748 ; https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py | 借鉴AdaLN-Zero与时间嵌入；使用轴向共享双向Mamba替代自注意力，384×12为本项目设置 |
| FlowLet | https://arxiv.org/abs/2601.05212 ; https://github.com/sisinflab/FlowLet/blob/0741f4233642d93c6ec69b16bdf4262561f6a744/flowlet/wavelets/transforms.py | 借鉴小波坐标中的流建模；本项目单层Haar实现从exp131继承，无FlowLet框架代码复制 |
| FlowOCT | https://github.com/cnu-medilab/FlowOCT/tree/2309126956d2b80a95f2d307d85a50799cc658a6 | OCT到OCTA条件流与医学投影监督的任务背景；未复制其训练设置、声称其指标或等价官方复现 |
| exp131/exp112/exp018 | 本项目保留的独立目录及本实验provenance.json父文件哈希 | 继承固定划分、Haar、27指标、native640投影、可恢复训练及结果交付框架；科学改动全部在新exp172中 |

解析参考速度来自直线高斯插值的边际速度公式：
q=(1−t)²+t²；v_ref=(2t−1)zt/q。本文独立列明代数恒等关系，不宣称首创这一高斯公式。
训练集真实目标并非高斯，解析分量只是显式可学习残差基准，不能据此证明不存在条件捷径。
归一化、t=0占比25%、全t辅助损失、损失权重、batch2、40k更新、Heun16与诊断流程均为未发表的本项目设计选择。
研究效果需要保留的验证曲线和最终原best正式测试支持。

运行补充：仅用户home/workspace持久化目录；既有lyh_flowoct与lyh_tmux；单slot2；
BF16 autocast按本次用户指令覆盖草案AMP关闭设置；激活重计算与物理batch由完整尺寸验收决定，
有效batch/损失/数据/40k科学预算不变。
