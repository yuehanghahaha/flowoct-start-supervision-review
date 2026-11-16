# exp018＋完整 XOCT CDS · exp164

本实验从随机初始化训练200轮，每轮200次生成器更新，共40,000次。科学父实验为exp018，运行与整卷评价工具取自已验证的exp143；exp143额外的体素端点L1在这里关闭。迁移的是XOCT的三组二维监督机制，原FM生成器保持2,097,521参数，三套新增二维判别器只参与训练。

## 实验清单与差异

| 项目 | 原exp018 | 本实验exp164 |
|---|---|---|
| 生成器 | base16时间条件3D U-Net，当前状态＋OCT条件 | 逐字节继承同一实现与初始化规则 |
| 起点和采样 | OCT起点；每轮100次精确t=0、100次均匀随机t | 保持相同患者、裁块、时间随机序列 |
| 裁块和预算 | ZYX256×64×64，batch1，200×200 | 保持 |
| FM主目标 | 速度MSE | 保持，全部200次更新计算 |
| 原投影损失 | t=0时ILM–OPL最大投影L1×0.1 | 保持 |
| 新增CDS | 无 | t=0时三组分层均值投影，每组L1×10＋LSGAN×1＋VGG×1 |
| 三个CDS区域 | 无 | ILM–OPL、OPL–BM及其余深度 |
| 生成器优化器 | AdamW，lr2e-4，weight_decay0，clip1 | 保持 |
| CDS判别器 | 无 | 作者basic PatchGAN，64通道、3层、GroupNorm8，三套独立权重；合并Adam，lr1e-4，betas0.5/0.999 |
| 数值 | 历史float32、TF32关闭、确定性卷积设置 | float32、TF32两项允许、AMP/scaler关闭、benchmark开、deterministic关 |
| 生成与选模 | Euler8、Hann重叠0.5整卷；每5轮10人验证，以浅层MIP SSIM选best | 保持，平分保留较早轮次 |
| 留存 | 历史逐轮3D及多份断点 | 常驻best生成器与latest完整恢复状态；保留指标和2D证据，验证3D及时释放 |

数值设置与历史exp018不同，因此直接历史比较包含数值档位因素。完整的因果CDS比较还需同数值档位的基线。当前实验的目标是检验完整CDS迁移与原起点FM组合的行为，不预先保证效果提升。

## 训练目标与迁移边界

```text
c = OCT, y = OCTA
x_t = (1-t)c + t*y
v = G(x_t,t,c)
L_FM = mean((v-(y-c))^2)

当且仅当t=0：
  y_hat = c + v                    # 原始、有符号、不中途截断；复用同一次G前向
  L_old = L1(ILM-OPL max(y_hat), ILM-OPL max(y))
  对r∈{ILM-OPL, OPL-BM, OTHER}：
    P_r(a) = sum_z[(2*a-1)*M_r] / (sum_z[M_r]+1e-6)
    L_CDS = sum_r[10*L1(P_r(y_hat),P_r(y))
                  + LSGAN(D_r(P_r(c),P_r(y_hat)), real)
                  + VGG19_relu3_3_L1(P_r(y_hat),P_r(y))]
  L_G = L_FM + 0.1*L_old + 1.0*L_CDS
随机t>0：
  L_G = L_FM
```

每个精确t=0样本先用分离的预测投影更新三套D一次，再冻结D参数计算G损失；每轮100次D优化器更新，200轮合计20,000次，每套D均更新20,000次。VGG仅提供冻结特征，不参与优化。原生层标注不送入G，只用于损失或评分。没有训练时八步展开；完整八步自由生成用于验证。

作者代码固定为`4bf151bee591880a28556fa18e0a79aee63ade93`，104份公开文件保存在`source/author/`，逐文件哈希在`source/author_files.json`。使用作者的二维判别器、LSGAN及VGG感知实现，包括灰度扩展为RGB，保留其没有额外ImageNet均值/标准差处理的行为。

必要迁移选择已在实现前预定：FM使用[0,1]图像，CDS在取均值前仿射映射到作者的有符号图像尺度；FM端点允许越界，保留损失梯度。XOCT原模型具有tanh输出，本组合属于新的FM适配。作者10:1:1相对权重与三个区域求和直接沿用，CDS总权重1是未验证的研究设置；真实预检会记录各项损失和FM/CDS输出梯度，而不会依据测试集调权重。CDS的D采用固定lr1e-4配合本实验200轮固定学习率预算，属于调度适配。

分层沿用exp059的数据适配：原生640深度中心采样、MATLAB一基边界，浅层ILM≤z<OPL，深层OPL≤z≤BM，其余为补集。原exp018最大投影仍使用原闭区间定义。两套定义分别保留，不通过替换原损失来改变基线。

## 当前服务器启动与恢复

```bash
cd /docsystem/niuke/lvyuehang/Flow_oct2octa/experience/exp164_20261115_startproj_fullcds_200e
bash launch.sh
tail -f logs/train_console.log
```

独立tmux会话为`exp164_startproj_fullcds`。同一入口首次启动或自动恢复，已有会话不会重复启动。也可在自己的tmux内执行`environment/bin/python -B -u run.py train`，解释器的完整路径见`deployment.json`。`train`和`resume`都核验源代码、数据、运行环境及已有断点，已有状态自动恢复。

训练前按顺序执行一次：

```bash
/docsystem/niuke/lvyuehang/Flow_oct2octa/experience/exp059_20261102_xoct_paperlr_300e/environment/bin/python -B -u checks.py all
bash launch.sh
```

合成检查只用合成数据；真实预检只用固定训练患者，核验完整尺寸裁块、九项CDS损失、梯度、D/G更新、原始初始化和完整256×304×304 Euler8。预检模型不迁入正式训练。启动器不停止已有实验，不修改其代码和断点。

## 输出、验证与测试

每5轮在原固定10位验证患者上跑完整分辨率Euler8，主best只按原ILM–OPL最大投影SSIM选择。同一best用于全部指标和图像。保留完整3D、304张逐Y方向B-scan、主浅层最大投影、补充深层/全深度最大与均值、分层均值投影的MAE、PSNR、SSIM。固定除255、range1，SSIM win7、uniform、样本协方差；B-scan先患者内平均，再患者等权。XOCT量化/GT范围补充指标单列，详见`metrics.PROTOCOL`，不称为论文同协议复现。

- `checkpoints/best.pt`：原验证准则选中的生成器。
- `checkpoints/latest.pt`：G、三套D、两个优化器、RNG、输入身份及最新完整轮。
- `trained_pending_validation.pt`和`best_candidate.pt`：验证/提交中的临时事务断点，提交后清理；可恢复中断验证和best提升。
- `runs/train/epochs/`：逐轮指标；`runs/train/status.json`：状态；`training_complete.json`：预算完成证明。
- `logs/epochs/`：全部病例、裁块、时间、FM/原投影/CDS九项损失及梯度证据。
- `reports/validation/`：每次验证的指标、浮点投影、B-scan表、指定位置切片和对比图；不常驻验证3D。
- `verification/`：源代码、数学、恢复、实际GPU与完整训练病例验证。

本次授权启动200轮训练及验证，不自动打开固定50位测试患者。预算完成后可显式运行`run.py freeze`和`run.py test`测试原best，保存50份完整3D预测。测试标注独立在`test_annotations/`，仅供这一阶段使用；训练/预检继续只读共享150人标注与数据缓存。

源代码上传、预检通过、正式训练启动分别记录；当前实际状态见`verification/deployment.json`、各预检JSON及`runs/train/status.json`。不生成交付压缩包。
