# exp059：XOCT，论文学习率，固定项目划分，300轮

用户于2026-11-02授权按已审阅的论文/公开代码差异清单部署并启动训练。父方法为公开XOCT代码 `4bf151bee591880a28556fa18e0a79aee63ade93`，数据来源为本项目既有3M固定划分及原始层标注；不继承其他实验模型权重，不改动其他实验。

上传、验收、训练启动分别以 `verification/delivery.json`、`verification/preflight.json`、`verification/training_launch.json` 为证据。仅上传不代表开训。动态状态见 `runs/status.json`，完成以 `runs/training_complete.json` 为准。

2026-11-02已完成独立环境、450个数据文件哈希和原尺寸GPU验收，并在tmux `xoct_exp059`启动正式训练。具体启动快照、实测资源及证据入口见 [DELIVERY.md](DELIVERY.md)。此状态不表示300轮训练或正式测试已完成。

配置见 `experiment.json`，完整逐项差异见 `provenance/protocol_decisions.json`。公开作者源码独立保存在 `source/author/`，104个文件按固定提交的原始Git blob字节保存，未改动模型、损失或数据加载器；SHA256/Git blob清单分别见 `provenance/author_files.json`、`author_git_blobs.json`。运行包装器调用作者的数据加载、裁块、模型、D/G优化、学习率调度和窗口融合方法。论文副本保存在 `provenance/paper.pdf`。

## 训练协议

- 3M固定140训练／10验证／50测试，患者身份继承现有项目，所有测试病例保持原归属。训练和验收只使用训练/验证数据，GPU短程验收只用训练病例；不下载或使用作者已训练XOCT权重。
- 论文§4.1明确学习率1e-4，覆盖公开代码默认2e-4；其余保留作者优化方法：Adam，beta1=0.5、beta2=0.999及默认epsilon/weight decay；配置200+100轮线性策略，保留每轮训练前调用scheduler的时机及原警告。由于这一调用顺序，实际第1–199轮为1e-4，第200轮为1e-4×100/101，第300轮为0；最后一轮仍照作者执行Adam迭代和评价，不人为改成非零学习率。原环境验算记录见 `verification/scheduler_audit.json`，正式训练逐轮记录实际值。
- 正式300轮，每轮140位训练患者各一个随机裁块，batch1，共42000次D/G更新迭代（各执行一次D与G优化）。ZYX=256×96×96，全部深度，无翻转，4个DataLoader worker；未缩短预算或降低分辨率。
- 原base64/max512、GroupNorm、add_scale、MSFF分组5³卷积和辅助二维特征分支保留。对抗损失为作者LSGAN，记录其与论文式(4)对数GAN的差异，不替换公式；3D/2D L1权重均10，VGG19 relu3_3感知L1权重1，保留未应用ImageNet均值/标准差的作者行为，亦未修写D_other的原梯度切换行为。
- Python/NumPy/PyTorch/CUDA固定seed2026，补充可控种子，替代作者入口随机种子；不强制确定性卷积。模型/输入/梯度/Adam及推理float32，AMP、GradScaler、cuBLAS/cuDNN TF32关闭，保留cuDNN benchmark。严格精度是项目补充，不声称作者设置了TF32关闭。CPU数学线程固定8是运行配置。

## 数据与层标注

原始来源为 `OCTA500_common_v1/split_3M_z256.csv` 指定的ZYX=256×304×304、float32、0–255存储单位NPZ。训练/验证150例的OCT/OCTA分别逐值写入本实验 `data/OCT2OCTA3M_3D/A/`、`B/` 的NPY，不经现有0–1缓存反向还原，不再重采样。作者加载器仍先转float64执行 `/255*2-1`、同位置裁块，再返回float32。

作者仅提供现成三类NPY标签的读取代码，未公开原始层边界生成这些标签的完整流程。本实验明确采用未发表的必要适配：原始MAT `Layer[6,Y,X]`、native640、MATLAB一基坐标；256层对应采样中心 `z=(i+0.5)*640/256-0.5`，边界先减1；ILM≤z<OPL为1，OPL≤z≤BM为2，其他为0；OPL只归2，避免类别重叠。横向不转置，作者按0/1/2生成one-hot。原始MAT、转换结果及原始NPZ身份哈希留存。这一边界约定可能不同于作者内部预处理，不能声称逐像素同协议。

`prepare.py`只转换140训练+10验证，不打开测试影像，不创建测试数据清单或测试派生文件。50位测试患者仅保留在冻结身份表中；测试准备只能在预算完成、检查点冻结后独立安排。

## 验证、选模和整卷补充

作者验证每轮执行一次：10例中心256×96×96裁块，按每例GT范围归一化，计算ILM–OPL、OPL–BM、全深度均值投影SSIM的平均，再患者等权平均。原始 `epoch>200` 的best准入规则保留；前200轮不选best。主正式测试权重预先固定最终第300轮；作者规则best仅为预先声明的补充，测试不参与选择。

另每5轮及最终轮执行全部10例、完整256×304×304评价。推理调用作者16窗口（4×4，横向96×96、保留全部深度），按覆盖次数平均，不采用Hann或本项目其他网络的窗口策略。补充评价保存并恢复主RNG，绝不用于改变主权重或作者best选择。

作者参考口径：预测从[-1,1]映射到[0,255]并截断后uint8量化；计算指标前转浮点。每患者/每评价对象用GT min/max共同映射到0–255，预测截断，PSNR/SSIM `data_range`为归一化GT范围，MAE单位0–255。体积SSIM直接三维计算，投影SSIM二维计算。full为ILM–BM层内加权均值，mean为全深度均值；ILM–OPL和OPL–BM均值投影另外报告，保持作者工具定义。Perp.复用作者VGG19。

项目固定范围口径另名报告：未量化float32预测对固定GT/255、`data_range=1`，保留MAE/MSE/PSNR/三维SSIM和对应二维投影指标。ILM–OPL、OPL–BM最大投影只列为诊断。两套结果不混排，每例先算，再患者内平均及患者等权宏平均；目前一患者一例。

论文表1有三维Perp.，公开评价器未实现其切片方向或聚合规则，本实验明确列为缺失，不能声称完整复现该列。常数GT或无限指标标为null并说明，不编造有限数值。

## 留存与恢复

每轮保存完整 `checkpoints/latest.pt`，第50/100/150/200/250/300轮保存完整编号断点，另保留完整initial和作者规则best；仍按作者原频率保存兼容的网络权重。完整断点含全部5个网络、两个Adam、scheduler、Python/NumPy/PyTorch/CUDA RNG、完整轮次/步数、best、源码/配置/输入身份。恢复以最后完整轮为边界，未完成轮的attempt日志保留；中断的整卷补充可从同一完整断点补做。

每轮损失与作者验证指标、每5轮整卷指标、原始二维投影、病例PNG和训练曲线保留。不永久保存任何中间验证轮（含best）的三维预测，也不创建临时三维预测文件；完成病例指标后释放三维数组。正式测试将另行保存完整float32三维预测及作者量化版本。只承诺从仍保留的断点再生对应输出，不承诺任意历史轮次。

验收临时断点完成模型/Adam/scheduler/RNG恢复核对后删除，保留哈希和清理记录；训练checkpoint不删除。不生成交付ZIP/tar，不修改共享数据、已有环境或其他实验。

## 环境与运行

目标 `eias-member.eitech.edu.cn:42060`，A800 80GB。独立环境在本实验 `environment/`，Python3.11、PyTorch2.7.0+cu126、torchvision0.22.0+cu126，按作者requirements安装。仅为兼容visdom0.2.4构建补充setuptools80.9.0并关闭构建隔离，原失败日志保留，参考[setuptools的pkg_resources移除说明](https://setuptools.pypa.io/en/latest/history.html)。预训练VGG19来自官方PyTorch，完整哈希见 `verification/vgg19.json`，不使用预训练XOCT。

```bash
cd /docsystem/niuke/lvyuehang/Flow_oct2octa/experience/exp059_20261102_xoct_paperlr_300e
bash START.sh status
# 首次训练只在独立验收通过、确认未开跑后执行一次
bash START.sh train
# 训练中断后，从最后完整轮恢复，禁止再次使用train覆盖
bash START.sh resume
tail -f logs/formal_train.log
```

正式训练在tmux `xoct_exp059`内运行；安装、数据转换、GPU验收使用分开的短期会话。完整环境记录和pip清单放在verification中。本地可运行 `python control.py fetch`直接回传日志、指标和PNG；权重、环境和大数组留服务器。源码同步使用 `control.py upload`，正式启动后拒绝覆盖源码。
