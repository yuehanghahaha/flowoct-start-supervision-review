# 当前起点监督模型：固定第70轮的50例测试集评价

父实验为exp001（100轮、50%精确t=0训练）与exp003（10例验证复核）。本次用户明确要求为当前模型补跑50例测试并生成新报告。只评价验证集已选定的第70轮，不训练、不重新选轮次、步数、阈值或模型。旧FlowOCT/旧3DFM的测试结果不会冒用为本次结果。

沿用原FP32、Euler8、ZYX=256×304×304、体块256×64×64、50%重叠逐步融合速度、仅最终截断到[0,1]。不加入投影损失或修改模型。每个滑窗独立运行同一网络；批量组织多个窗口仅为加速，先在原验证病例10302上与exp003原始推理核对，再锁定运行方式；不使用任何测试病例做该核对。

服务器3090当前仍报告Reboot。将使用可用的本机CUDA和服务器CPU分工，各病例保持同一科学设置，记录设备。所有日志、权重、原始投影、图表、报告和核对记录保存在本目录。保存小型投影和指定B-scan，不重复保存大型完整三维预测，不制作交付压缩包。

## 文件与运行

- `report/index.html`、`report/test_analysis.md`：完整图文报告；50例完整推理前不生成最终汇总。
- `results/`：50例指标与二维原始数值；`logs/`：执行记录；`verification/`：冻结配置、来源与核对；`evidence/`：验证参照和调度历史。
- `checkpoints/best70.pt`：同一权重的无损推理版本。原训练文件SHA256与此提取文件SHA256分别记录，模型未变。
- `fm3d/`：原始冻结模块；`evaluation_metrics.py`：逐字继承exp003的评价函数；`inference_batched.py`：独立窗口的批量计算封装。

本机推理使用 `D:\software\anaconda3\envs\flowoct\python.exe`（PyTorch2.5.1+cu121），报告使用 `D:\software\anaconda3\python.exe`，需要NumPy、SciPy、scikit-image、Pillow、Matplotlib和Markdown。数据目录只读引用，不复制大型数据集。

```powershell
# 完整冻结协议/已分配GPU病例。已完成病例先核对哈希后复用。
& 'D:\software\anaconda3\envs\flowoct\python.exe' run_test.py --data 'D:\data\code\keyan_fuxian\FlowOCT\data\OCTA500_common_v1' --layers 'D:\data\code\keyan_fuxian\FlowOCT\data\OCTA500_common_v1\labels_native\GT_Layers' --role gpu

# 汇齐50例后，按顺序生成全部图文报告与数值核对。
& 'D:\software\anaconda3\python.exe' aggregate.py
& 'D:\software\anaconda3\python.exe' build_report.py
& 'D:\software\anaconda3\python.exe' verify_report.py
```

服务器CPU同一入口增加 `--role cpu --data /docsystem/niuke/lvyuehang/OCTA500_common_v1 --assets /docsystem/niuke/lvyuehang/fm3d_ubuntu/runs/fm3d_projection_assets_test_20261020`。层数组与本地原MAT数值哈希一致；只在正式测试推理中读取测试图像。PyTorch环境为 `/docsystem/niuke/lvyuehang/conda_envs/lyh_fm3d/bin/python`。

## 冻结与恢复

`verification/frozen_contract.json`锁定模型、推理源文件、设置、患者分组及输入身份；运行时逐一核对。仅在原验证病例10302上核对批量模式：最大指标差2.56e-6、最大投影像素差1.64e-6，均低于预设1e-4；之后开始50例测试。

本地运行曾出现长时间等待，原CPU16例分工已完成。2026-10-26恢复时只将9例未完成病例重新分配到空闲CPU，最后GPU/CPU各25例，科学设置与冻结源文件未变；原调度、原执行记录和新分工分别保存。逐病例seconds保留原值，包含暂停或等待，不能当纯计算耗时。

## 已完成结果

50/50位测试患者已完成：体积MAE=0.0154577，PSNR=29.9713dB，浅层SSIM=0.4074248（患者bootstrap95%区间0.394665–0.420331），PCC=0.6595182。全部50例原始投影指标复算、哈希、患者划分、误差分解及58张图/64个报告链接核对通过。模型原始SHA256未变，未增加训练或测试集调参。

报告包含5页全部病例总览、50张逐例完整图和3张统计图。浅层信号通常偏弱、细网仍平滑；42例浅层均值不高于GT，8例高于GT。全体积均值却可能偏高，不能用单一全局提亮解释或解决细节问题。细血管分割准确性未在本次评价中测量。

## 服务器同步状态

2026-10-26 19:07（UTC+8）SSH曾连接超时，同步中断。20:48以后连接恢复；20:56已补齐全部缺失结果，服务器端所有原始交付文件SHA256与本地一致，50例保存结果独立复算、58张图和64个报告链接核对通过，见verification/server_delivery_20261026_2055.json。完成后清理11个可再生字节码文件，保留所有科学证据与正式权重。

连接恢复不表示GPU计算恢复：20:50两张3090仍在cuInit报CUDA错误999，PyTorch分别25秒超时，驱动恢复动作仍为Reboot，UVM打开报errno5；完整诊断保存在实验007的verification/server_health_20261026_2050.json。本次未执行重启、GPU重置或新的正式训练。
