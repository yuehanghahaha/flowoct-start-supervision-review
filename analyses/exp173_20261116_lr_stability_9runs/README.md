# exp173：学习率与指标稳定性分析

2026年11月16日。仅分析既有训练和验证记录，新训练/推理为0。

[打开完整报告](report/index.html) · [详细文字报告](report/report.md) · [波动统计](data/stability_summary.csv) · [真实学习率](data/learning_rates.csv)

确实存在跨轮指标波动，主要集中在exp018、exp148、exp163、exp164的完整3D体积，以及exp155的ILM–OPL最大投影。XOCT完整模型和exp154的已观测整卷点较稳。现有记录没有只改变学习率的配对对照，不能认定学习率是主要原因。

## 范围与科学设置

主实验为XOCT exp059、TransPro exp035、exp018，加按编号倒序、在2026年11月16日21:13前后快照已完成原预算的exp164、exp163、exp155、exp154、exp148。另将不同训练/验证协议的TransPro exp040独立列作补充。分析/测试编号不重复算作完成训练；未完成与未启动的组不计入最近五个。

保留原科学设置、训练预算、精度、选模和验证频率。所有主best由原准则复核：XOCT E213、TransPro035 E194、exp018 E170、exp148 E130、exp154/155 E180、exp163 E190、exp164 E150、TransPro040 E83。XOCT原交付主权重E300作为最终轮补充，E213缺同期完整验证记录，不借邻近轮代替。

完整标准对象包括3D体积、逐B-scan、主ILM–OPL最大投影、OPL–BM最大、全深度均值/最大和三种层内均值，分别给出MAE/PSNR/SSIM及缺失项。TransPro原生随机裁块MAE、原Gaussian11逐B-scan协议和XOCT原中心裁块投影评分分开保留。没有读取测试数组或据测试结果判断学习率。

## 复算与证据

使用已安装的Python、numpy、pandas、matplotlib，从本目录快照复算：

```text
python -B analyze.py
python -B plots.py
python -B report.py
python -B verify_report.py
```

`collect.py`从既有本地证据目录和现有SSH连接收集小型JSON/代码/配置；不下载3D数组和权重，不修改服务器。重复采集会更新快照时间，查看本次结果无需重新运行收集。

- `source/`：逐轮历史、验证JSON及原协议代码，来源和SHA-256见`provenance.json`。
- `data/`：所有标准/原生逐轮指标、学习率、训练损失、统计、best/final与逐患者回落配对。
- `report/`：离线HTML、Markdown、PNG及可缩放SVG，不依赖外部网页库。
- `verification/`：两台服务器快照、数值审计、文件与登记记录。

完整验证记录236次；标准指标5,532个；完整训练轮1,980条；所有标准聚合值按同10位患者重新核对。每30轮验证的消融仅有4个后半程点，不足以排除短时波动。各模型网络、损失、精度与推理不同，跨模型稳定性比较为描述性结果。

## 交付状态

本地分析与报告完成。服务器仅只读收集，没有上传报告、启动/暂停训练、模型推理、申请/释放GPU或变更环境。没有新建训练对照，没有默认ZIP/tar。相关比较结论归入已有科学实验的Obsidian笔记，不新建一个虚构训练实验条目。
