# 复核与离线复算

使用已安装Python环境，先从仓库根运行 `python verify_snapshot.py`。这校验发布时文件指纹、二维数组形状、范围与各组数据覆盖；不要在运行训练代码后仍期待生成文件的时间戳/哈希不变。

离线分析依赖numpy、pandas、matplotlib、Pillow、scikit-image；具体已测版本见两份分析的verification/analysis_environment.json。无需PyTorch、CUDA或原始三维数据。

```text
cd analyses/exp173_20261116_lr_stability_9runs
python -B analyze.py
python -B plots.py
python -B report.py
python -B verify_report.py
```

```text
cd ../exp174_20261116_fm_stability_diagnosis_0e
python -B analyze.py
python -B report.py
python -B verify.py
```

本快照省略只负责原作者Obsidian写入的脚本和记录。`collect.py`保留原来源采集逻辑，需要原私有工作目录/SSH环境，离线复算不运行它。报告重新生成后本机路径/时间戳可能变化，数值应保持一致；公开版本仅把Markdown链接改成仓库相对链接，科学代码保持原字节。

训练源码属于不同历史环境，分别看各实验requirements、协议和运行记录。原始OCTA-500数据及教师权重须依原作者合法来源准备；本仓库没有执行依赖安装、重新训练或模型推理。


## exp182 A–F数值复算

在 analyses/exp182_20261117_fm_stability_validation10_0e 运行 python -B analyze_af.py，读取50条完成记录，核对5,400项标准值，生成汇总和判断。已有NumPy环境即可，执行离线分析。


## 2026年11月18日G/H补充

在 analyses/exp182_20261117_fm_stability_validation10_0e 运行 python -B verify_gh_data.py，复核320条求解器、840条扰动与对照、10,440行逐患者标准指标及配对统计。已有NumPy环境即可；不运行模型。
