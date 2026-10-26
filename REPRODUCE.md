# 复算与源码范围

## 本项目可直接运行的分析

```bash
python -m pip install -r requirements.txt
python analyze_scales.py
python verify_share.py
```

支持Python3.10及以上；本次实际环境版本记录在`verification/runtime.json`。入口只读取`derived/ilm_opl/`的50对二维投影，复算亮度分区、Gaussian低/高频统计和两张补充图。Gaussian只用于诊断，不改变原测试预测或原58张图。

`experiment.json`固定强度分区、所有尺度、病例数和展示示例。分区和尺度在新增统计运行前写入，所有结果都报告，没有按测试效果择优挑选设置。它们是事后描述性工具，并未用于选择模型或训练超参数。

## 派生数据来源

`prepare_review_data.py --source /path/to/exp010_20261025_3dfm_start70_test50`从父实验已保存的`results/case_views/`提取原始浮点`gt_ilm_opl`和`pred_ilm_opl`，没有读取三维体数据。每例原文件SHA256、二维数组SHA256、共享文件SHA256保存在`verification/derived_data.json`。现有二维数据已包含在仓库中，复算无需父实验目录。

这些文件是报告投影的数值证据，不是原始OCT/OCTA三维体，也不是额外的交付压缩包。

## 原训练和推理源码

`source/training/`是exp001训练源码和配置副本；`source/evaluation/`是exp010固定第70轮评价源码和配置副本。逐文件来源及SHA256见`verification/source_manifest.json`。源代码/原报告副本保持原样，其中部分路径和历史执行说明描述的是原实验环境，应结合本说明阅读。

完整重新训练/推理需要独立取得OCTA-500数据、原固定划分、层标注和相应权重。这些资源未随代码发布；本次也未重新训练。当前公开的50对浅层投影可以复算新增分析，但不足以重新计算全部三维误差或生成完整三维OCTA。

训练配置为100轮、每轮200次成功更新、50%精确t=0、batch1、原256×64×64体块、AMP训练。评价为FP32、Euler8、50%重叠、逐步Hann速度融合、最终截断到[0,1]。不能把AMP训练误写为全程FP32，也不能把验证集选择用的投影SSIM误写为训练投影损失。

权重身份：原训练best70文件SHA256为`69c3f984e86b3dc702822cfb35a492d412f8116262b2f6e31a555792071798c1`；父实验提取的推理权重文件SHA256为`28a31cb177dbc91b0ecc293951abdf2c47429884ea3c04c9c70c2dfaa8e112d2`。两种文件格式不同，模型张量来源相同。

原报告的50例推理采用本机GPU与服务器CPU各25例完成；GPU/CPU不是逐位等价。窗口批量加速先仅在原验证病例10302核对，测试病例没有用于该核对。见原报告的推理来源和核对部分。
