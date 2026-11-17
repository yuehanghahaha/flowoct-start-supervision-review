# OCT→OCTA：学习率、指标波动与FM强度漂移审阅快照

本分支集中发布这次讨论的9组代码及其保留训练/验证结果，以及exp173/174的分析。用于他人独立检查实现、指标波动和可能原因。

**先看 [REVIEW_GUIDE.md](REVIEW_GUIDE.md)**，再核对原始数据。分析中的“上游原因”仍是待检验假设，不是已经由单变量对照证明的结论。

## 直接阅读

- [全实验学习率与稳定性分析](analyses/exp173_20261116_lr_stability_9runs/report/report.md)：完整3D、逐B-scan、主及补充投影MAE/PSNR/SSIM，原best与最终轮分开。
- [强度漂移原因诊断](analyses/exp174_20261116_fm_stability_diagnosis_0e/report/diagnosis.md)：60对原验证投影、代码机制、已证实事实与推断的边界。
- [全程曲线](analyses/exp173_20261116_lr_stability_9runs/report/figures/01_stability_overview.png) · [实际LR与原生协议曲线](analyses/exp173_20261116_lr_stability_9runs/report/figures/02_learning_rates_native_metrics.png) · [同窗强度变化](analyses/exp174_20261116_fm_stability_diagnosis_0e/report/figures/02_same_window_validation_patient.png)。
- [逐轮标准指标](analyses/exp173_20261116_lr_stability_9runs/data/validation_metrics.csv) · [真实学习率](analyses/exp173_20261116_lr_stability_9runs/data/learning_rates.csv) · [逐患者强度数据](analyses/exp174_20261116_fm_stability_diagnosis_0e/data/validation_intensity_by_patient.csv)。

## 实验清单

| 编号 | 完整科学代码 | 原预算 | 原best | 已保留记录 |
|---|---|---|---|---|
| exp018 | [exp018_20261027_3dfm_start_zproj_200e](experiments/exp018_20261027_3dfm_start_zproj_200e/) | 200×200 | E170 | 40次完整验证 |
| exp035 | [exp035_20261029_transpro_paperpatch200e](experiments/exp035_20261029_transpro_paperpatch200e/) | 200×140 | E194 | 200次原随机裁块MAE；原best完整B-scan补充 |
| exp040 | [exp040_20261029_transpro_tf32bench200e](experiments/exp040_20261029_transpro_tf32bench200e/) | 200×140 | E83 | 200次原完整B-scan验证（补充版本） |
| exp059 | [exp059_20261102_xoct_paperlr_300e](experiments/exp059_20261102_xoct_paperlr_300e/) | 300×140 | E213 | 300次中心裁块评分＋60次完整验证 |
| exp148 | [exp148_20261112_octstart_nocond_200e](experiments/exp148_20261112_octstart_nocond_200e/) | 200×200 | E130 | 40次完整验证 |
| exp154 | [exp154_20261113_xoct_no_msff_240e](experiments/exp154_20261113_xoct_no_msff_240e/) | 240×140 | E180 | 8次完整验证 |
| exp155 | [exp155_20261113_xoct_no_cds_240e](experiments/exp155_20261113_xoct_no_cds_240e/) | 240×140 | E180 | 8次完整验证 |
| exp163 | [exp163_20261114_startproj_layermean_a100_200e](experiments/exp163_20261114_startproj_layermean_a100_200e/) | 200×200 | E190 | 40次完整验证 |
| exp164 | [exp164_20261115_startproj_fullcds_200e](experiments/exp164_20261115_startproj_fullcds_200e/) | 200×200 | E150 | 40次完整验证 |

这里的“最近五个完整训练”冻结在2026年11月16日约21:13的exp173快照：exp164、163、155、154、148。之后的其他训练不在此表补计。TransPro035/040有不同科学与验证协议，不能当作单纯换学习率/硬件的配对实验。XOCT原交付主权重E300作为最终轮补充，本次按比较偏好以原作者准入规则best E213为主；E213缺同期每5轮完整验证，未借邻近轮代替。

## 仓库包含什么

- 9组原代码、固定划分、损失/数值/推理/选模配置与来源许可证。
- 1,980条完整训练轮记录；236次标准整卷验证、5,532个标准指标值；另有TransPro原生与XOCT中心裁块逐轮指标。
- 可用原始训练日志；exp018本地保留逐步JSONL；exp164后50轮梯度分支汇总。原始日志可能包含中断尝试，分析以已提交轮记录为准。
- exp018/163共60对原验证二维全深度均值投影及第一位验证患者的MIP证据，支持CPU复算诊断。
- `frozen_test_metrics/`：已冻结的正式测试数值作为独立补充，含MAE/PSNR/SSIM等原记录；本次不利用测试选择LR、检查点或训练方案。exp148未取得正式50人测试，保留缺口。
- 原始三维影像、模型/教师权重、完整正式测试病例图集不在本次审阅快照中；具体范围见 [数据说明](DATA_AVAILABILITY.md) 和 [未纳入清单](provenance/not_included.json)。

## 复核与复算

```text
python verify_snapshot.py
```

先验证初始快照，再按 [REPRODUCE.md](REPRODUCE.md) 运行两个分析。数值分析不需要GPU，不运行任何训练入口。原训练代码仍需要其各自的数据/环境/权重，不能与离线分析依赖混为一套环境。

当前分支是 `lr-stability-review-20261116`。同一仓库的历史 [main](https://github.com/yuehanghahaha/flowoct-start-supervision-review/tree/main) 与 [exp172审阅分支](https://github.com/yuehanghahaha/flowoct-start-supervision-review/tree/exp172-review-20261116) 保留。
