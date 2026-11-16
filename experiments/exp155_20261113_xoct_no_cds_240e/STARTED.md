# exp155 A800 启动记录

2026-11-14 00:17（UTC+8）创建独立 tmux 会话 exp155_xoct_ablation。

| 阶段 | 核验结果 |
| --- | --- |
| 上传 | 本实验包装器、配置与作者固定提交文件通过逐文件 SHA-256 |
| 真实预检 | 7 个训练病例裁块，前向、反向和网络更新通过，裁块值与作者加载器一致 |
| 完整推理 | 一个训练病例的 256×304×304 输出，九类标准指标与作者补充指标通过 |
| 显存 | 预检训练 reserved 峰值 34.17 GiB |
| 正式训练 | PID 2216539，抽查首轮 29 次成功更新，全部病例和裁块与配对记录一致 |
| 预算 | 240 轮，每轮 140 步，共 33,600 次更新 |
| 验证 | E30、E60、E90、E120、E150、E180、E210、E240，各 10 人完整体积 |

实时进度读 runs/status.json 和 logs/train_console.log。240 轮及八次验证均提交后写入 runs/training_complete.json。训练会话存在时，bash launch.sh 显示当前会话。中断后同一命令从最近完整断点恢复。

实际启动环境和更新抽查在 [training_confirmed.json](verification/training_confirmed.json)，训练及推理检查在 [preflight.json](verification/preflight.json)。
