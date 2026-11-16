# exp148 服务器启动记录

| 阶段 | 核验结果 |
| --- | --- |
| 上传 | A800 服务器新目录，18 个源码与配置文件通过 SHA-256 校验 |
| 数据 | 固定 140 训练、10 验证、50 留出，病例顺序与 exp017 一致 |
| 本地检查 | 单通道及速度目标、Euler 8 步、训练轮次和验证中断恢复通过 |
| 服务器检查 | BF16 完整裁块反向、真实训练整卷 Euler 8 步及九类指标通过，病例 10301 的缓存与源预处理逐值一致 |
| 正式训练 | 2026-11-12 16:12 开始，A800 80 GB，tmux 会话 `exp148_octstart_nocond` |
| 首轮提交 | E1 完成 200 次更新并写入 `latest.pt`，已进入 E2。病例、裁块、时间和 BF16 输出记录见 `verification/first_epoch.json` |

训练日志在 `logs/train_console.log`，当前状态在 `runs/train/status.json`。每 5 轮整卷验证，第 200 轮完成后写入 `runs/train/training_complete.json`。复现设置、恢复命令和回传文件见 [README](README.md)。
