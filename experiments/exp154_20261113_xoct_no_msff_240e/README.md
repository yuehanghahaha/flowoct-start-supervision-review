# XOCT 去除 MSFF＋CAM · exp154

父实验为已完成的完整 XOCT 复现 exp059。使用作者 UNet3D2D 入口，在相同 64/128/256/512/512/512 六尺度下用普通 3D U-Net 替换 MSFF＋CAM 骨干。3D 主干 64,638,145 参数，保留 3D LSGAN、体素 L1 以及 CDS 的三套分层投影 L1、二维 LSGAN 和 VGG19 感知损失。保留作者各判别器的更新方式。该对照测量公开实现的整体骨干替换。

与 exp155 配对的训练与验证设置见 [PROTOCOL.md](PROTOCOL.md)。

```bash
cd /docsystem/niuke/lvyuehang/Flow_oct2octa/experience/exp154_20261113_xoct_no_msff_240e
bash launch.sh
tail -f logs/train_console.log
```

会话名为 `exp154_xoct_ablation`。首次运行和中断恢复均执行 `bash launch.sh`，会话存在时会提示当前状态。查看进度可读 `runs/status.json`，全部训练及验证完成后读取 `runs/training_complete.json`。验证报告、逐病例结果和原始浮点投影在 `reports/validation/`，训练证据在 `logs/epochs/` 和 `runs/epochs/`。原最优生成器保存在 `checkpoints/best.pt`，可恢复状态在 `checkpoints/latest.pt`。
