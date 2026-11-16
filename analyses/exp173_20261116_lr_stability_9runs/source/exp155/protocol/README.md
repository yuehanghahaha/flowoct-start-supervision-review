# XOCT 去除 CDS · exp155

父实验为已完成的完整 XOCT 复现 exp059。使用作者 UNet3DMSFF 入口，保留原 12 处 MSFF＋CAM、残差比例和 64/128/256/512/512/512 六尺度，3D 生成器为 52,615,169 参数。训练保留 3D LSGAN 与体素 L1。CDS 的三套分层投影监督、二维判别器和 VGG19 训练感知损失关闭。层标注和 VGG19 用于统一验证计算。

与 exp154 配对的训练与验证设置见 [PROTOCOL.md](PROTOCOL.md)。

```bash
cd /docsystem/niuke/lvyuehang/Flow_oct2octa/experience/exp155_20261113_xoct_no_cds_240e
bash launch.sh
tail -f logs/train_console.log
```

会话名为 `exp155_xoct_ablation`。首次运行和中断恢复均执行 `bash launch.sh`，会话存在时会提示当前状态。查看进度可读 `runs/status.json`，全部训练及验证完成后读取 `runs/training_complete.json`。验证报告、逐病例结果和原始浮点投影在 `reports/validation/`，训练证据在 `logs/epochs/` 和 `runs/epochs/`。原最优生成器保存在 `checkpoints/best.pt`，可恢复状态在 `checkpoints/latest.pt`。
