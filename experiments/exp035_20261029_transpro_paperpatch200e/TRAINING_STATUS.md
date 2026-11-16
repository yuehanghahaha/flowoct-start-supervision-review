# 正式训练启动记录

用户于2026-10-29明确授权后，本版已在服务器开始正式200轮训练。此记录更新交付说明中“验收完成、尚未开训”的历史状态，科学源码及配置没有改变。

- 启动时间：2026-10-29 15:36:23，北京时间。
- 服务器：eias-member.eitech.edu.cn:42060；GPU为A800 80GB。
- tmux会话：`transpro_exp035`；主训练进程PID=26673。
- 15:38现场确认：已进入第1/200轮、完整轮次0；训练进程及4个数据加载worker存在，GPU利用率100%。
- 代码/数据协议指纹与已通过的GPU验收一致；此次没有修改科学代码或参数。

最近观察见`verification/training_launch.json`，实时日志在`logs/train_console.log`，实际轮次以服务器`runs/status.json`为准。首轮有cuDNN算法测速，作者默认每100次训练更新才打印一次损失，模型初始化结束后日志暂时没有新行不等同于训练停止。

进入会话：`tmux attach -t transpro_exp035`。实时看日志：

```bash
tail -n 50 -F /docsystem/niuke/lvyuehang/Flow_oct2octa/experience/exp035_20261029_transpro_paperpatch200e/logs/train_console.log
```

退出日志查看用Ctrl+C；从tmux分离用Ctrl+B再D。训练会继续。
