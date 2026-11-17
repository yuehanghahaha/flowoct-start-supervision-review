# 固定清单与数据约定

程序不自动搜索数据目录，也不生成新划分。请导入你原来使用的固定140/10/50成员，保持患者不跨组。当前交付没有读取、复制或绑定旧实验的成员文件；正式启动前必须核对导入成员与历史固定划分一致。

CSV采用UTF-8，每位患者一行，需有以下字段：

| 字段 | 内容 |
| --- | --- |
| case_id | 唯一字母数字标识；用于输出目录，不包含姓名 |
| patient_id | 真实的去标识患者分组键；本3M协议一人一例，共200位不同患者 |
| split | train / validation / test，分别140/10/50 |
| npz | 相对清单或绝对路径；包含oct、octa数组，均ZYX=256×304×304，存储范围0–255 |
| axis_order | 必须明确为ZYX；程序不猜测轴序 |
| layers | .npy或.mat；若为.mat键名Layer；必须已经对齐为6×Y×X、native640、MATLAB一基坐标 |
| capillary | 二维二值毛细血管标注；可包含大血管，程序显式减去large并记录重叠数 |
| large | 优先使用GT_LargeVessel；若由已明确编码的artery∪vein合成，需记录它可能与独立大血管标注不同 |
| faz | 二维二值FAZ标注 |
| annotation_orientation | YX或XY；XY明确转置后才和体数据匹配 |

二值文件支持.npy或PNG等Pillow可读图像，取值仅允许0/1/255。RGB或多类别CAVF不能直接当二值图；先检查你下载版本的编码，用tools/decode_labels.py显式解码。该工具要求完整类别映射，拒绝未识别颜色/索引，不猜某个数值代表毛细血管。

若直接适配单独的GT_Capillary/GT_LargeVessel/GT_FAZ文件，可用tools/decode_binary_mask.py，显式指定前景、背景的灰度或RGB编码。任何未映射值会报错。程序不预先声称已经识别你实际文件的颜色含义。

```text
case_id,patient_id,split,npz,layers,capillary,large,faz,annotation_orientation,axis_order
CASE001,PATIENT001,train,pairs/CASE001.npz,layers/CASE001.mat,masks/CASE001/capillary.npy,masks/CASE001/large.npy,masks/CASE001/faz.npy,YX,ZYX
```

这里只展示格式，不提供另造的患者划分。数组可为uint8或存储0–255的float；float数组必须在NPZ中明确写入intensity_scale=255，代码拒绝猜测浮点强度单位。已经0–1的数据不能虚报为255。输入转换工具及原始标签的版本、方向变换和来源应在用户数据准备记录中注明。

## 投影对齐

标注必须对应OCTA的ILM–OPL最大投影。层边界索引依次为ILM/IPL/OPL/ISOS/RPE/BM。
输出体素中心先映射回原生640深度，使用闭区间纳入射线样本。此离散选择是本项目明确声明的实现选择，不声称论文规定了相同的全部离散细节。

当前Z256必须覆盖原生640的完整轴向范围并采用中心对齐重采样。如果曾经裁切轴向范围，不能直接套用原native640边界，须先按实际变换修正适配器和几何记录。本次没有读取旧预处理代码，所以不宣称已核验该前提。

prepare生成train/validation的annotation_audit叠图：绿色毛细、红色大血管、蓝色FAZ。正方形图像无法仅凭shape判断转置，因此必须核查这些图。应确认血管、FAZ和层区均对齐后才进行正式训练。test只在冻结最终best后读取。

第一次绑定会保存split_lock.json和manifest SHA256；后续清单或成员变化会拒绝继续。每次进程首次使用的图像、层边界及标签还会计算SHA256，防止续训悄悄换数据。

## 几何与物理单位

默认不写入未知的物理间距。若预处理发生轴向裁切/重采样，应从真实处理记录推导spacing，并在单独实验配置中声明；不能直接套用原始采集的像素间距。本项目像素宽度分组只表达标签上的表观宽度。

Linux读写路径经过真实解析并限制在当前用户持久化home内。对本用户的HPC，操作者还须按既有约束核实账号、/home与/workspace映射、授权slot。代码不会访问公共/datasets或其他用户目录，也不申请GPU或安装环境。
