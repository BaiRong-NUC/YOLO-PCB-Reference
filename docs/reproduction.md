# 复现卡

本次逐篇复现从 [ChangeChip 的中文框架图与复现结果](papers/changechip/README.md) 开始。下表保留已经核验的候选与公开工程，分别标明实际完成的范围。

## 同任务直接竞争方法

| 工作与原文位置 | 作者给出的条件和指标 | 本机核验与证据等级 |
|---|---|---|
| He 等，ESANN 2020，§4 表 1 `ours-MP` | 原版 DeepPCB，作者划分 1,000/500，6 类；mAP 98.6%，62 FPS。实验采用 500 轮、Adam、batch 16、Titan X。 | **文献报告**。`tangsanli5201/DeepPCB@08e98c4db5922613fb97176eb3d6497d48260cb1` 有配对图、标注、工具、计分脚本与 `gt.zip`；未找到该网络训练入口或权重。实际复跑范围是标注与计分器，真值回灌 500 图，P/R/F1/AP 均为 1。 |
| Jiang 等，IEEE TCPMT 2025，DOI `10.1109/TCPMT.2025.3543396` | PIDDN，使用模板与待检图，作者摘要报告 DeepPCB 上 99.6% mAP。没有取得全文表号与统一 IoU 定义，这个数字不参与本地排名。 | **文献报告**。`QixingJiang/PIDDN@cb78fd2ff33243fa35fa541c9a6f11d9eb574966` 提供基于 MMDetection 2.23.0 的训练和评测代码，有单图、双图特征相减、完整方法三种配置。`sh tools/train_3090.sh` 和 `sh tools/test_3090.sh` 是作者入口。README 仍将 DeepPCB-mulframe 标注、模型及训练日志列为待发布项。在所核提交未找到相应权重或 JSON 标注。 |
| Deng 等，EEDD，§2 与 §3；Saiyod 等，Sensors 2026，方法节 | EEDD 在成像与对齐条件有变化的 DeepPCB-A 上做特征点配准和边缘异常分析。Saiyod 等使用整体相似度判断是否接受配准。 | **文献报告**。这些工作界定本项目的先例。Saiyod 作者仓 `nonsakhoo/PCB-defect-localization@42270a3b2226c550ad98d55c1a27d4ade867128b` 有 MATLAB 脚本 `dev/v0_14_eval2_release1.m`。本机未发现可执行 MATLAB/Octave，未将其列入实测。 |

来源见 [ESANN 表 1](https://www.esann.org/sites/default/files/proceedings/2020/ES2020-33.pdf)、[DeepPCB](https://github.com/tangsanli5201/DeepPCB/tree/08e98c4db5922613fb97176eb3d6497d48260cb1)、[PIDDN 论文](https://doi.org/10.1109/TCPMT.2025.3543396)、[PIDDN 代码](https://github.com/QixingJiang/PIDDN/tree/cb78fd2ff33243fa35fa541c9a6f11d9eb574966)、[EEDD](https://doi.org/10.3390/electronics12102306)、[Saiyod 作者实现](https://github.com/nonsakhoo/PCB-defect-localization/tree/42270a3b2226c550ad98d55c1a27d4ade867128b)。

## 可取得权重的六类检测对照

选取公开工程 `iamvu3006/PCB-YOLO@70e3991f7eaa34b4e356ab42d6fe15d4ae930984`。该工程提供预处理、训练、评测、`best.pt`、逐框预测和报告，适合作为 **权重评测复现**。作者项目 README 的 *YOLO11n Baseline → Results* 是指标出处。它是工程参考，论文直接竞争方法仍是 PIDDN 等工作。

作者训练划分为 800/200/500，种子 42；本机载入作者权重，在相同 `test.txt` 的 500 图、3,140 框上验证。六类顺序和 HBB 转换一致。本机使用既有单图 PNG 导出、CPU FP32、Ultralytics 8.4.172、640、batch 16、conf 0.001、NMS IoU 0.7。作者评测入口配置 GPU。权重 SHA-256 为 `23e511d0599f351cd115d9711b97371ccc1e8d10773e07de4590ee267901e554`。

| 结果入口 | P | R | AP50 | AP50–95 |
|---|---:|---:|---:|---:|
| 作者 README，所核提交 | 97.4% | 94.6% | 98.3% | 73.1% |
| 本机作者权重，Ultralytics 工作点与 AP | 91.55% | 90.35% | 95.01% | 68.14% |
| 作者发布的 43,972 个框，在本地真值上用 pycocotools 重算 | — | — | 94.672% | 68.017% |
| 本机预测的 43,951 个框，同一真值与 pycocotools | — | — | 94.681% | 68.000% |

本机 Ultralytics 的 AP50–95 比首页低 4.96 个百分点。相同评测器下，两份逐框结果相差 −0.017 个百分点。权重、原始预测与首页声明应分别引用。当前证据不能定位首页差异的具体原因。[作者仓库](https://github.com/iamvu3006/PCB-YOLO/tree/70e3991f7eaa34b4e356ab42d6fe15d4ae930984) · [本机结果](../outputs/reproduction/yolo11n/metrics.json)

## 可运行的双图方法

Fridman 等的 ChangeChip 先做 SIFT 单应性配准和亮度匹配，再以灰度与彩色局部差分描述做 PCA、K-means 和基于 MSE 的筛选。论文 §IV.C 的 20 对 CD-PCB 图报告像素 P=0.80、R=0.87。图 5 是定性例子。发布代码 `Scientific-Computing-Lab/ChangeChip@1c3281fabc009fc46a4a11c4c49441992fe22352` 有作者实现、示例图与数据包入口。基本管线无需学习权重，DEXTR 权重仅用于可选裁切。[论文原文](https://arxiv.org/abs/2109.05746) · [作者代码](https://github.com/Scientific-Computing-Lab/ChangeChip/tree/1c3281fabc009fc46a4a11c4c49441992fe22352)

**仅代码可运行，并完成子集实测**。数据包的 20 个目录都有 GT，其中 8 个目录同时提供 `input.JPG` 和 `reference.JPG`，编号为 1、2、4、5、11、18、19、20。本机全部运行；其余 12 对未推测配对关系。沿用作者示例缩放 0.2、窗口 5、聚类数 16、PCA 维数 3/9、配准和亮度修正，关闭可选裁切。固定 `seed=0`，适配 SciPy 图像保存接口、SIFT 名称与 K-means 旧默认 `n_init=10`。发布代码固定 `eps=0.02`，论文给射线图单列 `0.05`。8 对子集和参数差异限制了与全文数字的比较。

| 同一份输出，8 对、共 5,025,660 像素 | 预测变化像素 | 真值变化像素 | 命中像素 | 汇总 P / R | 逐图等权 P / R |
|---|---:|---:|---:|---:|---:|
| 作者 `evaluation.py` 的纯白前景规则 | 945 | 5,024,458 | 139 | 14.71% / 0.0028% | 11.63% / 0.0080% |
| 按发布图的稀疏深色标记计分，三通道均 <128 | 945 | 870 | 664 | 70.26% / 76.32% | 75.37% / 87.24% |

这组差异来自前景定义。第二行是显式改动的敏感性分析。只看第二行逐图召回接近 0.87 会掩盖子集、掩膜与聚合口径的差异。[逐图数据](../outputs/reproduction/changechip/per_pair.csv)

## 本机命令

在项目根目录运行。`$py` 指向本机 Python 3.12.4 的 yolo 环境。公开工程复算时，保留作者权重 `weights/external_yolo11n.pt` 和发布的 `outputs/reproduction/yolo11n_author_predictions.json`；两者分别来自该提交 `runs/yolo11n_baseline/weights/best.pt` 和 `runs/yolo11n_baseline_test/predictions.json`。ChangeChip 的仓库位于 `data/ChangeChip`，作者 README 链接的 CD-PCB 压缩包解压在 `data/cdpcb`。

```powershell
$py = 'C:\Users\BaiRong-NUC\.conda\envs\yolo\python.exe'
& $py -m pcb_reference.reproduction.yolo11n
& $py -m pcb_reference.reproduction.changechip
& $py -m pcb_reference.reproduction.changechip --rescore
& $py -m pcb_reference.reproduction.deeppcb_metric --oracle-only
```

ChangeChip 的额外依赖装于 `outputs/reproduction/deps`（`scikit-image==0.25.2`、`imageio==2.37.0`、`lazy-loader`、`tifffile`）。官方计分器使用 conda-forge `win-64/polygon3-3.0.9.1-py312h196c9fc_3.conda` 解包到 `outputs/reproduction/polygon`。适配器在内存中转换 Python 2 语法，并补回作者 `script.py` 两行参数字典被注释吞掉的逗号。原 `gt.zip` 是空格分隔的 XYXY 与类别编号，适配器逐框对照官方测试标注后转换为脚本支持的 `LTRB=True` 和类别字符串。作者匹配顺序、严格 `IoU > 0.33`、全局按置信度累计的 AP 均沿用。该 AP 与六类 COCO mAP 单独报告。
