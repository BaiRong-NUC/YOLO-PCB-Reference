# 参考论文与复现记录

Zotero 位置为 `Research → YOLO-PCB-Reference`，分类键 `ICNZIWQR`。本次逐条回读得到 7 个论文条目、6 份 PDF。PIDDN 目前保存书目和 DOI。权重、运行环境、指标差异与作者材料缺口见[复现卡](reproduction.md)。

| 论文 | 在本文中的作用 | 当前核查范围与复现状态 |
|---|---|---|
| He 等，*A Real-time PCB Defect Detector Based on Supervised and Semi-supervised Learning*，ESANN 2020，527–532。`TWV23UNT` | DeepPCB 的配对输入、六类 HBB、官方划分与指标。论文已有双图输入与多尺度特征提取。 | 已核原文第 2 节、仓库说明、数据及评测器。仓库包含数据、标注工具和评测脚本，未找到对应检测器的完整训练实现及权重；原论文 98.6% mAP 为作者报告。[论文](https://www.esann.org/sites/default/files/proceedings/2020/ES2020-33.pdf) · [作者仓库](https://github.com/tangsanli5201/DeepPCB) |
| Jiang 等，*PIDDN: Pair-Image-Based Defect Detection Network With Template for PCB Inspection*，IEEE TCPMT 2025，15(4)，830–841。`MWVB5EM6` | 同任务直接竞争方法。作者提供单图、双图特征相减和提出方法三组配置，可用来区分模板输入与融合设计的价值。 | 已核书目及作者仓库 `configs/deeppcb/swin_t_faster_rcnn_deeppcb_{single,backbone_post_sub,ours}.py`。正式复现需对齐数据、权重和评测，当前没有本机复现值。[论文](https://doi.org/10.1109/TCPMT.2025.3543396) · [作者配置](https://github.com/QixingJiang/PIDDN/tree/main/configs/deeppcb) |
| Deng 等，*EEDD: Edge-Guided Energy-Based PCB Defect Detection*，Electronics 2023，12(10)，2306。`YH57S6WT` | 与本项目失效假设最相近。作者已研究模板错位、边缘干扰和微小缺陷，并采用特征点配准与轮廓异常度量。 | 已核作者摘要、数据集说明与算法概述。需注明 DeepPCB-A 与当前原始 DeepPCB 的差异，本机未复现原法。[论文](https://doi.org/10.3390/electronics12102306) |
| Saiyod 等，*Defect-Intent Ambiguity Addressing for Training-Free Deterministic PCB Defect Localization via Template Selection and Dissimilarity Mapping*，Sensors 2026，26(5)，1541。`NZYMRHWB` | 提供以 SSIM 改善为条件接受全局配准的先例，同时报告误报负担。 | 已核作者摘要及方法描述，未运行原法。[论文](https://doi.org/10.3390/s26051541) |
| Huang 等，*Registration Based Few-Shot Anomaly Detection*，ECCV 2022，303–319。`64AKFTLM` | 思想来源。先建立正常参照的对应关系，再评估异常。其少样本异常检测设定与六类监督检测不同。 | 已核会议书目、作者仓库及其运行入口。此处借鉴配准思想，未移植其网络，也未复现其性能。[论文](https://doi.org/10.1007/978-3-031-20053-3_18) · [作者代码](https://github.com/MediaBrain-SJTU/RegAD) |
| Jocher 等，*Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models*，arXiv:2606.03748，2026。`A2YBVJG6` | 检测器基线来源。 | 已核官方文档与本机配置。[论文](https://arxiv.org/abs/2606.03748) · [官方模型文档](https://docs.ultralytics.com/models/yolo26/) |
| Fridman、Rusanovsky、Oren，*ChangeChip: A Reference-Based Unsupervised Change Detection for PCB Defect Detection*，IEEE PAINE 2021。`DSIJ2PD7` | 提供了可执行的双图配准、局部差分描述与无监督像素定位对照。 | 已核原文 III、IV 节特别是 IV.C、图 5，运行了作者代码和公开压缩包中资料齐全的 8 对图。保留其计分脚本与发布掩膜前景极性冲突的结果。[论文](https://arxiv.org/abs/2109.05746) · [作者代码](https://github.com/Scientific-Computing-Lab/ChangeChip) |

本次复跑的双图论文有独立的[ChangeChip 框架图与复现卡](papers/changechip/README.md)。
