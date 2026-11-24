# PCB 缺陷检测 · 论文复现

当前只做第一步，每次复现一篇论文。每篇在 `docs/papers/` 下使用独立目录，并排画出基线与论文结构，说明各处改动为何可能有效，再记录已有证据、复跑命令和复现缺口。

## 本次论文

**ChangeChip**（IEEE PAINE 2021）比较正常模板与待检 PCB，输出发生变化的像素位置。论文使用 CD-PCB 的 20 对图像。本机取得的发布包中有 8 对同时提供两张输入图和标注，复现结果按这 8 对计。

[Celik 基线、ChangeChip 结构与同图对照结果](docs/papers/changechip/README.md) · [高清结构对照图](docs/papers/changechip/framework.png) · [ChangeChip 逐图指标](outputs/reproduction/changechip/per_pair.csv)

## 其他第一步记录

[候选论文及其作用](docs/references.md) · [公开权重、计分器与作者报告的核验记录](docs/reproduction.md)
