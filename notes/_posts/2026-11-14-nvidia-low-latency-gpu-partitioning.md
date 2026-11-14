---
title: "低延迟推理里，GPU 资源该怎么切，TP4 又切在哪"
date: 2026-11-14 08:16:00 +0800
description: 整理一场 NVIDIA 讲座的四页幻灯：低延迟要先把模型拆到多卡，再填上每张卡内部的空隙；不同切分方式的天花板不一样，TP4 示例里各算子切的维度也不同。
topics: [training-infra]
tags: [GPU, 推理, TP]
mathjax: true
---

## 问题

我拿到几页 NVIDIA 现场讲座幻灯，案例模型是 MiniMax M3.1 Flash。想把低延迟推理里“跨卡怎么切、卡内空隙怎么填、TP4 的矩阵到底切在哪一维”整理成一条能以后回头看的笔记。

## 要点

- 低延迟要从两层下手：先把模型拆到多张 GPU 上，再消除每张卡内部的空隙。
- 共同目标是让每一种 GPU 资源都保持忙碌：Tensor Cores、CUDA cores、SM、TMA、显存带宽和 NVLink。真正的问题不是要不要占用它们，而是怎么切分。切得越细，可重叠的工作越多，但天花板也不一样。
- 从 Multi-stream、PDL、Green Context，到 CTA / warp specialization，切分粒度越来越细。现有方案也沿着同一条谱演化：先保留 kernel 边界，再用分区让不同 kernel 共存，再走到自带调度器的大 kernel。
- TP4 示例里，所有 token 在每张卡上都保留，但不同算子切的维度不同：Input Proj 切 $N$（head 分组），Indexer 切 $M$（Indexer 的 query heads），Sparse Attention 按本地 heads 和对应 KV pages 落卡，O Proj 切 $K$，最后 AllReduce 求和。

## 展开

### 低延迟对系统提出的要求

跨卡（batch = 1）时，Attention 和 MoE 在 8 张 GPU 上的切分，决定了每张卡的工作量，以及关键路径上要走几轮通信。幻灯上的写法是：Attention 用 DP、TP、CP；MoE 用 EP、TP。

卡内对照了两种排程。一种是 kernel by kernel，小 kernel 之间有明显空隙。另一种是一条持续调度，把 weights、compute、comm 排在一起：工作量相同，但 launch 和空隙应从关键路径上离开。这一部分要回答三个问题：什么可以提前跑，什么必须等待，以及这在 warp、SMEM 和 SM 上要花多少代价。

### 五种让 GPU 保持忙碌的切分

| 机制 | 范围 | 资源怎么切 | 上限在哪 |
| --- | --- | --- | --- |
| Multi-stream | 跨 kernel | 硬件把相互独立 kernel 的 CTA 填到空出来的 SM 上 | 只有 SM 恰好空闲时才能重叠 |
| PDL | 跨 kernel | 前一个 kernel 排空时，下一个 kernel 的 CTA 接管 SM | 只能在 kernel 边界重叠 |
| Green Context | 跨 kernel | 每个 kernel 固定一块 SM 分区 | 静态切分，负载变了也不能再平衡 |
| CTA specialization | 单 kernel 内 | 同一 kernel 里的 CTA 分担 comm 或 compute | 按 SM 分角色：做通信的 SM 会让 Tensor Cores 空转 |
| Warp specialization | 单个 CTA 内 | 同一 CTA 里的 warp 分别做 MMA、epilogue 或 comm | 受限于各角色的寄存器和 SMEM |

这些手段可以叠加。幻灯提到 TileRT 同时用了 warp specialization 和 CTA specialization，甚至把角色拆到多张 GPU 上。

### 现有方案怎么排成一条谱

CUDA Graph + PDL 仍保留 kernel 边界：Graph 切掉 CPU launch 开销，PDL 让下一个 kernel 的 setup 提前开始。幻灯的评价是，这一招往往已经够用。

GC MoE 用两个 Green Context，让 dispatch 和 compute 各占一块 SM，用 per-tile 的 ready flags 交接。DeepSeek Mega MoE 则把 Dispatch、GEMM 和 Combine 放进一个自带调度的大 kernel，按 chunk 流水线。TileRT 更进一步，一个 kernel 里放进多个算子和 scheduler，TMA 跨过操作边界持续装数。这种“自带调度器的大 kernel”形态，幻灯注明参考了 S. Jones 的 Megakernels。

### TP4：矩阵如何划分

记号：$x$ 是 token 行数，$H$ 是 hidden dimension，$L$ 是 context length。矩阵乘法写成 $C[M,N] = A[M,K] \times B[K,N]$。一共 4 张 GPU，所有 $x$ 个 token 在每张卡上都保留。

Input Proj（Q projection）切的是 $N$，也就是 query-head groups：

$$
X[x,H] \times W_q[H, 64 \times d] = Q[x, 64 \times d]
$$

64 个 head 分给 4 张 GPU，每卡 16 个，于是 $N = 64 \times d \rightarrow 16 \times d$。$H$ 维输入在各卡上都有。$d$ 是 Query/Key 的 head dimension。

Indexer（Scores → TopK）切的是 $M$，也就是 Indexer 的 query heads：

$$
Q_i[4 \times x, d_i] \times K_i^{\top}[d_i, L] = \mathrm{Scores}[4 \times x, L]
$$

4 个 head 分给 4 张 GPU，每卡 1 个，因此每卡是 $x$ 行。$d_i$ 是 Indexer 的 head dimension。

Sparse Attention 按本地 query heads 和对应 KV head 的 selected pages 落在各卡上：GPU 0 负责 heads 0–15 和 KV head 0，GPU 1 负责 16–31 和 KV head 1，以此类推。每卡上的 $Q$ 形状是 $[x, 16, d]$，输出 $O[x,16,d_v]$ 再交给对应的 O Proj。

O Proj 切的是 $K$，也就是 attention-head 的输出：

$$
A[x, 64 \times d_v] \times W_o[64 \times d_v, H] = Z_r[x, H]
$$

$K = 64 \times d_v \rightarrow 16 \times d_v$ per GPU。$d_v$ 是 Value 的 head dimension。各卡先算出偏 $Z$，再对 4 张卡做 AllReduce SUM。

## 还没确认的

- 幻灯上没有讲者署名，公开录像也还没找到，所以这是对现场四页幻灯的整理，不是对整场讲座逐字核对后的记录。
- “Part 03 要回答的三个问题”那一行，有几个字被讲者挡住，是按上下文补全的。
- TileRT、DeepSeek Mega MoE、GC MoE 的实现细节，以及 S. Jones Megakernels 的原始出处，这几页幻灯里没有展开，也没有另做文献核对。
- TP4 示例是 MiniMax M3.1 Flash 场景下的划分，换模型或换切分策略后，切的维度不一定还是这样。

## 参考

- 依据是 2026 年 11 月 14 日整理的 NVIDIA 现场讲座幻灯四页，案例模型为 MiniMax M3.1 Flash。目前还没有已核对、可公开访问的讲座页面或录像链接。
