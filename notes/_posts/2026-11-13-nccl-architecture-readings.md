---
title: "读 NCCL 架构，可以按这条路径走"
date: 2026-11-13 13:32:00 +0800
description: 先读 2025 年把 NCCL 内部机制拆开的论文，再配官方博客和文档，就能把 channel、协议和 ring/tree 算法串起来。
topics: [training-infra]
tags: [NCCL, GPU, 通信]
---

## 问题

训练和推理里经常碰到 NCCL，但官方文档偏接口，内部的 channel、协议和算法不太好一次看清。我想先找到一套靠谱的阅读路径，而不是从源码盲目挖起。

## 要点

- 最值得先看的，是 2025 年这篇 *Demystifying NCCL*。它把内部机制拆开，覆盖 channel，Simple / LL / LL128 三种协议，以及 ring 和 tree 算法。
- 官方早期博客讲清了 ring、单 kernel，以及 Copy、Reduce 这些原语；NCCL 2.4 的博客则补充了后来加入的 double binary tree。
- 调优一篇讲算法、协议、CTA 和成本模型怎么选，有中文版。想看实现细节，可以再读 *How is NCCL so fast?*。
- 官方 Overview 和源码是查证用的，适合在读完分析文章之后对照，而不是当作入门的第一篇。

## 展开

推荐的顺序是：先用 *Demystifying NCCL* 建立内部图景，再读一篇把它翻译成更易读笔记的中文整理，然后回看 NVIDIA 自己按时间写出的架构说明。这样能把三件事连起来：通信怎么被切成 channel 并行；小消息和大消息为什么要换协议；集合通信为什么在 ring 和 tree 之间切换。

*Demystifying NCCL* 基于 NCCL 2.19.1 做分析，作者来自 ETH Zürich，并有 NVIDIA、Broadcom 的参与。具体实现细节以后可能改，但 channel、协议和 ring/tree 这套结构是后续调优和读源码的底图。NCCL 2.4 引入 double binary tree，是为了在大规模上同时照顾带宽和延迟。调优文则说明：协议和算法往往由成本模型来选，长期把环境变量写死通常不是首选。

这次只是把路径找齐，还没有按章节把论文拆完。

## 还没确认的

- 中文笔记是对论文的转述，术语和示意图是否和原文完全一致，需要对照论文再核对。
- 这篇分析用的是 NCCL 2.19.1，当前默认版本里的实现细节可能已经有出入。
- 各篇文章对“为什么快”的解释侧重不同，还没有用同一组测试把协议和算法的选择验证一遍。

## 参考

- [Demystifying NCCL: An In-depth Analysis of GPU Communication Protocols and Algorithms](https://arxiv.org/abs/2507.04786)
- [NCCL 论文阅读（中文笔记）](https://www.cnblogs.com/CQzhangyu/p/19108788)
- [Fast Multi-GPU collectives with NCCL](https://developer.nvidia.com/blog/fast-multi-gpu-collectives-nccl/)
- [Massively Scale Your Deep Learning Training with NCCL 2.4](https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/)
- [Understanding NCCL Tuning to Accelerate GPU-to-GPU Communication](https://developer.nvidia.com/blog/understanding-nccl-tuning-to-accelerate-gpu-to-gpu-communication/)
- [了解 NCCL 调优以加速 GPU 之间的通信（中文版）](https://developer.nvidia.cn/blog/understanding-nccl-tuning-to-accelerate-gpu-to-gpu-communication/)
- [How is NCCL so fast?](https://gessfred.xyz/nccl-behind-the-curtain/)
- [Overview of NCCL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html)
- [NVIDIA/nccl 源码](https://github.com/NVIDIA/nccl/)
