---
title: "曾妍：从视觉语言预训练到 Seedance"
date: 2026-11-13 16:21:00 +0800
description: 根据可核对的公开材料，整理字节跳动 Seed 团队曾妍的研究路径：早期对话和视觉语言预训练，再到 PixelDance 与 Seedance。
topics: [people-teams]
tags: [曾妍, Seedance, 字节跳动]
---

## 问题

想了解字节跳动 Seed 团队的曾妍是谁、主要做了哪些工作。只依据能在论文和官方主页上核对的信息，不写职级、薪资或来源不清的报道。

## 要点

- 曾妍现在在字节跳动 Seed。她的 [Google Scholar](https://scholar.google.com/citations?user=-zT0sBsAAAAJ) 页面写的是 Lead of Seedance，邮箱为 bytedance.com 验证地址。
- 2020 到 2021 年，她与蒙特利尔大学的聂建云（Jian-Yun Nie）合作发表对话系统论文，内容围绕预训练语言模型和对话生成、对话状态跟踪。
- 2021 年底到 2022 年，她以第一作者发表 X-VLM（ICML 2022），做文本和不同粒度视觉概念的对齐；随后还有 X2-VLM 和跨语言跨模态的 CCLM。
- 2023 年她以第一作者发表 PixelDance（CVPR 2024），用文本加上首尾帧约束做高动态视频生成。
- 2025 年 Seedance 1.0 技术报告公开；她是作者之一。2026 年 Seed 发布了 Seedance 2.0。她还参与了把图像编辑迁到视频编辑的 VINCIE-NExT。

## 展开

Scholar 上的研究关键词是 Video Generation 和 Vision Language Pre-training，和论文时间线一致：先做对话和视觉语言预训练，再进入视频生成。

X-VLM 的正式题目是 *Multi-Grained Vision Language Pre-Training: Aligning Texts with Visual Concepts*，代码仓库在她的 GitHub 账号下。X2-VLM 对应后来的 *X-VLM: All-In-One Pre-trained Model For Vision-Language Tasks*。CCLM 是 *Cross-View Language Modeling*，做跨语言、跨模态的统一预训练。

PixelDance 的正式题目是 *Make Pixels Dance: High-Dynamic Video Generation*，核心是用文本指令，再加上首帧、尾帧这样的图像约束，提升复杂场景和大动作的视频生成。Seedance 1.0 的技术报告是 *Seedance 1.0: Exploring the Boundaries of Video Generation Models*。Seedance 2.0 有官方发布页和技术报告 *Seedance 2.0: Advancing Video Generation for World Complexity*。

VINCIE-NExT 把视频编辑拆成“视频 → 图像 → 图像 → 视频”的链，用图像编辑对作为上下文示范，降低对大规模成对视频编辑数据的依赖。曾妍是作者之一，单位写作 ByteDance Seed。

## 还没确认的

- 本科学校、硕士毕业时间和以校招生身份加入字节 AI Lab 的日期，常见于媒体报道，我没有在论文或官方页面上核对，这里不写定。
- Scholar 上写的是 Lead of Seedance，但 Seedance 2.0 的官方发布页没有点名预训练负责人，不能把“2.0 预训练负责人”写成已核实的职务表述。
- PixelDance 是否被官方视为 Seedance 的前身，这是后来的归纳，不是论文里的原话。
- VINCIE-NExT 已在 arXiv 公开，对话里说它被 NeurIPS 2026 接收，这一点我没有在会议程序里核对。

## 参考

- [Yan Zeng 的 Google Scholar](https://scholar.google.com/citations?user=-zT0sBsAAAAJ)
- [X-VLM（ICML 2022）](https://arxiv.org/abs/2111.08276)
- [X-VLM ICML 论文页](https://proceedings.mlr.press/v162/zeng22c.html)
- [X-VLM 代码仓库](https://github.com/zengyan-97/X-VLM)
- [X2-VLM](https://arxiv.org/abs/2211.12402)
- [CCLM: Cross-View Language Modeling](https://arxiv.org/abs/2206.00621)
- [与聂建云合作的对话生成论文（NAACL 2021）](https://aclanthology.org/2021.naacl-main.392/)
- [PixelDance（CVPR 2024）](https://openaccess.thecvf.com/content/CVPR2024/html/Zeng_Make_Pixels_Dance_High-Dynamic_Video_Generation_CVPR_2024_paper.html)
- [PixelDance 预印本](https://arxiv.org/abs/2311.10982)
- [Seedance 1.0 技术报告](https://arxiv.org/abs/2506.09113)
- [Seedance 1.0 官方论文页](https://seed.bytedance.com/en/public_papers/seedance-1-0-exploring-the-boundaries-of-video-generation-models)
- [Seedance 2.0 官方发布](https://seed.bytedance.com/en/blog/official-launch-of-seedance-2-0)
- [Seedance 2.0 技术报告](https://arxiv.org/abs/2604.14148)
- [VINCIE-NExT](https://arxiv.org/abs/2610.12104)
- [VINCIE-NExT 项目页](https://vincie-next.github.io/)
