---
title: "All Code Is Slop：先投资 Agent 的环境，再谈产量"
date: 2026-11-14 13:55:00 +0800
description: Lauren Tan（poteto）在 AI Engineer Code Summit 上的主线是：产量上去的同时要保住质量。先把约束做进代码和验证，再自动化外循环；/bro 和 /ketchup 则是她同时盯多个项目时用的沟通技能。
topics: [ai-coding, people-teams]
tags: [poteto, pstack, Agent]
---

## 问题

我想弄清两件连在一起的事。一件是 poteto（Lauren Tan）在 AI Engineer Code Summit 上那通大约 18 分钟的演讲 *All Code Is Slop (Except Mine)* 到底主张什么；另一件是 pstack 里的 `/bro` 和 `/ketchup` 怎么用，为什么她同时推进很多项目时会依赖它们。

## 要点

- 工程师的工作变成了：在产量大幅提高的同时，保住甚至提高质量。不要先上“软件工厂”，先投资 Agent 的环境，建立可验证的信任。
- 优先级像倒金字塔：先改代码本身和静态检查（API、架构、lint、类型、测试），再做运行时和形式化验证，最后才用规则、技能、AGENTS.md 这类需要判断的文字。
- 代码库是 Agent 最大的输入，坏例子会被复制。每次纠错都应变成可执行检查，或从设计上消掉这类错误，而不是再往 AGENTS.md 里加一句警告。
- 审查越来越变成看视频、截图、状态图和证明。她说自己大部分 token 都花在验证上。顺序很明确：内循环还在出 slop，就先别自动化外循环。
- `/bro` 和 `/ketchup` 不是写代码的命令，而是把对话说清楚的技能。`/bro` 用大白话重说上一句；`/ketchup` 补上你不在时发生了什么，并列出需要拍板的事，这一轮里不继续改代码。

## 展开

### 演讲的主线

官方单场页面上有文字整理，播放器会从她这一段开始并在结束时停下。视频来源是 Day 1 直播，时间范围是 1:01:02 到 1:18:38。

她举了自己团队里的几个做法。在 Dune 框架里，产品代码默认不能使用 `useEffect`，因为这个 API 太依赖细微判断，Agent 很难用对；用 `electron-main` 和 `electron-renderer` 目录，再加上 lint 和依赖分析，禁止跨进程乱引。他们也几乎不用注释，因为 Agent 会用注释给权宜之计找理由，下一轮又会把权宜之计当成既定做法。更好的反应是：写一条能指向正确修法的 lint，或从 API 上取消错误选项。

验证这一层，他们用功能地图让 Agent 从用户视角跑应用，用 TLA+ 检查跨设备通知，用 Lean 证明虚拟化库，用 Bombadil 做 UI 属性测试。发版后，Grok Bot 会自动开一个 Cursor 项目，像真实用户一样 fuzz，再提修复 PR 并自动合并。这种外循环能成立，前提是内循环里的错误已经被设计掉、被检查拦住，或被验证暴露出来。

她团队自报：上个月合并了约 11,000 个 PR（她自己约 3,000 个），相对回滚率约 0.5%，客户端大约每天发一个稳定版，10 月有 227 人至少合过一个 PR，启动、打字、开关对话都变快了。这些是团队自己报告的结果，不是对照实验。

### /bro 和 /ketchup

两者都属于 pstack。`/bro` 针对上一句：如果说得太技术、太绕，就用更短、更像跟人说话的方式再说一遍。`/ketchup` 针对你不在的这段时间：用同样直白的口气补上进展，并列出还需要你决定的事，比如合并、批准、做选择。它只做汇报。

对话里对她用法的转述是：同时盯大约 30 个项目时，在项目之间来回“helicopter”，回到一个对话时先 `/ketchup` 看清进展和要决定的事；哪段说不清，再 `/bro` 听成人话。在 Cursor 里对 bot 直接输入这两个命令即可。

### 她是谁

poteto 是她在 X 和 GitHub 上的用户名。她参与过 React 核心团队和 React Compiler，并开源了 Cursor 插件 pstack。pstack 的思路是先把一个任务做深、做对，再追求快和多；每一步都要求有真正跑过代码的证据。会议页面上的头衔是 SpaceXSI Principal Engineer。

## 还没确认的

- 11,000 个 PR、0.5% 回滚率、227 位贡献者，都是演讲里的团队自报，没有外部审计，也不能直接外推到别的代码库。
- “同时盯大约 30 个项目”是对话里对她工作方式的转述，不是演讲正文里的原话。
- 会议页面写作 SpaceXSI，此前履历介绍里还有 Cursor、SpaceXAI 等不同写法，公开头衔的规范名称我没有再和 LinkedIn 对过。
- pstack 里 playbook 的数量、默认用哪些模型，会随版本变化，这里不把当时的细节写成长期有效的说明。

## 参考

- [All Code Is Slop (Except Mine) 官方单场页面](https://ai.engineer/code/2026/talks/lauren-tan)
- [pstack 插件](https://cursor.com/marketplace/cursor/pstack)
- [pstack 说明（cursor/plugins）](https://github.com/cursor/plugins/blob/main/pstack/README.md)
- [poteto 的 GitHub](https://github.com/poteto)
- [no.lol（Lauren Tan 的博客）](https://www.no.lol/)
