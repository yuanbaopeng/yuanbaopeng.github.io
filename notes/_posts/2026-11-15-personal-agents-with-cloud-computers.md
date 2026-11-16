---
title: "和 Grok Bot 类似的个人 Bot 有哪些，谁有自己的云端电脑"
date: 2026-11-15 08:04:00 +0800
description: 只看能持续聊天、有记忆、会在后台主动干活的个人 Bot。先按托管还是自托管、有没有自己的电脑来分，再对比大厂托管、创业公司、自托管，以及厂商托管的 OpenClaw。
topics: [llm-agents]
tags: [Agent, Grok Bot, OpenClaw]
---

## 问题

市面上和 Grok Bot 类似的产品很多。这次我把范围收窄到「长期在线的个人 Bot」：能持续聊天、有记忆、会在后台主动干活，像一个一直在线的同事，而不是交一份任务、交付完就结束的工具。想弄清两件事：用什么标准来分；各家有没有自己的电脑。

## 要点

- 先看两个维度：托管还是自托管；有没有自己的电脑。Grok Bot、dots、Claude Duet、Muse 这一类是托管，并且有自己的云端电脑；Hermes、OpenClaw 是自托管，借用你的机器。
- 大厂托管里，按发布时间排列，是 Gemini Spark（现多称 Tasks in Gemini）、Grok Bot、Muse、Copilot Autopilot、dots、Claude Duet，以及 Perplexity Teammates。创业公司里，Genspark Claw、Cue、Hark Pro 也给了云端电脑；Poke 走消息，没有完整的云电脑。
- 自托管主要是 Hermes Agent 和 OpenClaw。介于两者之间的，是 Kimi Claw 和 MaxClaw：厂商在云端帮你托管一个 OpenClaw，不用自己部署。
- 从 2026 年 8 月 11 日的 Grok Bot 到 10 月 21 日的 Claude Duet，大约十周内，xAI、Meta、OpenAI、Anthropic 各推出了一个长期在线的 Bot，而且都跑在各自的云端电脑上。原版 Manus、Devin、Cursor 云端代理这类任务型产品，不在这次的范围内。

## 展开

比较时，我不想只看「能不能聊天」，而是看对话能不能一直续上、关了电脑它还在不在，以及它跑在谁的机器上。

托管省事，模型和账号连接是现成的，通常换不了模型。自托管更自由，模型可以自己选，但机器得一直开着，维护也得自己来。有自己的云端电脑，就可以把浏览器和终端交给它在后台跑；没有的话，多半是借你的设备，或只做到消息和工作流自动化。

### 大厂托管

按公开发布的时间排列。Autopilot 此前以 Scout 为名发布，这里按改名时间放在 Muse 和 dots 之间。

| 产品 | 公司 | 发布时间 | 自己的电脑 | 特点 | 可用性 |
| --- | --- | --- | --- | --- | --- |
| [Gemini Spark](https://blog.google/innovation-and-ai/products/gemini-app/next-evolution-gemini-app/) | Google | 2026-05-19 宣布 | 云端远程浏览器和代码环境 | 现多称 Tasks in Gemini，和 Workspace 集成很深 | 需 Google AI Pro 或 Ultra；欧洲经济区、英国、瑞士暂不可用 |
| [Grok Bot](https://x.ai/news/introducing-grok-bot) | xAI | 2026-08-11 | 有，多个 Bot 共用一台 | 可并行干活，也能互相交办任务 | 已上线 |
| [Muse](https://about.fb.com/news/2026/09/introducing-muse-personal-ai-agent/) | Meta | 2026-09-08 | 有，每人一台 Muse Secure VM | 也可在 WhatsApp 里使用 | 美国先上线 |
| [Copilot Autopilot](https://blogs.microsoft.com/blog/2026/09/25/introducing-the-new-copilot-with-home-code-and-autopilot/) | Microsoft | 2026-09-25 改名（2026-06-02 曾以 Scout 发布） | 公开介绍未写明有独立云电脑 | 基于 OpenClaw，住在 Teams 和 Outlook 里，有独立的 Entra 身份 | 企业预览 |
| [dots](https://openai.com/index/introducing-dots/) | OpenAI | 2026-09-29 | 有，可打开查看或接管；也可授权连接你的笔记本 | 能连 4000 多个应用 | Pro、Business Premium、Enterprise，地区受限 |
| [Claude Duet](https://claude.com/product/claude-duet) | Anthropic | 2026-10-21 研究预览 | 有，云端虚拟机和浏览器 | 沿用 Claude 账号里的记忆、skills 和已连接的应用；每天早报；每隔几小时检查一次已连接的账号；发消息、付款、签字、删除前会先问你 | Pro、Max、Ultra，需排队 |
| [Perplexity Teammates](https://www.perplexity.ai/hub/blog/teammates-are-agents-for-complex-ongoing-work) | Perplexity | 2026-11-04 | 有，每个 teammate 一台隔离的桌面和浏览器 | 面向持续、复杂的工作 | 需 Max 会员，研究预览 |

### 创业公司

原版 Manus 是任务型产品，个人 Bot 形态的是 Cue。

| 产品 | 公司 | 发布时间 | 自己的电脑 | 特点 | 可用性 |
| --- | --- | --- | --- | --- | --- |
| [Genspark Claw](https://www.genspark.ai/genspark-claw) | Genspark | 2026-03-12 | 有，一直在线的云电脑 | 定位是「AI 员工」 | 公开介绍未写清门槛 |
| [Cue](https://cue.im/) | Manus | 2026-09-28 | 有，每个 agent 都有自己的电脑 | 每个 agent 还有自己的邮箱、电话和钱包，支持多智能体群聊 | 公开介绍未写清门槛 |
| [Hark Pro](https://hark.com/) | Hark | 2026-10-06 | 有，专属云电脑，称作 Handoff | 长期记忆，会主动给建议 | 公开介绍未写清门槛 |
| [Instinct](https://instinct.com/) | Instinct | 未见明确日期 | 像人一样使用手机和电脑 | 通过短信和电话交流 | 邀请制 |
| [Poke](https://poke.com/) | Poke | 未见明确日期 | 没有完整的云电脑 | 走 iMessage、WhatsApp、Telegram，有记忆，用 Recipes 做自动化 | 公开介绍未写清门槛 |

### 自托管

| 产品 | 公司 | 发布时间 | 自己的电脑 | 特点 | 可用性 |
| --- | --- | --- | --- | --- | --- |
| [Hermes Agent](https://github.com/NousResearch/hermes-agent) | Nous Research | — | 借用你的机器 | 开源，模型可以换，可接 Telegram、Slack | 需自己部署，机器要一直开着 |
| [OpenClaw](https://github.com/openclaw/openclaw) | 开源项目 | — | 借用你的机器 | 插件生态最大；Copilot Autopilot、Kimi Claw、MaxClaw、JoyClaw 都基于它 | 需自己部署，机器要一直开着 |

### 托管版 OpenClaw

不用自己搭，但底座仍是 OpenClaw。

| 产品 | 公司 | 发布时间 | 自己的电脑 | 特点 | 可用性 |
| --- | --- | --- | --- | --- | --- |
| [Kimi Claw](https://www.kimi.com/help/kimi-claw/overview) | 月之暗面 | 约 2026-02 | 厂商云端托管 | 云端一键部署，可连接飞书、企业微信 | 公开介绍未写清其他门槛 |
| [MaxClaw](https://agent.minimaxi.com/activity/max-claw) | MiniMax | 2026-02-26 | 厂商云端托管 | 云端托管的 OpenClaw | 约 39 元/月 |

原版 Manus、Devin、Cursor 云端代理这类任务型产品，交一份任务、交付完就结束，不在这次的范围内。信息截至 2026-11-16。

## 还没确认的

- 各产品的具体能力、地区是否可用、订阅门槛，都只来自当时的公开介绍，没有逐一注册或自己长期用过。
- Copilot Autopilot 的公开介绍强调基于 OpenClaw、住在 Teams 和 Outlook 里、有独立的 Entra 身份，但没有写明是否给每个用户一台独立的云端电脑。
- Instinct、Poke 的准确发布日期，公开资料里我没有找到；Kimi Claw 的时间只精确到大约 2026 年 2 月。
- 创业公司和托管版 OpenClaw 的可用性、价格之后可能会变，以各产品当时页面为准。

## 参考

- [Gemini 应用的下一步](https://blog.google/innovation-and-ai/products/gemini-app/next-evolution-gemini-app/)
- [Introducing Grok Bot（xAI）](https://x.ai/news/introducing-grok-bot)
- [Introducing Muse（Meta）](https://about.fb.com/news/2026/09/introducing-muse-personal-ai-agent/)
- [Introducing the new Copilot with Home, Code and Autopilot（Microsoft）](https://blogs.microsoft.com/blog/2026/09/25/introducing-the-new-copilot-with-home-code-and-autopilot/)
- [Introducing dots（OpenAI）](https://openai.com/index/introducing-dots/)
- [Claude Duet](https://claude.com/product/claude-duet)
- [Teammates are agents for complex, ongoing work（Perplexity）](https://www.perplexity.ai/hub/blog/teammates-are-agents-for-complex-ongoing-work)
- [Genspark Claw](https://www.genspark.ai/genspark-claw)
- [Cue](https://cue.im/)
- [Hark](https://hark.com/)
- [Instinct](https://instinct.com/)
- [Poke](https://poke.com/)
- [Hermes Agent](https://github.com/NousResearch/hermes-agent)
- [OpenClaw](https://github.com/openclaw/openclaw)
- [Kimi Claw 说明](https://www.kimi.com/help/kimi-claw/overview)
- [MaxClaw](https://agent.minimaxi.com/activity/max-claw)
