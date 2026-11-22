---
title: "主流大模型还剩哪些是纯文本的"
date: 2026-11-20 09:00:00 +0800
description: 截至 2026 年 11 月，闭源主力聊天模型基本都能看图。还明显偏纯文本的，主要是开源权重、推理或编程专用版本，以及个别系列里的文本档。
topics: [llm-agents]
tags: [多模态, 大模型]
---

## 问题

先确认像 Claude Opus 这样的主力是不是多模态，再问清两件事：多模态和纯文本的核心区别是什么；现在主流大模型是不是都已经能看图，还剩下哪些是纯文本。

## 要点

- 核心区别是：多模态模型能直接把图片、声音或视频当成输入来理解，而不必先由别的系统转成文字。输出仍以文本为主的模型，只要输入已经跨了模态，也算多模态。
- 截至 2026 年 11 月，闭源主力聊天模型基本都能看图。差别更多在还能不能直接处理音频和视频，以及生图、语音是不是另一条模型线。
- 仍以纯文本为主的，主要是开源权重和部分系列里的文本档，例如 OpenAI 的 gpt-oss、智谱的 GLM-5.3。纯文本还在，但更多是为了便宜、易部署，或把能力集中在代码和推理上。

## 展开

普通语言模型只在文本空间里推理。多模态模型则把不同感官信号映到同一个表示里，所以能看图答问、对照截图排错、结合画面和说明一起判断。

闭源主力这边，2026 年 11 月 22 日核对过官方文档，情况大致如下。

- [Claude](https://docs.anthropic.com/en/docs/about-claude/models)：官方写明，当前型号都支持文本和图片输入、文本输出。Opus 5.5、Sonnet 5.5 都在这个范围内。
- [OpenAI](https://developers.openai.com/api/docs/models)：官方写明，最新模型都支持文本和图片输入、文本输出。当前旗舰是 GPT-6.2 Astra，日常用 GPT-6.2 Sol，便宜档是 GPT-6 Luna。
- [Gemini](https://ai.google.dev/gemini-api/docs/models)：当前的 Gemini 4 系列（Carbon、Neon、Hydrogen、Argon）官方都标明支持文本、图片、音频、视频和 PDF。覆盖面仍然是这几家里最完整的之一。
- [Grok](https://docs.x.ai/docs/models)：模型总览把生图、视频、语音列成独立接口和定价，聊天旗舰写的是 Grok 4.7。[Grok 4.7 型号页](https://docs.x.ai/developers/models/grok-4.7)写明输入是文本和图片，输出是文本。

仍以纯文本为主、或刚从纯文本切过来的，核对到的是这些：

- OpenAI 的开源权重 [gpt-oss-120b](https://developers.openai.com/api/docs/models/gpt-oss-120b) 和 [gpt-oss-20b](https://developers.openai.com/api/docs/models/gpt-oss-20b)，官方都只支持文本进、文本出。
- 智谱 [GLM-5.3](https://docs.z.ai/guides/llm/glm-5.3) 官方仍是纯文本输入；看图、看视频要走 [GLM-5.3-Flash](https://docs.z.ai/guides/vlm/glm-5.3-flash) 这类多模态版本，其输入模态写的是视频、图片、文本和文件。
- DeepSeek 已在淘汰 API 里的 V4-Pro：从 2026 年 9 月 14 日 04:00 UTC 起，`deepseek-v4-pro` 会转到带原生视觉的 [V4.1-Flash](https://api-docs.deepseek.com/news/news260910/)。自己部署的 V4 Pro 权重是否仍是纯文本，没有在这篇里核对。

所以现在更准确的分法是：通用旗舰默认都能看图；Gemini 4 在音频和视频理解上更完整；纯文本还在，但已经不是主流默认形态。另外，能理解图片，不等于能生成图片或语音，这两件事通常是不同模型。Gemini 型号页上的生图线包括 Nano Banana 系列，OpenAI 型号页列出 GPT-Live-1.1 和 GPT-Image 系列；这里只说明它们是另一条模型线，没有逐项核对能力。

11 月 20 日对话里，Gemini 举的例子还是 3.1 Pro 和 3.8 Flash。两天后核对官网时，当前主力已经是 Gemini 4，3.8 Flash 被标为上一代 Flash。型号迭代很快，对照应以当时的官方型号页为准。

## 还没确认的

- 对话里还提到 Llama 4 Scout/Maverick、Kimi K3、Qwen3.8 Max 等已经能看图，有的还支持视频。这些没有在本篇里逐一核对其官方型号页。
- 自己部署的 DeepSeek V4 Pro 权重是否仍是纯文本，没有核对。公告只写了 API 上的 `deepseek-v4-pro` 会转到 V4.1-Flash。
- 型号和能力变化很快，本文是 2026 年 11 月 22 日这一时点的对照。

## 参考

- [Claude models overview](https://docs.anthropic.com/en/docs/about-claude/models)
- [OpenAI Models](https://developers.openai.com/api/docs/models)
- [gpt-oss-120b](https://developers.openai.com/api/docs/models/gpt-oss-120b)
- [gpt-oss-20b](https://developers.openai.com/api/docs/models/gpt-oss-20b)
- [Gemini models](https://ai.google.dev/gemini-api/docs/models)
- [xAI Models](https://docs.x.ai/docs/models)
- [Grok 4.7](https://docs.x.ai/developers/models/grok-4.7)
- [GLM-5.3](https://docs.z.ai/guides/llm/glm-5.3)
- [GLM-5.3-Flash](https://docs.z.ai/guides/vlm/glm-5.3-flash)
- [DeepSeek-V4.1-Flash 发布说明](https://api-docs.deepseek.com/news/news260910/)
