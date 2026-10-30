---
layout: default
title: 关于
description: 关于袁宝鹏，以及这本慢慢积累的工程手记。
image: /assets/images/og.png
permalink: /about/
---

<div class="site-shell document-page">
  <header class="page-heading about-heading">
    <div><h1>关于</h1><p class="page-lede">我是袁宝鹏，从事训练系统与性能工程。</p></div>
    <img class="profile-avatar" src="{{ '/assets/images/avatar.png' | relative_url }}" alt="袁宝鹏的 GitHub 头像" width="88" height="88">
  </header>
  <div class="prose">
    <p>我关心计算、存储和通信之间的取舍，也关心一个系统为什么会慢、为什么会失败，以及怎样找到足够可靠的证据。</p>
    <h2>从具体问题出发</h2>
    <p>站里较早的文章是 C++、Redis 和 xLearn 的笔记，后来写到 CUDA Samples 与 TensorRT。它们保留了读源码、理解机制时的一些过程，不同阶段的笔记也有各自的局限。</p>
    <p>现在更关注训练系统与性能工程，也在想 Agent 如何参与有真实资源约束的工程工作。还没想清楚的部分，先放在<a href="{{ '/now/' | relative_url }}">「此刻」</a>。</p>
    <h2>关于这本手记</h2>
    <p>这里用来留下值得回头看的东西：一个问题是怎样被理解的，什么证据改变了判断，哪些约束影响了选择。源码分析、学习笔记和公开资料研究会尽量说明各自的背景。</p>
    <p>例如，<a href="{% post_url 2026-08-28-training-platform %}">训练任务平台这篇文章</a>是基于公开资料的学习整理，文中的判断是一种分析框架，不代表亲历项目的完整实践。</p>
    <p>有想清楚的部分就记下来，暂时没有结论也没关系。以后如果有想留住的阅读或生活片段，也可以放在这里。</p>
    <h2>联系</h2>
    <p>如果某篇文章让你想到一个问题，欢迎交流。</p>
    <div class="contact-links"><a href="mailto:{{ site.email }}">{{ site.email }}</a><a href="https://github.com/{{ site.github_username }}" target="_blank" rel="noreferrer">GitHub <span aria-hidden="true">↗</span></a></div>
  </div>
</div>
