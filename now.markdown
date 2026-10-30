---
layout: default
title: 此刻
description: 最近在想的两个问题：训练系统对 Agent 的可观测性，以及等待 GPU 时如何组织异步性能实验。
image: /assets/images/og.png
permalink: /now/
---

<div class="site-shell document-page">
  <header class="page-heading">
    <h1>此刻</h1>
    <p class="page-lede">记录最近还在想的问题。它们可以继续变化，也不必马上变成文章或项目。</p>
    <p class="quiet-date">更新于 <time datetime="2026-10-30">2026 年 10 月 30 日</time></p>
  </header>
  <div class="prose current-questions">
    {% for question in site.data.questions %}
    <section><h2>{{ question.title }}</h2><p>{{ question.summary }}</p><p>{{ question.detail }}</p></section>
    {% endfor %}
    <section class="reading-note">
      <h2>已经写下的一个起点</h2>
      <p><a href="{% post_url 2026-08-28-training-platform %}">训练任务平台真正难的是什么：从提交作业到交付结果</a>已在 2026 年 8 月发布。这篇公开资料研究梳理了生命周期、调度、可观测性与容错；上面的问题，是沿着这些线索继续想下去。</p>
    </section>
  </div>
</div>
