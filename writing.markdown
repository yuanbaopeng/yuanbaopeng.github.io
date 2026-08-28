---
layout: default
title: 技术文章
description: 关于 AI 工程、GPU 性能、复杂系统与研发效率的技术文章。
image: /assets/images/og.png
permalink: /writing/
---

<section class="page-hero archive-hero">
  <div class="site-shell page-hero-inner">
    <p class="page-eyebrow">WRITING / 技术文章</p>
    <h1>写下来，<br><span>才能真正想清楚。</span></h1>
    <p class="page-lede">这里不追逐信息流。每一篇文章都从一个具体问题出发，尽量把机制、过程和取舍讲清楚。</p>
  </div>
</section>

<section class="archive-section">
  <div class="site-shell">
    <div class="archive-toolbar">
      <p><strong>{{ site.posts | size }}</strong> 篇公开文章</p>
      <div class="topic-list" aria-label="文章主题">
        <span>CUDA</span><span>TensorRT</span><span>C++</span><span>机器学习</span><span>系统</span>
      </div>
    </div>

    <div class="archive-list">
      {% for post in site.posts %}
      <article class="archive-row">
        <time datetime="{{ post.date | date_to_xmlschema }}"><strong>{{ post.date | date: "%d" }}</strong><span>{{ post.date | date: "%Y.%m" }}</span></time>
        <div class="archive-body">
          <p class="archive-topic">{{ post.categories | first | default: 'TECH NOTE' }}</p>
          <h2><a href="{{ post.url | relative_url }}">{{ post.title | escape }}</a></h2>
          <p>{{ post.description | default: post.excerpt | strip_html | strip_newlines | truncate: 150 }}</p>
        </div>
        <a class="archive-arrow" href="{{ post.url | relative_url }}" aria-label="阅读《{{ post.title | escape }}》">↗</a>
      </article>
      {% endfor %}
    </div>
  </div>
</section>

<section class="archive-end">
  <div class="site-shell">
    <p>这些早期文章记录了我从系统、算法到 GPU 工程的技术轨迹。新的内容会沿着 AI 工程与研发效率继续向前。</p>
    <a class="text-link" href="{{ '/feed.xml' | relative_url }}">通过 RSS 订阅 <span aria-hidden="true">→</span></a>
  </div>
</section>
