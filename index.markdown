---
layout: default
description: 袁宝鹏的工程手记，记录训练系统与性能工程中计算、存储和通信之间的取舍，以及一些还在验证的问题。
image: /assets/images/og.png
---

<div class="site-shell home-page">
  <header class="home-intro">
    <p class="intro-label">工程手记</p>
    <h1>袁宝鹏</h1>
    <p class="home-lede">我是袁宝鹏，从事训练系统与性能工程。这里记录计算、存储和通信之间的取舍，以及一些还在验证的问题。</p>
    <a class="text-link" href="{{ '/about/' | relative_url }}">关于我与这个站 <span aria-hidden="true">↗</span></a>
  </header>
  <section class="home-section selected-writing" id="writing" aria-labelledby="writing-title">
    <div class="section-heading">
      <h2 id="writing-title">从这几篇读起</h2>
      <a class="text-link" href="{{ '/writing/' | relative_url }}">全部文章 <span aria-hidden="true">→</span></a>
    </div>
    <div class="post-list">
      {% for selection in site.data.selected_posts %}
      {% assign post = site.posts | where: 'slug', selection.slug | first %}
      {% if post %}
      <article class="post-row">
        <p class="post-meta"><time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: '%Y.%m.%d' }}</time><span>{{ selection.kind }}</span></p>
        <h3><a href="{{ post.url | relative_url }}">{{ post.title | escape }}</a></h3>
        {% if post.description %}<p class="post-description">{{ post.description | escape }}</p>{% endif %}
      </article>
      {% endif %}
      {% endfor %}
    </div>
  </section>
  <section class="home-section thinking-section" aria-labelledby="thinking-title">
    <div class="section-heading">
      <h2 id="thinking-title">最近在想</h2>
      <a class="text-link" href="{{ '/now/' | relative_url }}">此刻 <span aria-hidden="true">→</span></a>
    </div>
    <p class="section-note">先把问题留下来，答案可以慢慢找。</p>
    <div class="question-list">
      {% for question in site.data.questions %}
      <article><h3>{{ question.title }}</h3><p>{{ question.summary }}</p></article>
      {% endfor %}
    </div>
  </section>
  <section class="home-section early-notes" aria-labelledby="early-title">
    <div class="section-heading"><h2 id="early-title">早期笔记</h2><span class="quiet-date">2019 — 2020</span></div>
    <p>从 C++、xLearn 到 CUDA，留着当时读源码、理解机制的过程。内容有它的时间背景，也是一条可以回头看的技术轨迹。</p>
    <div class="early-links">
      <a href="{% post_url 2019-01-02-CPP的名字重整 %}">C++ 名字重整</a>
      <a href="{% post_url 2019-01-24-xlearn源码分析之FM的CalcScore实现 %}">xLearn 源码分析</a>
      <a href="{{ '/writing/#year-2020' | relative_url }}">CUDA 与 TensorRT 笔记 <span aria-hidden="true">→</span></a>
    </div>
  </section>
</div>
