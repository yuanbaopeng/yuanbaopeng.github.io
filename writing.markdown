---
layout: default
title: 文章
description: 训练系统、性能工程与早期源码阅读笔记，按时间慢慢积累。
image: /assets/images/og.png
permalink: /writing/
---

<div class="site-shell document-page">
  <header class="page-heading">
    <h1>文章</h1>
    <p class="page-lede">从源码阅读到系统问题，一些已经写下的思考。早期笔记保留原来的时间与背景。</p>
    <div class="archive-summary"><span>{{ site.posts | size }} 篇文章</span><a href="{{ '/feed.xml' | relative_url }}">RSS 订阅 <span aria-hidden="true">↗</span></a></div>
  </header>
  {% assign years = site.posts | group_by_exp: 'post', "post.date | date: '%Y'" %}
  {% for year in years %}
  <section class="archive-year" aria-labelledby="year-{{ year.name }}">
    <h2 id="year-{{ year.name }}">{{ year.name }}</h2>
    <div class="post-list">
      {% for post in year.items %}
      {% assign selection = site.data.selected_posts | where: 'slug', post.slug | first %}
      <article class="post-row">
        <p class="post-meta"><time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: '%m.%d' }}</time><span>{{ selection.kind | default: post.categories.first | default: '技术笔记' }}</span></p>
        <h3><a href="{{ post.url | relative_url }}">{{ post.title | escape }}</a></h3>
        {% if post.description %}<p class="post-description">{{ post.description | escape }}</p>{% endif %}
      </article>
      {% endfor %}
    </div>
  </section>
  {% endfor %}
</div>
