---
layout: default
description: 袁宝鹏的个人技术站，记录 AI 工程、GPU 性能、复杂系统与研发效率实践。
image: /assets/images/og.png
---

<section class="hero">
  <div class="site-shell hero-grid">
    <div class="hero-copy">
      <p class="eyebrow"><span></span> AI ENGINEERING · SYSTEMS · PRACTICE</p>
      <h1>把复杂系统，<br><em>做得更清楚。</em></h1>
      <p class="hero-lede">我是袁宝鹏，长期关注 AI 工程、GPU 性能与研发效率。这里记录真实问题的拆解、取舍与实现。</p>
      <div class="hero-actions">
        <a class="button button-primary" href="#writing">阅读技术文章 <span aria-hidden="true">→</span></a>
        <a class="button button-quiet" href="{{ '/about/' | relative_url }}">了解我</a>
      </div>
    </div>

    <aside class="focus-card" aria-label="当前关注方向">
      <div class="focus-card-top">
        <span class="status-dot" aria-hidden="true"></span>
        <span>NOW / 2026</span>
      </div>
      <p class="focus-label">CURRENT FOCUS</p>
      <ul class="focus-list">
        <li><span>01</span>AI 训练与推理工程</li>
        <li><span>02</span>Agent 驱动研发流程</li>
        <li><span>03</span>系统性能与可靠性</li>
      </ul>
      <div class="focus-path" aria-label="技术关注轨迹">
        <span>C++</span><i>→</i><span>CUDA</span><i>→</i><span>AI Infra</span>
      </div>
    </aside>
  </div>
</section>

<section class="track-section" aria-labelledby="tracks-title">
  <div class="site-shell">
    <div class="section-heading">
      <p class="section-kicker">WHAT I WRITE ABOUT</p>
      <h2 id="tracks-title">三条长期主线</h2>
      <p>不追逐信息流，沉淀可以重复使用的判断和方法。</p>
    </div>
    <div class="track-grid">
      <article class="track-card track-card-blue">
        <p class="track-number">01</p>
        <h3>AI 工程</h3>
        <p>训练、推理、任务调度与可观测性，从系统视角理解 AI 基础设施。</p>
        <span class="track-tag">INFRASTRUCTURE</span>
      </article>
      <article class="track-card track-card-orange">
        <p class="track-number">02</p>
        <h3>GPU 与性能</h3>
        <p>CUDA、TensorRT 与性能分析，关注每一次抽象背后的真实开销。</p>
        <span class="track-tag">PERFORMANCE</span>
      </article>
      <article class="track-card track-card-ink">
        <p class="track-number">03</p>
        <h3>研发效率</h3>
        <p>Agent、工具链与工程工作流，让经验变成团队和个人的复利。</p>
        <span class="track-tag">DEVELOPER EXPERIENCE</span>
      </article>
    </div>
  </div>
</section>

<section class="writing-section" id="writing" aria-labelledby="writing-title">
  <div class="site-shell">
    <div class="section-heading section-heading-row">
      <div>
        <p class="section-kicker">SELECTED WRITING</p>
        <h2 id="writing-title">技术文章</h2>
      </div>
      <p class="archive-note">从底层机制出发，记录问题是怎样被理解和解决的。</p>
    </div>
    <div class="post-list">
      {% assign featured_posts = site.posts | where_exp: "post", "post.title != 'Welcome to Jekyll!'" %}
      {% for post in featured_posts limit: 5 %}
      <article class="post-row">
        <div class="post-meta-block">
          <time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%Y.%m.%d" }}</time>
          <span>{{ post.categories | first | default: 'NOTE' }}</span>
        </div>
        <div class="post-summary">
          <h3><a href="{{ post.url | relative_url }}">{{ post.title | escape }}</a></h3>
          <p>{{ post.excerpt | strip_html | strip_newlines | truncate: 105 }}</p>
        </div>
        <a class="post-arrow" href="{{ post.url | relative_url }}" aria-label="阅读《{{ post.title | escape }}》">↗</a>
      </article>
      {% endfor %}
    </div>
    <a class="text-link" href="{{ '/writing/' | relative_url }}">查看全部文章 <span aria-hidden="true">→</span></a>
  </div>
</section>

<section class="home-cta">
  <div class="site-shell cta-inner">
    <div>
      <p class="section-kicker">KEEP THINKING IN PUBLIC</p>
      <h2>把做过的事，变成可以复用的经验。</h2>
    </div>
    <a class="button button-light" href="{{ '/about/' | relative_url }}">关于这个站 <span aria-hidden="true">→</span></a>
  </div>
</section>
