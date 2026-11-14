# yuanbaopeng.com

袁宝鹏的工程手记，<https://www.yuanbaopeng.com>。Jekyll 站点，由 GitHub Pages 从 `master` 分支直接构建。

## 本地预览

需要 Ruby 3.x 和 Bundler。`Gemfile` 把 `github-pages` 固定在线上正在用的版本（见 <https://pages.github.com/versions.json>），本地构建结果与线上一致。

```sh
bundle install
bundle exec jekyll serve          # http://localhost:4000
```

提交前检查：

```sh
bundle exec jekyll build
bundle exec ruby scripts/check-site.rb
```

`check-site.rb` 会检查文章与札记的 front matter、`_data/selected_posts.yml` 里的 slug、站内链接与锚点、每页的标题和描述、图片的 alt，以及 `feed.xml`、`sitemap.xml`。加上 `--external` 还会请求一遍外部链接。同样的检查在每个 pull request 上自动运行（`.github/workflows/build.yml`）。

## 目录

| 路径 | 内容 |
| --- | --- |
| `_posts/` | 文章，显示在「文章」页 |
| `notes/_posts/` | 对话札记，显示在「札记」页 |
| `notes/_template.md` | 札记模板，不会被发布 |
| `_data/note_topics.yml` | 札记的主题表 |
| `_data/selected_posts.yml` | 首页「从这几篇读起」 |
| `_data/questions.yml` | 首页「最近在想」和「此刻」页 |
| `_layouts/`、`_includes/`、`assets/main.scss` | 版式与样式 |
| `scripts/` | `new-note.rb` 新建札记，`check-site.rb` 检查，`build-sites.mjs` 把 `_site` 打包成单文件 Worker |

## 写一篇文章

在 `_posts/` 下新建 `YYYY-MM-DD-slug.md`：

```yaml
---
title: "标题"
date: 2026-08-28 20:00:00 +0800
description: 一两句话的摘要，用于列表、搜索结果和 RSS。
categories: AI-Infra        # 可选；会成为 URL 的一部分，发布后不要改
tags: [训练平台, 调度]
reading_time: 16            # 可选，分钟
mathjax: true               # 可选，正文里有公式时打开
---
```

- 新文章的 slug 用英文，URL 更干净。旧文章的中文文件名和 `categories` 决定了它们现在的地址，不要改。
- 正文小标题从 `##` 开始，`#` 留给页面标题。
- 公式用 `$$ ... $$`（独立成行）或 `$ ... $`（行内），并在 front matter 里写 `mathjax: true`。

## 发布一篇对话札记

对话札记是从与 AI 助手的对话里整理出来的笔记。它们是普通的 Jekyll 文章，只是放在 `notes/_posts/` 下：会进入 RSS 和 sitemap，在「札记」页按主题归档，不出现在「文章」页。

1. 新建文件。用脚本：

   ```sh
   ruby scripts/new-note.rb --slug nccl-topology --title "标题" --topics training-infra --tags NCCL,GPU
   ```

   或者手动把 `notes/_template.md` 复制为 `notes/_posts/YYYY-MM-DD-slug.md`，删掉开头的注释行。

2. 填写 front matter：

   ```yaml
   ---
   title: "标题"
   date: 2026-11-14 21:30:00 +0800
   description: 一两句话写清这篇札记回答了什么问题。
   topics: [training-infra]     # 必填，取自 _data/note_topics.yml 的 id，一般一到两个
   tags: [NCCL, GPU]            # 可选，自由关键词
   updated: 2026-12-01          # 可选，之后有实质修改时填写
   mathjax: true                # 可选
   ---
   ```

   | 主题 id | 名称 |
   | --- | --- |
   | `llm-agents` | 大模型与 Agent |
   | `ai-coding` | AI 编程 |
   | `training-infra` | 训练与推理基础设施 |
   | `recsys-llm` | 推荐系统与大模型 |
   | `people-teams` | AI 人物与团队 |

3. 写正文。模板的结构是：问题、要点、展开、还没确认的、参考。可以增删小节，但建议保留「还没确认的」：对话里的说法没有核对过，就写在这里。

4. 运行上面的构建与检查，然后提交。还没写完的札记可以先加一行 `published: false`，它不会被发布。

约定：

- 文件名 `YYYY-MM-DD-slug.md`，slug 只用小写英文、数字和连字符。地址是 `/notes/YYYY/MM/slug/`，发布后不要改 slug。
- `date` 带时区（`+0800`），日期与文件名一致。日期在未来的札记不会被发布。
- 新增主题：在 `_data/note_topics.yml` 里加一项。已经用过的 id 不要改，它也是「札记」页的锚点。
- 每篇札记页首会自动带一句说明：内容整理自与 AI 助手的对话。措辞在 `_layouts/post.html` 里。
- 没有核对过的说法放进「还没确认的」；涉及具体的人、团队和数字时，在「参考」里给出来源。

## 部署

推送到 `master` 后，GitHub Pages 自动构建并发布到 `www.yuanbaopeng.com`（`CNAME`）。GitHub Pages 用自己的 gem 版本构建，不读取 `Gemfile.lock`，也只支持它白名单里的插件，所以不要加自定义插件。
