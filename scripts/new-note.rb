#!/usr/bin/env ruby
# frozen_string_literal: true

# Creates notes/_posts/<date>-<slug>.md from notes/_template.md.
#
#   ruby scripts/new-note.rb --slug nccl-topology --title "标题" --topics training-infra
#
# Run with --help for all options. The convention is described in README.md.

require "json"
require "optparse"
require "yaml"

ROOT = File.expand_path("..", __dir__)
TEMPLATE = File.join(ROOT, "notes", "_template.md")
NOTES_DIR = File.join(ROOT, "notes", "_posts")
TOPIC_IDS = YAML.safe_load_file(File.join(ROOT, "_data", "note_topics.yml")).map { |topic| topic["id"] }

options = { topics: [], tags: [] }
parser = OptionParser.new do |opts|
  opts.banner = "用法: ruby scripts/new-note.rb --slug <slug> --title <标题> --topics <id[,id]> [选项]"
  opts.on("--slug SLUG", "URL 里的短名，小写英文、数字和连字符") { |value| options[:slug] = value }
  opts.on("--title TITLE", "札记标题") { |value| options[:title] = value }
  opts.on("--topics IDS", Array, "主题 id，逗号分隔；可选值: #{TOPIC_IDS.join(', ')}") { |value| options[:topics] = value }
  opts.on("--tags TAGS", Array, "关键词，逗号分隔（可选）") { |value| options[:tags] = value }
  opts.on("--date DATE", "发布日期 YYYY-MM-DD，默认今天（北京时间）") { |value| options[:date] = value }
end
parser.parse!

def fail_with(message, parser)
  warn "错误: #{message}\n\n#{parser}"
  exit 1
end

fail_with("需要 --slug、--title 和 --topics", parser) unless options[:slug] && options[:title] && options[:topics].any?
fail_with("slug 只能用小写英文、数字和连字符: #{options[:slug]}", parser) unless options[:slug].match?(/\A[a-z0-9]+(-[a-z0-9]+)*\z/)
unknown = options[:topics] - TOPIC_IDS
fail_with("未知主题: #{unknown.join(', ')}（主题表在 _data/note_topics.yml）", parser) if unknown.any?

now = Time.now.getlocal("+08:00")
if options[:date]
  fail_with("日期格式应为 YYYY-MM-DD: #{options[:date]}", parser) unless options[:date].match?(/\A\d{4}-\d{2}-\d{2}\z/)
  date = options[:date]
  # Midnight, so a note dated today is never "in the future" and skipped by Jekyll.
  timestamp = "#{date} 00:00:00 +0800"
else
  date = now.strftime("%Y-%m-%d")
  timestamp = now.strftime("%Y-%m-%d %H:%M:%S +0800")
end

if Dir.glob(File.join(NOTES_DIR, "*-#{options[:slug]}.md")).any?
  fail_with("已有同名 slug 的札记: #{options[:slug]}", parser)
end

front_matter, body = File.read(TEMPLATE).split(/^---[ \t]*\n/, 3).drop(1)
front_matter = front_matter.lines.reject { |line| line.start_with?("#") }.join
front_matter = front_matter
  .sub('"__TITLE__"') { options[:title].to_json }
  .sub("__DATE__", timestamp)
  .sub("__TOPICS__", options[:topics].join(", "))
  .sub("tags: []", "tags: [#{options[:tags].join(', ')}]")

path = File.join(NOTES_DIR, "#{date}-#{options[:slug]}.md")
Dir.mkdir(NOTES_DIR) unless Dir.exist?(NOTES_DIR)
File.write(path, "---\n#{front_matter}---\n#{body}")
puts "已创建 #{path.delete_prefix("#{ROOT}/")}"
puts "填完所有“待填”后，运行 bundle exec jekyll build && bundle exec ruby scripts/check-site.rb"
