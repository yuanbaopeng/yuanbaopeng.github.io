#!/usr/bin/env ruby
# frozen_string_literal: true

# Checks the sources and the built site. Run after `bundle exec jekyll build`:
#
#   bundle exec ruby scripts/check-site.rb             # sources + _site
#   bundle exec ruby scripts/check-site.rb --external  # also request external links
#
# Exits non-zero when anything is wrong, so it can gate a pull request.

require "date"
require "net/http"
require "nokogiri"
require "uri"
require "yaml"

ROOT = File.expand_path("..", __dir__)
SITE = File.join(ROOT, "_site")
CONFIG = YAML.safe_load_file(File.join(ROOT, "_config.yml"))
SITE_URL = CONFIG.fetch("url")
PLACEHOLDER = /待填|__[A-Z]+__/.freeze

@errors = []
def error(where, message)
  @errors << "#{where}: #{message}"
end

def relative(path)
  path.delete_prefix("#{ROOT}/")
end

def front_matter(path)
  text = File.read(path)
  match = text.match(/\A---\s*\n(.*?)\n---\s*\n/m)
  return [nil, text] unless match

  [YAML.safe_load(match[1], permitted_classes: [Date, Time]) || {}, match.post_match]
end

def blank?(value)
  value.nil? || value.to_s.strip.empty?
end

# --- Sources -----------------------------------------------------------------

articles = Dir.glob(File.join(ROOT, "_posts", "*.{md,markdown}")).sort
notes = Dir.glob(File.join(ROOT, "notes", "_posts", "*.{md,markdown}")).sort
topic_ids = YAML.safe_load_file(File.join(ROOT, "_data", "note_topics.yml")).map { |topic| topic["id"] }
slugs = {}

(articles + notes).each do |path|
  name = File.basename(path, ".*")
  where = relative(path)
  file_date, slug = name.match(/\A(\d{4}-\d{2}-\d{2})-(.+)\z/)&.captures
  unless slug
    error(where, "文件名应为 YYYY-MM-DD-slug.md")
    next
  end
  error(where, "slug 与 #{slugs[slug]} 重复") if slugs[slug]
  slugs[slug] = where

  data, body = front_matter(path)
  if data.nil?
    error(where, "缺少 front matter")
    next
  end
  %w[title date description].each { |key| error(where, "front matter 缺少 #{key}") if blank?(data[key]) }
  if data["date"].is_a?(Time) && data["date"].getlocal("+08:00").strftime("%Y-%m-%d") != file_date
    error(where, "front matter 的 date 与文件名日期不一致")
  end

  next unless notes.include?(path)

  error(where, "札记的 slug 只能用小写英文、数字和连字符") unless slug.match?(/\A[a-z0-9]+(-[a-z0-9]+)*\z/)
  topics = Array(data["topics"])
  error(where, "topics 至少填一个主题 id") if topics.empty?
  unknown = topics - topic_ids
  error(where, "topics 含未知主题: #{unknown.join(', ')}（主题表在 _data/note_topics.yml）") if unknown.any?
  if data.values.any? { |value| value.to_s.match?(PLACEHOLDER) } || body.match?(PLACEHOLDER)
    error(where, "还有没填完的模板占位（“待填”或 __XXX__）")
  end
end

selected = YAML.safe_load_file(File.join(ROOT, "_data", "selected_posts.yml")) || []
selected.each do |entry|
  next if slugs.key?(entry["slug"])

  error("_data/selected_posts.yml", "找不到 slug 为 #{entry['slug']} 的文章")
end

# --- Built site --------------------------------------------------------------

unless File.exist?(File.join(SITE, "index.html"))
  warn "找不到 _site/index.html，请先运行 bundle exec jekyll build"
  exit 2
end

def resolve(url_path)
  path = File.join(SITE, url_path)
  candidates = url_path.end_with?("/") ? [File.join(path, "index.html")] : [path, "#{path}.html", File.join(path, "index.html")]
  candidates.find { |candidate| File.file?(candidate) }
end

documents = {}
Dir.glob(File.join(SITE, "**", "*.html")).sort.each do |file|
  documents[file] = Nokogiri::HTML(File.read(file))
end

def anchor?(document, fragment)
  id = URI::DEFAULT_PARSER.unescape(fragment)
  document.xpath("//*[@id=$id or @name=$id]", nil, id: id).any?
end

external = {}
documents.each do |file, document|
  where = relative(file)
  page_url = "/#{file.delete_prefix("#{SITE}/")}"

  error(where, "缺少 <title>") if blank?(document.at_css("title")&.text)
  error(where, "缺少 meta description") if blank?(document.at_css('meta[name="description"]')&.[]("content"))
  headings = document.css("h1").size
  error(where, "应当只有一个 <h1>，实际有 #{headings} 个") unless headings == 1
  document.css("img").each do |image|
    error(where, "图片缺少 alt: #{image['src']}") if image["alt"].nil?
  end

  targets = document.css("a[href], link[href]").map { |node| node["href"] } +
            document.css("img[src], script[src]").map { |node| node["src"] }
  targets.uniq.each do |target|
    target = target.delete_prefix(SITE_URL) if target.start_with?("#{SITE_URL}/")
    next if target.empty? || target.start_with?("mailto:", "tel:", "data:")

    if target.match?(%r{\A(?:[a-z][a-z0-9+.-]*:)?//}i)
      (external[target] ||= []) << where
      next
    end

    path, fragment = target.split("#", 2)
    path = path.to_s.split("?", 2).first.to_s
    url_path = path.empty? ? page_url : URI::DEFAULT_PARSER.unescape(URI.join("http://site#{URI::DEFAULT_PARSER.escape(page_url)}", path).path)
    resolved = resolve(url_path)
    if resolved.nil?
      error(where, "站内链接失效: #{target}")
    elsif !blank?(fragment) && documents[resolved] && !anchor?(documents[resolved], fragment)
      error(where, "锚点不存在: #{target}")
    end
  end
end

%w[feed.xml sitemap.xml].each do |name|
  path = File.join(SITE, name)
  if File.file?(path)
    begin
      Nokogiri::XML(File.read(path)) { |config| config.strict }
    rescue Nokogiri::XML::SyntaxError => e
      error("_site/#{name}", "XML 不合法: #{e.message}")
    end
  else
    error("_site/#{name}", "没有生成")
  end
end

# --- External links (opt-in: slow, and other sites fail for their own reasons) --

if ARGV.include?("--external")
  external.sort.each do |url, pages|
    uri = URI.parse(url.sub(%r{\A//}, "https://").gsub(/[^\x00-\x7F]/) { |char| URI.encode_www_form_component(char) })
    status =
      begin
        Net::HTTP.start(uri.host, uri.port, use_ssl: uri.scheme == "https", open_timeout: 10, read_timeout: 15) do |http|
          http.request_get(uri.request_uri.empty? ? "/" : uri.request_uri, "User-Agent" => "Mozilla/5.0 (link check)") { |response| break response.code }
        end
      rescue StandardError => e
        e.class.name
      end
    puts format("%-22s %s  (%s)", status, url, pages.uniq.join(", ")) unless status.to_s.match?(/\A[23]\d\d\z/)
  end
end

if @errors.empty?
  puts "检查通过：#{articles.size} 篇文章，#{notes.size} 篇札记，#{documents.size} 个页面。"
else
  warn @errors.join("\n")
  warn "\n共 #{@errors.size} 个问题。"
  exit 1
end
