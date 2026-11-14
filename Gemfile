source "https://rubygems.org"

# GitHub Pages builds this site with its own pinned gem set and ignores this
# file. Keep the version below in sync with https://pages.github.com/versions.json
# so that a local `bundle exec jekyll build` behaves like production.
gem "github-pages", "232", group: :jekyll_plugins

# `jekyll serve` needs webrick on Ruby 3.0+.
gem "webrick", "~> 1.7"

# Silences the "install faraday-retry" notice printed by jekyll-github-metadata.
gem "faraday-retry", "~> 2.2"

# Windows and JRuby do not ship zoneinfo files.
platforms :windows, :jruby do
  gem "tzinfo-data"
end

# Performance-booster for watching directories on Windows
gem "wdm", "~> 0.1.1", platforms: :windows
