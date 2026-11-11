# bananighosh.github.io

Personal blog. Jekyll, built natively by GitHub Pages.

## Publish a post

1. Add `_posts/YYYY-MM-DD-slug.md`:

   ```yaml
   ---
   title: "Your title"
   summary: One-line description shown on the index and in link previews.
   tags: [gpu, vllm]
   mermaid: false   # true if the post has ```mermaid blocks
   ---
   ```

2. Images go in `assets/img/`.
3. Commit and push to `main`. Live in about a minute.

Math (`$...$`, `$$...$$`) is on for every post.

## Preview locally

```bash
bundle install
bundle exec jekyll serve --livereload
```
