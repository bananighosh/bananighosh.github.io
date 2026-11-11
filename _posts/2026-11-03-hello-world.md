---
title: "Template post: math, code, tables, diagrams"
summary: A reference post showing every formatting feature this site supports. Delete it once your first real post is up.
tags: [meta]
mermaid: true
---

This post exists so you can copy its patterns. Delete it before launch.

## Inline and display math

Arithmetic intensity is $I = \text{FLOPs} / \text{bytes moved}$. A kernel is memory bound when

$$
I < \frac{P_{\text{peak}}}{B_{\text{HBM}}}
$$

## Code

```python
achieved_bw = bytes_moved / elapsed_s
print(f"{achieved_bw / 1e12:.2f} TB/s")
```

## Tables

| GPU  | HBM BW (TB/s) | Notes          |
|------|---------------|----------------|
| H100 | 3.35          | SXM            |
| B200 | 8.0           | fill from spec |

## Diagrams

```mermaid
flowchart LR
  A[Prefill] --> B[KV cache] --> C[Decode]
```

## Images

Put files in `assets/img/` and reference them:

```markdown
![Roofline](/assets/img/roofline.png)
```

> Blockquotes work for callouts.
