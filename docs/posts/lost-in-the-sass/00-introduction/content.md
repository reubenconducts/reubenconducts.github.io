---
title: Lost in the SASS 0 -- Introduction
tags: [CUDA, hardware]
date: 2026-01-31
---

# {{ $frontmatter.title }}

As we push closer to the edge of peak computational performance, it becomes crucial to know how to talk to (or at least listen to) whatever hardware we are optimizing for.

## Explanation of the series

The point of this ongoing series is to spelunk into the depths of NVIDIA GPU programming, exploring compiler optimizations, how code is transformed into SASS, and what we as programmers can do to leverage this newfound knowledge for our benefit.

::: code-group

```sass
LDG.E R17, desc[UR12][R38.64]
```

```python
@cute.jit
def layout_test():
    # layout == layout -> layout
    layout = cute.make_layout((4, 4), stride=(4, 1)) # [!code focus]
    print(f"{layout = }")
```
:::
