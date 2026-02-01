---
title: Biology of a CUDA Kernel
tags: [CUDA, hardware]
date: 2026-01-31
---

# {{ $frontmatter.title }}

GPU programming involves careful coordination between the CPU (*"host"*) and GPU (*device*, sometimes many). In this first post, we will dive into the biology of a CUDA kernel: its *life cycle* and its *anatomy*.

::: info
This article is under construction! Check back soon.
:::


::: example
::: code-group
```ptx
add.s32  %r1, %r2, %r3;         // Integer addition
sub.f32  %f1, %f2, %f3;         // Float subtraction
mul.lo.s32  %r1, %r2, %r3;      // Integer multiply (low 32 bits)
div.rn.f32  %f1, %f2, %f3;      // Float division (round nearest)
mad.lo.s32  %r1, %r2, %r3, %r4; // Multiply-add
fma.rn.f32  %f1, %f2, %f3, %f4; // Fused multiply-add
```
```sass
IADD3 R0, R0, c[0x0][0x94], RZ
IMAD R0, R3, c[0x0][0x0], R0
IMAD.WIDE R2, R0, R7, c[0x0][0x160]
LDG.E.SYS R2, [R2]
```
:::
