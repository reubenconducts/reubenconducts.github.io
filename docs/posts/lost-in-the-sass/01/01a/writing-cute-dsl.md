---
title: Writing and Profiling Kernels in CuTe DSL
tags: [CUDA, hardware, CuTe DSL, programming]
date: 2026-02-01
---

# {{ $frontmatter.title }}

Writing CUDA kernels is more accessible than ever, thanks in large part to NVIDIA's recently-released _CuTe DSL_, a Python domain-specific-language that allows programmers to access the entirety of the CUDA programming model within Python.

::: info
Under construction!
:::

CuTe DSL is fully isomorphic with CUDA/C++, in that every kernel one could write in CUDA/C++ could also be written in CuTe DSL (note however that not all utilities and libraries have yet been ported over, notably `Cub`). This means that the basic structure of a CuTe DSL kernel is identical to that of a CUDA/C++ kernel, though with more Pythonic syntax and paradigms. Every kernel consists of _host_ code, run on the CPU, and _device_ code, run on the GPU. The following is a "hello world" kernel written in CuTe DSL, together with its corresponding CUDA/C++ implementation.

::: details
::: code-group

```python:line-numbers
@cute.kernel
def hello_world_kernel():
    cute.printf("Hello, world!") # [!code focus]

@cute.jit
def hello_world_host():
    hello_world_kernel().launch(grid=[1], block=[1]) # [!code focus]

def run_hello_world():
    hello_world_host()
```

```cpp:line-numbers
void __global__ hello_world_kernel() {
    printf("Hello, world!"); // [!code focus]
}
void hello_world_host() {
    hello_world_kernel<<<1, 1>>>(); // [!code focus]
}
```

:::

To profile a kernel, use `ncu`:

```bash
ncu -o output.ncu-rep --set full --import-source true python kernel.py
```

```bash
$ npm run build
# Output
Building site...
✓ Build complete in 2.3s

$ ls -la
total 48
drwxr-xr-x  12 user  staff   384 Feb  3 14:27 .
```
