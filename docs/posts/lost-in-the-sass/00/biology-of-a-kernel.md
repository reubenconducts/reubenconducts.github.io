---
title: Biology of a CUDA Kernel
tags: [CUDA, hardware]
date: 2026-01-31
---

# {{ $frontmatter.title }}

GPU programming involves careful coordination between the CPU (*"host"*) and GPU (*"device"*, sometimes many). In this first post, we will dive into the biology of a CUDA kernel: its *life cycle* and its *anatomy*.

::: info
This article is under construction! Check back soon.
:::

## Life Cycle

### Writing

#### CUDA C++ syntax and extensions

#### Function qualifiers

#### Other languages

### Compilation

#### CUDA to PTX

#### PTX to SASS

#### JIT compilation

### Runtime

#### Memory allocation & transfers

#### Kernel launch

#### Kernel execution

#### Synchronization & cleanup

## Anatomy

### Thread Hierarchy

#### Threads, Warps, Blocks, Grid

#### Thread indexing

### Memory Hierarchy

#### Registers

#### Shared memory

#### L1/L2 Cache

#### Global memory

#### Constant & texture memory

### Execution Model

#### SIMT

#### Warp scheduling & divergence

#### Occupancy

### Hardware Structure

#### Streaming Multiprocessors

#### CUDA cores and specialized cores

::: example Example of PTX and SASS
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

## References

### Official NVIDIA Documentation

1. NVIDIA. [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/). NVIDIA Developer Documentation.
2. NVIDIA. [PTX ISA Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/). Parallel Thread Execution instruction set architecture reference.
3. NVIDIA. [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html). Documentation for cuobjdump and nvdisasm.
4. NVIDIA. [CUDA Toolkit Documentation 13.1](https://docs.nvidia.com/cuda/). Complete CUDA toolkit documentation.
5. NVIDIA. [CUDA Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html). Official guide on grids, blocks, and threads.

### Execution Model & Thread Hierarchy

6. Wikipedia contributors. [Thread Block (CUDA programming)](https://en.wikipedia.org/wiki/Thread_block_(CUDA_programming)). Wikipedia.
7. Chng, Peter. [How 2D and 3D thread blocks are linearized into warps in CUDA](https://peterchng.com/blog/2024/03/09/how-are-2d-and-3d-thread-blocks-linearized-into-warps-in-cuda/). March 2024.

### Memory Hierarchy

8. Modal. [CUDA Memory Hierarchy](https://modal.com/gpu-glossary/device-software/memory-hierarchy). GPU Glossary.
9. ARC Compute. [Memory Hierarchy of GPUs](https://www.arccompute.io/arc-blog/gpu-101-memory-hierarchy). GPU 101 Blog.

### SIMT & Warp Execution

10. Cornell University. [SIMT and Warps](https://cvw.cac.cornell.edu/gpu-architecture/gpu-characteristics/simt_warp). Cornell Virtual Workshop - GPU Architecture.
11. Harris, Mark. [Using CUDA Warp-Level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/). NVIDIA Technical Blog.
12. Collange, Sylvain. [GPU Architecture: Revisiting the SIMT Execution Model](https://www.irisa.fr/alf/downloads/collange/cours/hpca2020_gpu_0.pdf). HPCA 2020.

### SASS Assembly

13. PNF Software. [Reversing Nvidia GPU's SASS code](https://www.pnfsoftware.com/blog/reversing-nvidia-cuda-sass-code/). JEB in Action Blog.
14. cloudcores. [CuAssembler](https://github.com/cloudcores/CuAssembler). Unofficial CUDA assembler for SASS. GitHub.
