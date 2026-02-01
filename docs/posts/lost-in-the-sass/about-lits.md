---
title: "Lost in the SASS: Exploring NVIDIA GPU Assembly"
description: A blog post series diving deep into NVIDIA GPU assembly programming
tags: [GPU, SASS, NVIDIA, Assembly, CUDA]
---

# Lost in the SASS

**A journey through NVIDIA GPU assembly programming**

As we push closer to peak computational performance on GPUs, understanding the assembly language becomes crucial. This series explores NVIDIA's PTX (Parallel Thread Execution) intermediate language and SASS (Streaming ASSembler) machine code, uncovering how CUDA kernels are transformed and optimized at the lowest level.

::: aside What You'll Learn
From reading PTX and SASS disassembly to understanding instruction scheduling, memory access patterns, and warp-level optimizations, this series covers the essential knowledge for serious GPU kernel developers.
:::

## Series Contents

::: info Current Status
This series is actively being written. New installments will be added regularly.
:::

### Introduction: Biology of a CUDA Kernel
[Under construction](/posts/lost-in-the-sass/00/biology-of-a-kernel)

Understanding the life cycle and anatomy of a CUDA kernel from development to execution.

### Part 1: Instruction Taxonomy, PTX vs. SASS
*Coming soon*

#### *Appendix A: Writing and Profiling a CuTe DSL Kernel*
[Under construction](/posts/lost-in-the-sass/01/01a/writing-cute-dsl)

#### *Appendix B: Registers*
*Coming soon*

### Part 2: Arithmetic
*Coming soon*

### Part 3: Memory Management
*Coming soon*

### Part 4: Control Flow
*Coming soon*

### Part 5: Comparison and Predication
*Coming soon*

### Part 6: Synchronization
*Coming soon*

### Part 7: Tensor Operations
*Coming soon*

### Part 8: Asynchronous Operations
*Coming soon*

### Part 9: Scheduling and Hazards
*Coming soon*

#### *Appendix A: Register Allocation, Spills, and Consequences*
*Coming soon*

### Part 10: Bit Manipulation and Warp-level Primitives
*Coming soon*

### Epilogue: Dissecting a GEMM Kernel on SM80, 90, and 100
*Coming soon*

---

## Resources

- [Instruction Glossary](/glossary) - Quick reference for PTX and SASS instructions
- [External Resources](/resources) - Tools and documentation links

---

*Questions or suggestions? [Get in touch](mailto:reuben-stern@colfax-intl.com)*
