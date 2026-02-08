---
title: Flex Linear Attention -  Fast and Flexible Fused Linear Attention Implementation
---
# NEW Flex Linear Attention

**A Computational Grammar for Recurrent Sequence Models**

Author: Claude · GPU Kernel Engineering · February 2026 · Draft v3

---

## Executive Summary

Every fused chunkwise linear attention kernel—Mamba-2, GLA, DeltaNet, Mamba-3—performs the same four matrix multiplications per chunk, with the same warp-specialized pipeline, on the same hardware. What changes between variants is a small amount of scalar preprocessing between pipeline stages. This project exploits that structural regularity to build a single parameterized kernel skeleton that serves the entire family.

But the deeper observation is that the kernel's computational structure—4 matmuls, a persistent state matrix, chunk-local interactions, and inter-chunk state propagation—is **more general than any single mathematical framework**. The test-time regression (TTR) interpretation is one way to derive update rules that fit this structure. Contrastive objectives, information-theoretic compression, predictive state representations, and spectral methods all produce update rules that fit the same 4-matmul pattern. The kernel doesn't care *why* the state is being updated; it cares about *shapes and data flow*.

This reframing changes the project's value proposition. The engineering deliverable is a set of fused forward kernels that match or exceed existing hand-written implementations for 4–5 known variants. The research deliverable is a *computational grammar*—a well-defined set of operations the hardware executes efficiently—that makes it cheap to explore what else this pattern can do. The cost of trying a novel sequence model drops from "write a new CUDA kernel" to "write a new `state_propagate` function."

The project proceeds in four phases: (1) extract and validate the skeleton with SSD-family variants, (2) extend to delta-rule variants via a preprocessing kernel strategy, (3) add Mamba-3 features (trapezoidal discretization, RoPE, MIMO), and (4) instantiate and evaluate novel objectives that go beyond the regression framework.

---

## 1. Mathematical Background

### 1.1 The Test-Time Regression Perspective

Wang et al.'s TTR framework unifies existing linear attention variants as instances of online weighted least-squares regression. The model maintains a memory matrix:

$$M_t = \arg\min_M \sum_{i=1}^{t} \gamma_i^{(t)} \| v_i - M \phi(k_i) \|^2$$

The analytical solution is $M_t = V_t^\top \Gamma_t \Phi_t (\Phi_t^\top \Gamma_t \Phi_t)^{-1}$. All practical variants drop the inverse covariance term $(\Phi_t^\top \Gamma_t \Phi_t)^{-1}$, yielding a one-step gradient descent approximation:

$$M_t \approx V_t^\top \Gamma_t \Phi_t = \sum_{i=1}^{t} \gamma_i^{(t)} v_i \phi(k_i)^\top$$

With geometrically decaying weights $\gamma_i^{(t)} = \prod_{j=i+1}^{t} \alpha_j$ and identity feature map, this admits the recurrence:

$$M_t = \alpha_t M_{t-1} + v_t k_t^\top, \qquad y_t = M_t q_t$$

The three TTR axes parameterize the design space:

| TTR Axis | Mathematical Object | Examples |
|----------|-------------------|----------|
| Regression weights $\{\gamma_i^{(t)}\}$ | Decay structure | Mamba-2: $\alpha_t = e^{\Delta_t A_t}$; GLA: $\alpha_t = \gamma_t$; RetNet: $\alpha_t = e^{-\lambda}$ |
| Function class $\mathcal{M}$ | Feature map $\phi$ | Identity, RoPE ($e^{i\theta_t}$ rotation), random features |
| Optimization algorithm | Update rule | One-step GD ($M \leftarrow \alpha M + vk^\top$), delta rule ($M \leftarrow (I - \beta kk^\top)M + \beta vk^\top$) |

### 1.2 The Two Families

**SSD family.** Scalar or diagonal state transition:

$$M_t = \alpha_t M_{t-1} + v_t k_t^\top$$

This covers Mamba-2 ($\alpha_t = e^{\Delta_t A_t}$, data-dependent), GLA ($\alpha_t = \gamma_t$, per-head learnable gate), RetNet ($\alpha_t = e^{-\lambda}$, fixed), RWKV-6 (data-dependent scalar with channel mixing), and mLSTM ($\alpha_t = \exp(f_t)$, forget gate). Mamba-3 extends this with MIMO (rank-$r$ updates: $H_t = \alpha_t H_{t-1} + B_t X_t^\top$ where $B_t \in \mathbb{R}^{N \times r}$), trapezoidal discretization, and data-dependent RoPE.

The pre-inter work for SSD variants is element-wise: scale keys by decay, multiply state by decay, accumulate. Cost: $O(ND)$ FMAs per chunk. Trivially cheap.

**Delta rule family.** State-dependent transition:

$$M_t = (I - \beta_t k_t k_t^\top) M_{t-1} + \beta_t v_t k_t^\top$$

The Householder factor $(I - \beta_t k_t k_t^\top)$ erases the old association at key $k_t$ before writing the new one. This is a rank-1 approximation to incorporating the inverse covariance. Within a chunk, the product $\prod_{j=1}^{C} (I - \beta_j k_j k_j^\top) = I - WY^\top$ is computed via the WY representation, where $W, Y \in \mathbb{R}^{N \times C}$ are built sequentially:

$$w_t = \beta_t k_t - W_{:,:t} (Y_{:,:t}^\top (\beta_t k_t)), \qquad Y_{:,t} = k_t$$

The WY construction is $O(C^2 N)$ and sequential in $C$. Its implications for the fused kernel are analyzed in §5.

### 1.3 Chunkwise Decomposition

All variants share a chunkwise structure. Partition the sequence into chunks of size $C$. The output for position $t$ in chunk $c$ decomposes as:

$$y_t = \underbrace{q_t^\top S_c}_{\text{inter-chunk: query accumulated state}} + \underbrace{\sum_{j \in \text{chunk}} L[t,j] \cdot q_t^\top k_j v_j^\top}_{\text{intra-chunk: attend within current chunk}}$$

where $S_c$ is the recurrent state entering chunk $c$ and $L[t,j]$ is the structured mask (causal, with decay). In matrix form, each chunk requires exactly 4 matrix multiplications:

| Phase | Computation | Shape | Purpose |
|-------|-------------|-------|---------|
| INTRA1 | $Q \times K^\top$ | $(L, N) \times (N, L) \to (L, L)$ | Raw intra-chunk attention scores |
| INTRA2 | $\tilde{A} \times V$ | $(L, L) \times (L, D) \to (L, D)$ | Masked intra-chunk output |
| INTER1 | $\tilde{K}^\top \times V$ | $(N, L) \times (L, D) \to (N, D)$ | This chunk's contribution to state |
| INTER2 | $Q \times S$ | $(L, N) \times (N, D) \to (L, D)$ | Output from accumulated state |

Here $\tilde{A}$ is the INTRA1 result after mask application (causal mask × decay structure), and $\tilde{K}$ is the preprocessed key matrix (scaled by decay). The final output is $Y = Y_{\text{intra}} + Y_{\text{inter}}$ with variant-specific scaling.

These 4 matmuls are structurally identical across all known linear attention variants. The data flows between them—which SMEM buffers feed which MMA operands, which TMEM accumulators hold which results—are invariant. What varies is the scalar preprocessing applied to the matmul inputs and the postprocessing applied to their outputs.

### 1.4 The Covariance Gap

The TTR framework identifies a systematic weakness: every practical variant approximates $\Phi_t^\top \Gamma_t \Phi_t \approx I$. This means the memory matrix $M_t$ is biased by the key covariance structure—frequently occurring key patterns dominate the state, even if they carry redundant information, while rare patterns are underrepresented.

The delta rule partially addresses this by erasing old associations before overwriting, but it's a rank-1 correction per timestep—it helps with *interference* (conflicting values at the same key) but doesn't address the *distributional* bias from correlated keys.

Even a lightweight correction—a diagonal preconditioner $\text{diag}(\Phi_t^\top \Gamma_t \Phi_t)^{-1}$ maintained as running per-dimension variance estimates—would capture per-dimension key concentration. This costs $N$ extra registers and $\sim ND$ divisions per chunk. Whether it helps at scale is unknown: Wang et al. demonstrate the issue on synthetic MQAR tasks with orthonormal embeddings, and it's unclear whether the diagonal captures enough structure when key correlations are the dominant problem. But the framework makes testing this trivial: define a new `state_propagate` that divides INTER1_ACC element-wise before accumulation. Days of work, not months.

This motivates a broader question: why stay within the regression framework at all?

---

## 2. The Kernel as Computational Grammar

### 2.1 Separating Shape from Meaning

The critical insight for this project is that the kernel's 4-matmul structure is **more general than regression**. The hardware sees:

1. A persistent $N \times D$ matrix (the "state") distributed across pre-inter warp registers
2. Per-chunk inputs arriving via TMA: keys $(L, N)$, values $(L, D)$, queries $(L, N)$, parameters
3. An $L \times L$ pairwise interaction within the chunk (INTRA1 + masked by pre-intra → INTRA2)
4. An $N \times D$ contribution from this chunk to the state (INTER1, modified by pre-inter)
5. An $(L, D)$ output from querying the state (INTER2, combined with intra by epilogue)
6. A state update rule (pre-inter's `state_propagate`)

The regression interpretation assigns *meaning* to these operations: the state is a memory matrix, INTER1 computes an outer product of keys and values, INTER2 retrieves stored associations. But the hardware doesn't know that. It sees matrix multiplications of specific shapes, element-wise operations in specific warp groups, and a persistent register tensor that gets updated each chunk.

Any sequence model whose per-chunk computation can be expressed as:

- A pairwise $(L, L)$ interaction within the chunk
- An $(N, D)$ state update from chunk inputs
- An $(L, D)$ output combining local interaction and state query
- A state propagation rule between chunks

...runs on this hardware at near-peak utilization, regardless of the mathematical framework that derived it.

### 2.2 The Grammar

We can formalize this as a computational grammar—a set of "slots" the hardware fills per chunk:

```
CHUNK_ITERATION(chunk_c, state_S):
    // TMA loads: K, V, Q, params → SMEM

    // INTRA1: L×L pairwise interaction
    A_raw = Q × K^T                          [Tensor Core, (L,L,N)]

    // PRE_INTRA: apply structure to interaction
    A_masked = mask_apply(A_raw, params)      [CUDA cores, pre-intra warps]

    // INTRA2: local output from masked interaction
    Y_intra = A_masked × V                   [Tensor Core, (L,D,L)]

    // PRE_INTER: prepare keys for state update
    K_prepared = key_preprocess(K, params)    [CUDA cores, pre-inter warps]

    // INTER1: chunk's contribution to state
    delta_S = K_prepared^T × V               [Tensor Core, (N,D,L)]

    // STATE_PROPAGATE: update persistent state
    S_new = state_propagate(S, delta_S, params)  [CUDA cores, pre-inter warps]

    // INTER2: output from accumulated state
    Y_inter = Q × S_new                      [Tensor Core, (L,D,N)]

    // EPILOGUE: combine outputs
    Y = output_combine(Y_intra, Y_inter, params) [CUDA cores, epilogue warps]

    return Y, S_new
```

The four customization points—`mask_apply`, `key_preprocess`, `state_propagate`, `output_combine`—are the grammar's *free variables*. Everything else is fixed by the hardware: TMA loads, MMA execution, pipeline barriers, SMEM/TMEM staging, tile scheduling.

### 2.3 What the Grammar Can Express

Any model whose update rule decomposes into this pattern can use the grammar. The key constraints are:

**Shape constraints.** The state must be expressible as an $N \times D$ matrix (or a small number of such matrices, if we extend the skeleton with additional persistent register tensors). The per-chunk interactions must factor into 4 matmuls of the shapes above.

**Computational budget.** The customization points run on CUDA cores in the preprocessing/epilogue warp groups. They have a cycle budget determined by how long the MMA pipeline takes (~8–12K cycles at typical tile sizes). If a customization point exceeds this budget, it becomes the bottleneck and MMA utilization drops. For SSD-family preprocessing, this is trivially satisfied. For DeltaNet's WY construction, it's tight (see §5). For arbitrary user-defined operations, it's a constraint the user must respect.

**Data flow constraints.** The customization points receive their inputs from SMEM (TMA-loaded tensors, MMA accumulators staged to SMEM) and produce outputs to SMEM (for MMA consumption) or registers (for state). They don't directly access HBM or communicate with other warp groups except through the pipeline barriers.

### 2.4 Why This Matters

The grammar changes how we think about sequence model design. Currently, the process is:

1. Choose a mathematical objective (regression, contrastive, information-theoretic, ...)
2. Derive the optimal update rule
3. Hope it admits an efficient chunked implementation
4. Write a CUDA kernel (months of work)
5. Discover performance is bad because the update rule doesn't map well to hardware
6. Simplify the math to fit the hardware
7. Repeat

The grammar inverts this: here's what the hardware does efficiently. *Find objectives whose optimal update rules live within the grammar.* Step 4 is eliminated—if it fits the grammar, it runs at near-peak speed automatically. Step 5 is replaced by upfront analysis: does your update rule decompose into 4 matmuls with the right shapes?

This is analogous to how FlexAttention changed the softmax-attention design space. FlexAttention didn't just make existing attention faster—it made researchers willing to try novel masking patterns (document masks, sliding window, dilated, etc.) because the cost of experimentation dropped to writing a `score_mod` function. Flex Linear Attention does the same for the *recurrent state update* design space.

---

## 3. Beyond Regression: Alternative Objectives

The TTR framework is a powerful unifying lens, but it's also a constraint. The regression objective has a specific inductive bias: it treats each key-value pair as an independent observation of a linear function $v = Mk$, weighted by recency. This is well-suited for associative recall but may not be the best objective for what language modeling actually needs—tracking logical state, counting, compression, relational reasoning.

This section sketches four alternative objectives whose update rules fit the computational grammar. These are not fully developed proposals; they're existence proofs that the grammar supports a broader design space than regression alone. Each would require empirical validation to determine whether it improves over existing methods. The framework's value is that such validation becomes cheap.

### 3.1 Online Contrastive State Update

**Motivation.** The regression objective minimizes reconstruction error for *all* stored associations equally (modulo decay). But in language modeling, what matters is *discriminative* retrieval: the correct next token should be more retrievable than alternatives. A contrastive objective directly optimizes for this.

**Objective.** At each step, maximize the contrastive score of the correct association:

$$\mathcal{L}_t = k_t^\top M^\top v_t - \log \sum_j \exp(k_t^\top M^\top v_j)$$

The one-step online gradient with respect to $M$ is:

$$\nabla_M \mathcal{L}_t = v_t k_t^\top - \left(\sum_j p_j v_j\right) k_t^\top = (v_t - \bar{v}_t) k_t^\top$$

where $p_j = \text{softmax}(k_t^\top M^\top v_j)$ over recent values, and $\bar{v}_t = \sum_j p_j v_j$ is the expected value under the current memory.

**Update rule:**

$$M_t = \alpha_t M_{t-1} + \eta (v_t - \bar{v}_t) k_t^\top$$

**How it maps to the grammar.** The update is structurally identical to SSD—a rank-1 additive update to $M$—but the "value" being stored is a *residual* $(v_t - \bar{v}_t)$ that depends on the current state. The state participates in computing its own update.

Concretely:
- INTER2 computes $Q \times S$ as usual, giving us access to $M^\top v_j$ for each position in the chunk
- Pre-inter or a small auxiliary computation derives $\bar{v}_t$ from INTER2's output (a softmax-weighted average within the chunk, which is an $L$-length reduction)
- The effective value $\tilde{v}_t = v_t - \bar{v}_t$ is computed in pre-inter before INTER1
- INTER1 computes $\tilde{K}^\top \times \tilde{V}$ as the state update contribution

**What changes in the kernel:** The key challenge is the feedback path—INTER2's output must feed back into INTER1's input within the same chunk. In the current pipeline, INTER2 runs after INTER1 (they're on the same MMA warp, sequenced). Supporting this requires either:
1. Using the *previous chunk's* state for the correction (one-chunk-lagged, simple, no pipeline change)
2. Adding an additional MMA phase for the feedback computation
3. Splitting the pipeline into a "state query" phase and a "state update" phase

Option (1) is the natural starting point. The approximation (using $S_{c-1}$ instead of $S_c$ for the contrastive correction) is analogous to the standard chunkwise approximation already used in all linear attention variants—within-chunk interactions use local computation, cross-chunk interactions use the lagged state.

**What we'd learn.** Whether discriminative retrieval (contrastive) outperforms reconstructive retrieval (regression) for language modeling at scale. The regression framework optimizes for faithful storage; the contrastive framework optimizes for useful retrieval. These are different objectives and there's no a priori reason to prefer one.

### 3.2 Information-Theoretic Compression

**Motivation.** The regression state stores everything weighted by recency. An information-theoretic objective would store *only what's predictive of the future*, discarding redundant or irrelevant past information.

**Objective.** The information bottleneck: minimize $I(X_{\text{past}}; S) - \beta \cdot I(S; X_{\text{future}})$. Store as little as possible (low $I(X_{\text{past}}; S)$) while remaining maximally predictive (high $I(S; X_{\text{future}})$).

For a linear Gaussian model, the gradient of mutual information w.r.t. the state gives:

$$\nabla_S I(S; x_t) \propto \Sigma_S^{-1} (x_t - \hat{x}_t) k_t^\top$$

where $\hat{x}_t = S k_t$ is the prediction and $\Sigma_S$ captures state uncertainty.

**Update rule:**

$$S_t = \alpha_t S_{t-1} + \eta \cdot \Sigma_S^{-1} (v_t - S_{t-1} k_t) k_t^\top$$

**How it maps to the grammar.** This is structurally a preconditioned delta rule. The prediction error $(v_t - S_{t-1} k_t)$ requires querying the state (like the contrastive case), and $\Sigma_S^{-1}$ is a preconditioner. Critically, the $\Sigma_S^{-1}$ term arises *naturally* from the information-theoretic objective—it's not a bolted-on correction to an approximate solution. It's the *right answer* to a different question.

If we approximate $\Sigma_S$ as diagonal (the same approximation we discussed for the covariance gap), the preconditioner costs $N$ extra registers and $ND$ divisions per chunk. The information-theoretic derivation *motivates* the diagonal approximation rather than assuming it post-hoc.

**What we'd learn.** Whether the information bottleneck principle—store only what's predictive—produces better state representations than the regression principle—store everything faithfully. This connects to a large body of information-theoretic work on representation learning (Tishby's information bottleneck, VIB, etc.) that has been explored for feedforward networks but not for recurrent state management.

### 3.3 Predictive State Representations

**Motivation.** Predictive state representations (PSRs), from Littman, Sutton, and Singh, maintain sufficient statistics for predicting future observations. Rather than storing key-value associations, the state encodes a *model* of the sequence's dynamics.

**Objective.** The state $S_t$ satisfies:

$$\mathbb{E}[f(x_{t+1:t+k}) \mid x_{1:t}] = S_t \phi(x_t)$$

for a set of test functions $f$ (which can be learned). The update rule derived from this is a temporal difference (TD) update:

$$S_t = S_{t-1} + \eta (f(x_t) - S_{t-1} \phi(x_{t-1})) \phi(x_{t-1})^\top$$

**How it maps to the grammar.** The TD update has the same rank-1 shape as SSD, but the "value" is the prediction error $f(x_t) - S_{t-1} \phi(x_{t-1})$, which depends on the state (via the prediction $S_{t-1} \phi(x_{t-1})$). This is the same feedback pattern as the contrastive case: the state correction depends on querying the state.

Using the previous chunk's state (option 1 from §3.1), the within-chunk computation becomes:

- Pre-inter computes predictions $\hat{v}_j = S_{c-1} k_{j-1}$ for each position (using the chunk-lagged state)
- The effective values become residuals $\tilde{v}_j = f(x_j) - \hat{v}_j$
- INTER1 computes the state update from these residuals

The test function $f$ is learned jointly with the model. In the simplest case, $f(x_t) = v_t$ (the next token's value projection), and the objective reduces to predicting the next value from the current key using the state—which is close to (but subtly different from) the regression objective. The difference is that TD learning updates based on *sequential prediction error* rather than *independent reconstruction error*.

**What we'd learn.** Whether TD-style sequential updates—which exploit the temporal structure of the data—outperform regression-style independent updates. PSRs have strong theoretical properties (they can represent any observable operator model) but have not been explored in the context of modern sequence models.

### 3.4 Spectral Filtering

**Motivation.** Instead of storing key-value associations, maintain a low-rank approximation to the sequence's temporal correlation structure—its Hankel matrix. The Hankel matrix $H_{ij} = x_{t+i}^\top x_{t+j}$ captures all pairwise temporal correlations, and its singular vectors reveal the "true" latent state.

**Approach.** Online incremental SVD of the Hankel-like matrix:

$$U_t, \Sigma_t, V_t = \text{rank-}N\text{-SVD}(H_{t-1} + x_t x_t^\top)$$

The rank-$N$ truncated SVD update decomposes into: (a) project the new observation onto current singular vectors, (b) update singular values, (c) rotate singular vectors. Steps (a) and (c) are matmuls. Step (b) is element-wise.

**How it maps to the grammar.** The state represents $(U, \Sigma, V)$ packed into an $N \times D$ matrix (or multiple register tensors). The state propagation involves rotation matrices applied to the state—$S_{\text{new}} = R \cdot S \cdot R'^\top + \Delta S$—which is still matmul-shaped but with the state on both sides. This requires the `state_propagate` contract to support matrix-matrix operations on the state, not just element-wise scaling + additive updates.

This is the most architecturally demanding alternative. The rotation matmuls are $O(N^2 D)$ per chunk, significantly heavier than the SSD family's $O(ND)$. At $N=128, D=64$, this is ~1M FMAs in state propagation alone—comparable to DeltaNet's WY cost. The analysis from §5 (DeltaNet cycle budget) applies similarly here.

**What we'd learn.** Whether maintaining a spectral summary of the sequence—which captures correlational structure rather than pointwise associations—is a useful inductive bias for language modeling. Spectral methods have strong connections to HMMs and observable operator models, and they've been successful in control and system identification, but haven't been applied as recurrent state update rules in neural sequence models.

### 3.5 What These Alternatives Have in Common

All four alternatives:

1. **Maintain an $N \times D$ state matrix.** Same shape as SSD, same register storage, same INTER2 query mechanism.

2. **Update the state with a function of the current chunk's inputs.** Maps to INTER1 + pre-inter's `state_propagate`. The *function* differs (additive, contrastive residual, TD error, rotation), but the *shape* is the same.

3. **Have an intra-chunk pairwise interaction.** Maps to INTRA1 + pre-intra's `mask_apply`. The mask structure may differ from the causal-times-decay pattern, but the $(L, L)$ shape is invariant.

4. **Produce output as a combination of local and global terms.** Maps to INTRA2 + INTER2 + epilogue's `output_combine`.

The differences live entirely within the customization points. The TMA loads, MMA tile shapes, pipeline barriers, TMEM/SMEM allocation, and tile scheduling are the same regardless of which objective derived the update rule.

### 3.6 Extending the Grammar for State-Dependent Updates

The contrastive, information-theoretic, and predictive-state objectives all share a feature not present in existing linear attention: the state update depends on *querying* the current state. In the grammar's terms, INTER1's effective input depends on INTER2's output (or something like it).

There are three approaches, in order of increasing invasiveness:

**Approach A: Chunk-lagged feedback (zero pipeline changes).** Use $S_{c-1}$ (the state entering this chunk, already in pre-inter registers) to compute the state-dependent correction. The intra-chunk computation doesn't see the state update from the current chunk. This is mathematically equivalent to the existing chunkwise approximation—all current linear attention variants use the lagged state for inter-chunk output, and the error is $O(C)$ in the chunk size.

For the contrastive objective: pre-inter computes $\bar{v}_j = \text{softmax}(S_{c-1} k_j)^\top V_{\text{chunk}}$ using data already in SMEM and the state in registers. This is an $N$-dimensional matmul per position, totaling $O(LND)$ FMAs—comparable to a single MMA phase. It fits in pre-inter's cycle budget for SSD-family tile sizes.

**Approach B: State query buffer in SMEM.** Write $S_{c-1}$ to SMEM once per chunk (it's already in pre-inter registers). Pre-intra warps can then access the state for mask_apply computations that depend on it. This adds SMEM traffic but no pipeline restructuring.

**Approach C: Additional MMA phase.** Add a 5th MMA phase that computes the state-dependent correction on the Tensor Core. This requires extending the pipeline (5 phases instead of 4) and the TMEM offset planning (5 accumulator buffers). It's the most invasive option but handles heavy state-dependent computations that exceed CUDA core budgets.

For Phase 4 of the project, we start with Approach A and evaluate whether the chunk-lagged approximation is sufficient. If not, Approach B is the next step. Approach C is future work that requires skeleton modifications.

---

## 4. Kernel Architecture

### 4.1 The Skeleton

The Mamba-2 CuTe DSL kernel uses 16 warps in 7 specialized groups:

| Group | Warps | Registers/Thread | Work |
|-------|-------|-----------------|------|
| TMA Load (K/Q) | 1 | 24 | Async load $K$, $Q$ from HBM to SMEM |
| TMA Load (V/Δ) | 1 | 24 | Async load $V$, parameters to SMEM |
| MMA Intra | 1 | 24 | INTRA1 + INTRA2 on Tensor Core |
| MMA Inter | 1 | 24 | INTER1 + INTER2 on Tensor Core |
| Pre-Inter | 4 | 168 | `key_preprocess` + `state_propagate` (CUDA cores) |
| Pre-Intra | 4 | 208 | `mask_apply` (CUDA cores) |
| Epilogue | 4 | 112 | `output_combine` + TMA store (CUDA cores) |

The state $S \in \mathbb{R}^{N \times D}$ is distributed across the 4 pre-inter warps' register files (128 threads), persisting for the entire sequence. At $N=128, D=64$ in fp32: 64 registers per thread consumed by state alone, out of the 168 budget.

**Pipeline mechanics.** The chunk loop pipelines TMA loads with MMA execution with preprocessing:
- TMA warps load chunk $c+2$'s data while MMA processes chunk $c+1$'s matmuls while pre-inter/pre-intra process chunk $c$'s intermediates
- Pipeline barriers synchronize: pre-inter must finish before INTER1 can consume the preprocessed keys; pre-intra must finish before INTRA2 can consume the masked $Q$
- Stage counts (`input_stages=2, output_stages=2, internal_stages=1, intra1_acc_stages=2`) determine how many chunks can be in-flight simultaneously

**SMEM layout details.** Each SMEM buffer is formatted for its consumer MMA:
- `x_smem_layout`: $V$ as B-operand of INTRA2 and INTER1 (shared, both need $(L, D)$ layout)
- `b_smem_layout`: $K$ as B-operand of INTRA1 ($(L, N)$ swizzled)
- `bt_internal_smem_layout`: preprocessed $\tilde{K}$ as A-operand of INTER1 ($(N, L)$ swizzled)
- `c_smem_layout`: $Q$ as A-operand of INTRA1 and INTER2 (shared, $(L, N)$ layout)

The internal buffer `bt_internal` is where pre-inter writes its output—the preprocessed keys ready for INTER1. Its layout is determined by `make_smem_layout_a(tiled_mma_inter1, ...)`, which computes the swizzle pattern needed for INTER1's A-operand. This is the critical coupling point: the pre-inter code must write data in *exactly this layout*, regardless of what the preprocessing does mathematically.

**TMEM layout details.** The `_plan_tmem_offsets` function computes non-overlapping TMEM column ranges for:
- INTRA1 accumulator: $(L, L)$ in fp32
- INTRA2 Q-operand (the masked result staged from TMEM): $(L, L)$ in io_dtype
- INTRA2 accumulator: $(L, D)$ in fp32
- INTER1 accumulator: $(N, D)$ in fp32
- INTER2 accumulator: $(L, D)$ in fp32

The TMEM planning depends on MMA tile shapes. If a variant changes tile shapes (e.g., MIMO changing INTER1 from $(N, D, L)$ to $(N, D, L \cdot r)$), the TMEM offsets must be recomputed.

### 4.2 The Four Customization Points

Each has a well-defined contract specifying inputs, outputs, and where data lives:

**`key_preprocess`** (pre-inter warps, 4 warps × 32 threads)
- *Input:* Raw $K$ in `b_smem` (B-operand layout) + decay parameters in SMEM
- *Output:* Preprocessed $\tilde{K}$ in `bt_internal_smem` (A-operand layout for INTER1)
- *Contract:* Output must conform to `bt_internal_smem_layout`. The transformation from B-operand layout to A-operand layout (effectively a transpose + swizzle) is part of this function's responsibility.
- *SSD cost:* ~100 FMAs (element-wise exp + multiply + layout transform)
- *DeltaNet cost:* Offloaded to preprocessing kernel (see §5)

**`mask_apply`** (pre-intra warps, 4 warps × 32 threads)
- *Input:* INTRA1 accumulator in TMEM ($(L, L)$ in fp32) + decay parameters in SMEM
- *Output:* Masked $\tilde{Q}$ written to TMEM in INTRA2's A-operand layout
- *Contract:* The output must be a valid A-operand for INTRA2_MMA. The mask encodes causal structure (zero above diagonal) and variant-specific decay.
- *SSD:* $\tilde{Q}[i,j] = A_{\text{raw}}[i,j] \cdot \delta_j \cdot \exp(\text{cumsum}_i - \text{cumsum}_j)$ for $i \geq j$
- *GLA:* $\tilde{Q}[i,j] = A_{\text{raw}}[i,j] \cdot \prod_{m=j+1}^{i} \gamma_m$ for $i \geq j$
- *Mamba-3:* $\tilde{Q}[i,j] = (L_{\text{decay}} \times L_{\text{conv}})[i,j] \cdot A_{\text{raw}}[i,j]$ (product of decay and bidiagonal)

**`state_propagate`** (pre-inter warps, 4 warps × 32 threads)
- *Input:* INTER1 accumulator in TMEM ($(N, D)$ in fp32) + persistent state $S$ in registers + decay parameters
- *Output:* Updated state $S$ in registers + state written to `s_smem` for INTER2_MMA's B-operand
- *Contract:* State must be written to SMEM in INTER2's B-operand layout. The register-to-SMEM transfer is part of this function.
- *SSD:* $S \leftarrow e^{\Delta_{\text{last}}} \cdot S + \text{INTER1\_ACC}$ (element-wise FMA, ~$ND$ FMAs)
- *DeltaNet:* $S \leftarrow (I - WY^\top)(\gamma S) + \text{correction}$ (two auxiliary matmuls, $W, Y$ loaded from SMEM)
- *Novel objectives:* State-dependent updates using Approach A (compute correction from $S$ and chunk data in registers/SMEM)

**`output_combine`** (epilogue warps, 4 warps × 32 threads)
- *Input:* INTRA2 accumulator in TMEM + INTER2 accumulator in TMEM + parameters in SMEM
- *Output:* Final $Y$ in `yt_smem` for TMA store
- *Contract:* Output must be in the TMA store layout.
- *SSD:* $Y = \exp(\text{cumsum}) \cdot Y_{\text{inter}} + Y_{\text{intra}} + D \cdot X$ (element-wise + optional $D$ fusion)

### 4.3 Known Skeleton Coupling Points

These are places where the "invariant skeleton" may encode SSD-specific assumptions:

1. **`bt_internal_smem_layout` shape.** Derived from `make_smem_layout_a(tiled_mma_inter1, tile_shape_mnk_inter1, ...)`. The tile shape $(N, D, L)$ is correct for all SISO variants. MIMO changes this to $(N, D, L \cdot r)$ or requires $r$ separate INTER1 calls.

2. **Pipeline stage counts.** Hardcoded `(2, 2, 1, 2)`. If pre-inter becomes a bottleneck, more `internal_stages` could allow INTER1's result to be buffered longer while pre-inter catches up. But more stages consume more SMEM.

3. **TMEM offset planning.** Assumes exactly 5 TMEM buffers (4 accumulators + 1 Q staging). Novel objectives requiring additional TMEM storage (e.g., a state query result) would need the planner extended.

4. **TMA descriptor set.** The kernel creates TMA descriptors for $K$, $Q$, $V$, $\Delta$, cumsum($\Delta$), $D$, $Y$, and the final state. Variants that need additional inputs (e.g., $\beta$ for DeltaNet, $W/Y$ from the preprocessing kernel) need additional TMA descriptors.

5. **Register budget allocation.** The `setmaxregister` directives assign 168 registers to pre-inter, 208 to pre-intra, 112 to epilogue. If a variant needs more pre-inter registers (e.g., for state-dependent corrections), it must borrow from another group.

Phase 1 will determine which of these are truly invariant (can be left as-is for all variants) and which need to become variant-configurable parameters.

---

## 5. The DeltaNet Strategy: Preprocessing Kernel

### 5.1 Why Separate WY Construction

The WY representation $I - WY^\top = \prod_{j=1}^{C} (I - \beta_j k_j k_j^\top)$ requires sequential construction:

$$w_t = \beta_t k_t - W_{:,:t} (Y_{:,:t}^\top (\beta_t k_t)), \qquad Y_{:,t} = k_t$$

At $C=64, N=128$: ~524K FMAs, inherently sequential in $t$, and requiring storage for the growing $W, Y$ matrices.

Trying to do this inside the pre-inter warp group faces three compounding problems:

1. **Register pressure.** Pre-inter has 168 registers/thread. The state $S$ consumes ~64. The WY construction needs to hold partial $W$ columns, partial $Y$ columns, the mat-vec accumulator, and loop variables. At $C=64, N=128$ distributed across 128 threads: each thread handles 1 row of $W$ and needs to accumulate over $C$ columns. This likely requires SMEM spilling, which adds latency.

2. **Sequential dependency.** Column $t$ of $W$ depends on all previous columns via the mat-vec $W_{:,:t}(Y_{:,:t}^\top (\beta_t k_t))$. This serializes across the chunk dimension, limiting ILP. The 128 threads parallelize across $N$ (1 element each), but the $C$ iterations are sequential.

3. **Cycle budget.** Realistic estimate: 15–20K cycles for WY construction + state propagation. The MMA pipeline runs in 8–12K cycles. Pre-inter would be the bottleneck by 1.5–2×, meaning the Tensor Cores sit idle waiting for state propagation.

A separate preprocessing kernel avoids all three problems:

### 5.2 The Preprocessing Kernel

A small, simple kernel that runs before the main fused kernel:

```
preprocess_wy(K, beta) -> W, Y:
    for each chunk c:
        for each head h:
            // Build W, Y ∈ R^{N×C} via the sequential loop
            // Full register file available (no state, no MMA pipeline to share with)
            // Write W, Y to HBM
```

This kernel is:
- **Embarrassingly parallel** across chunks and heads (no inter-chunk dependency)
- **Register-rich** (no competing warp groups, full 255 registers available)
- **Bandwidth-light** ($W, Y$ are small: $2 \times N \times C \times \text{sizeof(bf16)} = 2 \times 128 \times 64 \times 2 = 32\text{KB}$ per chunk per head)
- **Latency-tolerant** (can overlap with the main kernel's TMA loads for other tensors)

### 5.3 What Changes for the Main Kernel

With $W, Y$ precomputed and available in HBM, the main fused kernel treats them as additional TMA-loaded inputs:

- **TMA loads:** Add two more async loads per chunk: $W$ and $Y$ into SMEM (alongside $K, V, Q, \text{params}$). This requires two additional TMA descriptors and slightly more SMEM for the extra buffers. At 32KB per $(W, Y)$ pair: well within the 228KB SMEM budget.

- **`key_preprocess`:** For DeltaNet, this becomes: load $K$ from SMEM, apply $\beta$-scaling (element-wise), write to `bt_internal_smem`. The WY construction is no longer here. Cost: same as SSD (~100 FMAs).

- **`state_propagate`:** Load $W, Y$ from SMEM. Compute $Y^\top S$ ($(C, N) \times (N, D) \to (C, D)$, ~524K FMAs). Compute $W \times \text{result}$ ($(N, C) \times (C, D) \to (N, D)$, ~524K FMAs). Apply: $S \leftarrow \gamma \cdot S - W(Y^\top(\gamma S)) + \text{INTER1\_ACC}$. Total: ~1.05M FMAs across 128 threads = ~8.2K FMAs/thread ≈ **8–10K cycles**.

This is in the right ballpark to be hidden behind the MMA pipeline (~8–12K cycles). The sequential dependency is gone—the two matmuls are fully parallel. Register pressure is manageable: we need temporary storage for the $C \times D$ intermediate $Y^\top S$, but this can be tiled (compute $Y^\top S$ in tiles, apply $W \times \text{tile}$ immediately, accumulate).

### 5.4 Net Assessment

| Aspect | Fully-fused WY | Preprocessing kernel |
|--------|---------------|---------------------|
| WY construction | Inside pre-inter (15–20K cycles, sequential) | Separate kernel (~1K cycles, parallel) |
| State propagation | Same pre-inter group (8–10K cycles) | Same pre-inter group (8–10K cycles) |
| Extra HBM traffic | None | 32KB/chunk/head for $W, Y$ read+write |
| Pre-inter bottleneck | Likely (1.5–2× MMA) | Probably not (≈MMA budget) |
| Kernel launch overhead | One kernel | Two kernels (negligible for long sequences) |
| Skeleton complexity | Heavy (WY logic in pre-inter) | Light (just load W,Y, do matmuls) |

The preprocessing kernel strategy trades ~32KB of HBM traffic per chunk for dramatically simpler skeleton integration. At HBM bandwidth of ~8TB/s on B100, 32KB costs ~4ns—negligible compared to the ~10μs per-chunk computation time.

This also means the main kernel skeleton doesn't need to know about Householder reflectors *at all*. From the skeleton's perspective, DeltaNet's `state_propagate` is: "load two matrices from SMEM, do two matmuls with the state, accumulate." The mathematical meaning (WY representation of a Householder product) is hidden behind the preprocessing kernel.

---

## 6. Implementation Plan

### Phase 1: Skeleton Extraction + SSD Family (Weeks 1–6)

**Goal:** Factor the Mamba-2 kernel into skeleton + customization points. Validate with SSD (bit-exact) and GLA (new variant, same transition family).

| Week | Task | Deliverable |
|------|------|------------|
| 1–2 | Annotate every line of the kernel as SKELETON or SSD-SPECIFIC. Document every assumption about shapes, layouts, stages, register budgets. | Assumption catalog + risk assessment |
| 3–4 | Extract the 4 customization point functions. Parameterize the skeleton (stage counts, TMA descriptor sets, register budgets as config). Verify SSD instantiation is bit-identical. | Parameterized skeleton + SSD: bit-identical, ≤2% perf regression |
| 5–6 | Implement GLA customization point functions. GLA differs from SSD only in decay structure ($\gamma_t$ gates vs $e^{\Delta A}$ exponentials). | GLA: ≤10⁻³ relative error vs PyTorch ref, performance within 15% of SSD |

**What we learn:** Which of the 5 coupling points (§4.3) actually require parameterization. If GLA—the easiest non-SSD variant—requires unexpected changes, the generalization hypothesis is weaker than expected.

**Go/no-go gate (week 6).** If skeleton extraction requires touching >30% of the kernel's non-customization-point code, reassess. If GLA slots in cleanly (<200 lines of new code, no skeleton changes), proceed with confidence.

### Phase 2: Delta Rule via Preprocessing (Weeks 7–12)

**Goal:** Implement Gated DeltaNet using the preprocessing kernel strategy (§5). Validate that the main kernel's skeleton accommodates the additional TMA loads and heavier `state_propagate`.

| Week | Task | Deliverable |
|------|------|------------|
| 7–8 | Implement WY preprocessing kernel. Add TMA descriptors for $W, Y$ to the main kernel skeleton. | Preprocessing kernel: correct WY output. Main kernel: compiles with extended TMA. |
| 9–10 | Implement GDN `state_propagate` (two matmuls from SMEM-loaded $W, Y$). Implement GDN `mask_apply` and `output_combine`. | GDN: correct output vs FLA reference |
| 11–12 | Profile. Measure pre-inter latency vs MMA pipeline. Benchmark against FLA 3-kernel Triton. Try warp rebalancing and chunk size sweeps if pre-inter bottlenecks. | Performance characterization + comparison |

**What we learn:** Whether the preprocessing kernel strategy makes DeltaNet competitive. Whether `state_propagate` with two SMEM-loaded matmuls fits in the MMA pipeline's cycle budget. What the optimal chunk size is for delta-rule variants (the $O(C^2 N)$ WY cost in preprocessing favors smaller $C$, but the $O(CND)$ state propagation and chunk overhead favor larger $C$).

**Decision gate (week 12).** If fused GDN ≥1.0× vs FLA: proceed. If <1.0× and warp rebalancing helps: proceed with modified warp allocation as part of variant config. If <1.0× and nothing helps: the skeleton delivers value for SSD-family only; DeltaNet uses the 2-kernel (preprocessing + fused) approach without claiming it's faster than FLA, just architecturally cleaner.

### Phase 3: Mamba-3 Features (Weeks 13–18)

**Goal:** Add trapezoidal discretization, data-dependent RoPE, and MIMO to complete the SSD-family coverage with Mamba-3.

| Week | Task | Deliverable |
|------|------|------------|
| 13–14 | Trapezoidal mask: $L = L_{\text{decay}} \times L_{\text{conv}}$ with bidiagonal $L_{\text{conv}}$ in `mask_apply`. This tests whether pre-intra can handle non-trivial mask structures beyond simple exponential decay. | Trapezoidal SSD: correct output vs Mamba-3 reference |
| 15–16 | Data-dependent RoPE: cumulative rotation on $K, Q$ before they enter the MMAs. Decide placement: in TMA warps (rotate data as it arrives in SMEM) vs in pre-inter (rotate after SMEM load) vs in a preprocessing kernel (rotate before main kernel). | RoPE + trapezoidal: correct, compare rotation placement options |
| 17–18 | MIMO: rank-$r$ updates where $K \in \mathbb{R}^{N \times r}$. This changes INTER1's effective tile shape and may require SMEM layout recomputation. Implement as compile-time template parameter on rank $r$. | Full Mamba-3 (SISO + MIMO): correct, ≥90% of hand-written kernel |

**What we learn:** MIMO is the critical test of skeleton flexibility. If the tile shape change cascades through SMEM layouts, TMEM planning, and MMA configuration, then "parameterized skeleton" means "substantial code generation for each tile configuration." If the changes are localized to the layout computation functions (as we hope), the skeleton is genuinely general.

### Phase 4: Novel Objectives (Weeks 19–24)

**Goal:** Implement 2–3 objectives from §3 and evaluate empirically. This is the research phase—the engineering framework from Phases 1–3 enables it, but the outcomes are uncertain.

| Week | Task | Deliverable |
|------|------|------------|
| 19–20 | Diagonal preconditioner (§1.4): simplest novel variant. Add $N$ running variance registers to pre-inter, divide INTER1_ACC element-wise. Evaluate on MQAR + language modeling at 350M–1.3B scale. | Preconditioned SSD kernel + evaluation results |
| 21–22 | Contrastive state update (§3.1) with chunk-lagged feedback (Approach A). Modify pre-inter to compute state-dependent value residuals before INTER1. | Contrastive variant kernel + evaluation vs regression baseline |
| 23–24 | TD-style predictive state update (§3.3) or information-theoretic compression (§3.2), depending on which shows more promise from the contrastive results. | Second novel variant + comparative evaluation |

**What we learn:** Whether objectives beyond regression improve sequence modeling at scale. Even negative results (novel objectives don't help) are valuable: they would suggest that the regression framework's dominance is well-justified, not just path-dependent. Positive results would open a new direction for the field.

**Scope management.** Phase 4 is explicitly exploratory. If Phases 1–3 take longer than planned, Phase 4 contracts to just the diagonal preconditioner (the simplest, most directly motivated extension). If Phases 1–3 finish early, Phase 4 expands to include more objectives.

---

## 7. The API

### 7.1 What We Build

A Python-level specification that selects from a library of hand-written code fragments:

```python
from flex_linear_attn import FlexLinearAttention, decay, feature_map, transition

# --- Known variants (Phases 1-3) ---

mamba2 = FlexLinearAttention(
    decay=decay.exponential(data_dependent=True),
    features=feature_map.identity(),
    transition=transition.one_step_gd(),
)

gla = FlexLinearAttention(
    decay=decay.gated(per_head=True),
    features=feature_map.identity(),
    transition=transition.one_step_gd(),
)

gated_delta_net = FlexLinearAttention(
    decay=decay.gated(per_head=True),
    features=feature_map.identity(),
    transition=transition.delta_rule(),  # triggers preprocessing kernel
)

mamba3 = FlexLinearAttention(
    decay=decay.exponential(data_dependent=True, discretization="trapezoidal"),
    features=feature_map.rope(data_dependent=True),
    transition=transition.one_step_gd(mimo_rank=4),
)

# --- Novel objectives (Phase 4) ---

preconditioned_mamba3 = FlexLinearAttention(
    decay=decay.exponential(data_dependent=True, discretization="trapezoidal"),
    features=feature_map.rope(data_dependent=True),
    transition=transition.preconditioned_gd(preconditioner="diagonal", mimo_rank=4),
)

contrastive = FlexLinearAttention(
    decay=decay.gated(per_head=True),
    features=feature_map.identity(),
    transition=transition.contrastive(feedback="chunk_lagged"),
)

predictive_state = FlexLinearAttention(
    decay=decay.gated(per_head=True),
    features=feature_map.identity(),
    transition=transition.temporal_difference(),
)

# --- Novel combinations ---

delta_rope = FlexLinearAttention(
    decay=decay.gated(per_head=True),
    features=feature_map.rope(data_dependent=True),
    transition=transition.delta_rule(),
)
```

Each component maps to specific customization point implementations:

| Component | Determines | Code Path |
|-----------|-----------|-----------|
| `decay.*` | `mask_apply` + `output_combine` | Pre-intra mask structure, epilogue scaling |
| `features.*` | Input preprocessing | Key/query rotation before TMA or in pre-inter |
| `transition.*` | `key_preprocess` + `state_propagate` | Pre-inter logic, optional preprocessing kernel |

### 7.2 Extension Points for Future Work

The factored structure creates clean extension points:

**Custom mask tracing (future Tier 2).** The `mask_apply` contract is: receive $(L, L)$ accumulator + params → produce masked $(L, L)$ in TMEM. A `score_mod`-style callback that generates this code needs only to satisfy this contract. The compiler problem is well-scoped: trace a scalar function, verify causal factorization, extract inter-chunk decay, generate the element-wise loop body.

**Custom transitions (future Tier 3).** The `key_preprocess` + `state_propagate` contracts define what the code must produce. A class-based DSL declaring extra persistent state and defining update methods could compile to pre-inter code. Phase 2's experience with DeltaNet's register pressure and Phase 4's experience with state-dependent updates inform the design constraints.

**Backward pass (future).** The forward kernel's data flow graph (which tensors feed which warp groups, which intermediates are saved vs recomputed) is the blueprint. The backward requires ~6 MMA phases, reverse-direction state gradient propagation, and variant-specific adjoint computations. This likely needs a second skeleton.

**SM90 (Hopper) support (future).** The customization point contracts are architecture-independent. An SM90 skeleton using WGMMA and SMEM-based accumulators (no TMEM) could share the variant-specific logic while providing a different kernel backend.

**Auto-tuning (future).** Chunk size, warp allocation, pipeline staging, and register budget allocation as tunable parameters per variant. Phase 2's profiling data establishes whether this is necessary or whether per-variant manual tuning suffices.

### 7.3 What We're Not Building

- **Symbolic compilation of arbitrary Python functions** into kernel code. Each supported component is hand-written.
- **Backward pass kernels.** Forward only. Training uses `torch.autograd.Function` with fused forward + non-fused backward.
- **Auto-tuning infrastructure.** Manual tuning per variant, informed by profiling.
- **SM90 support.** Blackwell (SM100) only.
- **Runtime dispatch.** Variant selection is compile-time (CuTe DSL JIT). No overhead from generality.

---

## 8. Performance Targets

| Variant | Target | Baseline | Confidence |
|---------|--------|----------|-----------|
| SSD (regenerated) | ≥98% throughput | Original Mamba-2 CuTe DSL | High — same code, factored |
| GLA | Establish fused baseline | FLA 3-kernel Triton, same GPU | High — trivial pre-inter, full fusion benefit |
| Gated DeltaNet | ≥1.0× vs FLA (parity) | FLA 3-kernel Triton, same GPU | Medium — preprocessing kernel simplifies but adds HBM traffic |
| Gated DeltaNet | ≥1.3× vs FLA (stretch) | Same | Low-medium — requires state_propagate hidden behind MMA |
| Mamba-3 | ≥90% of hand-written | Published Mamba-3 CuTe DSL | Medium — MIMO tile changes add overhead |
| Novel variants | Correct + within 15% of SSD | SSD kernel at same tile sizes | Medium — state-dependent corrections add pre-inter work |

All comparisons at matched dimensions ($L=128, D=64, N=128$), same hardware (B100/B200), same precision (bf16 I/O, fp32 accumulators). Benchmarked at batch sizes and sequence lengths representative of both training prefill and inference decode.

---

## 9. Risks

| Risk | Likelihood | Impact | Phase | Mitigation |
|------|-----------|--------|-------|------------|
| SMEM layouts encode SSD-specific shapes | Medium | Medium | 1 | Parameterize layout computation by variant config |
| GLA requires unexpected skeleton changes | Low | High | 1 | Reassess generalization hypothesis; may need per-variant skeletons |
| Preprocessing kernel for WY adds too much latency | Low | Medium | 2 | WY is small (32KB/chunk); can overlap with TMA loads |
| State propagation matmuls bottleneck pre-inter | Medium | Medium | 2 | Warp rebalancing (6–8 pre-inter warps), chunk size reduction |
| MIMO tile shape change cascades through skeleton | High | Medium | 3 | Accept per-tile-shape SMEM configs; template on rank $r$ |
| Chunk-lagged feedback insufficient for contrastive/TD objectives | Medium | Low | 4 | Approach B (state in SMEM) or accept approximation |
| Novel objectives don't improve over regression | Medium | Low | 4 | Framework value doesn't depend on any single novel objective |
| Pipeline staging needs per-variant tuning | Medium | Low | 1–2 | Add staging to variant config, sweep empirically |

---

## 10. Success Criteria

### Engineering Success

1. **Skeleton generalizes.** ≥4 variants (SSD, GLA, GDN, Mamba-3) instantiated from the same parameterized skeleton with correct output.
2. **No generality tax.** SSD and GLA from skeleton achieve ≥95% throughput of hand-written kernels.
3. **DeltaNet characterized.** Clear, profiled understanding of preprocessing kernel overhead and state propagation bottleneck, regardless of whether the fused kernel beats FLA.
4. **Clean interfaces.** Adding a new SSD-family variant requires <200 lines of customization point code and zero skeleton changes.
5. **Composability works.** Novel combinations (e.g., GLA + delta rule, Mamba-3 + preconditioner) instantiable by composing existing components.

### Research Success

6. **Grammar validated.** At least one non-regression objective (contrastive, information-theoretic, or predictive-state) instantiated and benchmarked at ≥350M parameter scale.
7. **Design space expanded.** The set of sequence models that researchers can evaluate at hardware-efficient speeds is meaningfully larger than before (not just the 5–6 known linear attention variants, but arbitrary points in the TTR + beyond-TTR design space).

### Minimum Viable Outcome

If only Phase 1 succeeds: a well-documented, cleanly factored Mamba-2 kernel with GLA support and a clear catalog of what generalizes and what doesn't. This has value even without Phases 2–4.

---

## 11. The Research Program

This section describes the longer-term vision that the framework enables. None of this is in the current project scope, but it's the reason the project matters beyond "make existing kernels slightly more reusable."

### 11.1 Hardware-Aware Architecture Search

Today, neural architecture search for sequence models is bottlenecked by implementation cost. You can search over Transformer hyperparameters (number of heads, MLP ratio, etc.) because the Transformer kernel is fixed. You can't search over *recurrence structure* because each structure needs its own kernel.

The Flex Linear Attention grammar changes this. Any recurrence that fits the grammar (4 matmuls, $N \times D$ state, customization point functions) can be evaluated at near-peak hardware speed. This enables architecture search over:

- Decay structures (exponential, gated, polynomial, learned-per-layer)
- Feature maps (identity, RoPE, random features, learned kernel)
- State transitions (GD, delta rule, preconditioned GD, contrastive, TD, ...)
- Combinations of the above

The search space is discrete (which code fragment at each customization point) and the evaluation cost is bounded (each candidate runs at within 15% of the best hand-written kernel).

### 11.2 Connecting to Control Theory

The predictive state representation and information-theoretic objectives connect recurrent sequence models to a rich literature in control theory and system identification. Linear dynamical systems, Kalman filtering, subspace identification methods—all involve maintaining state matrices updated by structured rules that fit the computational grammar.

The specific connection: a linear attention variant with a contrastive or TD update rule is performing online system identification—learning a model of the data-generating process from streaming observations. The regression-based variants are performing online function approximation—memorizing input-output pairs. These are fundamentally different tasks, and the control theory literature suggests that system identification often requires less state (because it captures dynamics, not individual observations).

If this connection bears out empirically—if system-identification-inspired update rules achieve comparable perplexity with smaller state dimensions—it would have practical implications for inference efficiency (smaller $N$ means less register pressure, higher occupancy, faster decode).

### 11.3 Theoretical Characterization of the Grammar

A more theoretical direction: what is the class of sequence-to-sequence functions expressible by models that fit the computational grammar? This connects to circuit complexity and automata theory (which functions can finite-state machines compute?), but with the specific constraints imposed by the grammar's structure—finite state dimension $N$, linear state queries (INTER2 is a matmul, not a nonlinear function of $S$), and chunk-local nonlinearity (the mask_apply and state_propagate functions are the only nonlinear operations).

Understanding this class would tell us what the grammar *can't* do—which sequence tasks fundamentally require attention's quadratic computation and which can be solved within the grammar's linear-time constraints.

### 11.4 The FlexAttention Analogy, Seriously

FlexAttention's impact wasn't just performance—it was *velocity*. Before FlexAttention, trying a new attention mask pattern meant writing a custom kernel or accepting a slow PyTorch implementation. After FlexAttention, it meant writing a `score_mod` function. The number of attention variants explored in the literature increased dramatically.

Flex Linear Attention targets the same velocity improvement for recurrent models. The `state_propagate` function is the `score_mod` of linear attention—the small piece of code that defines the model's behavior, embedded in a high-performance skeleton that handles everything else. The difference is that `state_propagate` is more complex than `score_mod` (it involves small matmuls and reductions, not just scalar operations on individual attention scores), which is why we're building the library of hand-written implementations first rather than jumping to a compiler.

But the trajectory is the same: hand-written fragments today, traced/compiled user functions tomorrow, automated architecture search the day after.


# NEW: Flex Linear Attention

**A Composable Framework for Hardware-Efficient Fused Linear Attention Kernels**

Author: Reuben · GPU Kernel Engineering · February 2026 · Draft v2

---

## Executive Summary

Linear attention variants occupy a well-understood mathematical design space formalized by Wang et al.'s *test-time regression* (TTR) framework. Every variant—Mamba-2, GLA, RetNet, DeltaNet, Mamba-3—makes three choices: regression weights (forgetting structure), feature map (key/query transformation), and state transition (how the recurrent memory updates). Despite this unity, each variant requires its own hand-written CUDA kernel.

This project tests a specific architectural hypothesis: that the fused warp-specialized chunkwise linear attention kernel—as exemplified by the Mamba-2 CuTe DSL implementation—has sufficient structural regularity that a single parameterized kernel can serve the entire SSD family of variants (Mamba-2, GLA, RetNet, RWKV-6, Mamba-3), and that the same skeleton can be extended to the delta rule family (DeltaNet, Gated DeltaNet) with heavier but bounded modifications to one warp group.

The project is scoped in three phases: (1) validate the hypothesis by extracting and instantiating the skeleton for SSD-family variants, (2) extend to the delta rule family, and (3) add MIMO and data-dependent feature maps. Backward pass, symbolic compilation of user-defined masks, and auto-tuning are explicitly deferred but the architecture is designed to accommodate them.

---

## 1. Mathematical Background

### 1.1 The Test-Time Regression Perspective

Wang et al. show that the core of linear attention is a weighted least-squares problem maintained online. Given key-value pairs $(k_t, v_t)$, the model maintains a memory matrix:

$$M_t = \arg\min_M \sum_{i=1}^{t} \gamma_i^{(t)} \| v_i - M \phi(k_i) \|^2$$

All practical variants approximate the analytical solution $M_t = V_t^\top \Gamma_t \Phi_t (\Phi_t^\top \Gamma_t \Phi_t)^{-1}$ by dropping the inverse covariance, yielding one step of (optionally preconditioned) gradient descent:

$$M_t \approx V_t^\top \Gamma_t \Phi_t = \sum_{i=1}^{t} \gamma_i^{(t)} v_i \phi(k_i)^\top$$

With geometrically decaying weights $\gamma_i^{(t)} = \prod_{j=i+1}^{t} \alpha_j$ and identity feature map, this admits a recurrence:

$$M_t = \alpha_t M_{t-1} + v_t k_t^\top, \qquad y_t = M_t q_t$$

The three TTR axes map to concrete choices:

| TTR Axis | Mathematical Object | Kernel Impact |
|----------|-------------------|--------------|
| Regression weights $\{\gamma_i^{(t)}\}$ | Decay structure: $\alpha_t$, gating | Determines the structured mask $L[i,j]$ within chunks and the inter-chunk decay factor |
| Function class $\mathcal{M}$ | Feature map $\phi$: identity, RoPE | Transforms $K$, $Q$ before they enter the matmuls |
| Optimization algorithm | Update rule: one-step GD, delta rule | Determines how the state $S$ propagates across chunks |

### 1.2 The Two Families

**SSD family.** Scalar or diagonal state transitions. The recurrence $M_t = \alpha_t M_{t-1} + v_t k_t^\top$ covers Mamba-2 ($\alpha_t = e^{\Delta_t A_t}$), GLA ($\alpha_t = \gamma_t$ per-head gate), RetNet ($\alpha_t = e^{-\lambda}$ fixed), RWKV-6 (data-dependent scalar), and mLSTM ($\alpha_t = \exp(f_t)$). The MIMO extension generalizes to rank-$r$ updates: $H_t = \alpha_t H_{t-1} + B_t X_t^\top$ with $B_t \in \mathbb{R}^{N \times r}$.

Mamba-3 extends the SSD family with trapezoidal discretization (mask factors as $L = L_{\text{decay}} \times L_{\text{conv}}$ where $L_{\text{conv}}$ is bidiagonal), data-dependent RoPE (complex eigenvalues $\Leftrightarrow$ cumulative rotation on $B, C$), and MIMO.

**Delta rule family.** State-dependent transition:

$$M_t = (I - \beta_t k_t k_t^\top) M_{t-1} + \beta_t v_t k_t^\top$$

Before storing a new association, erase the old value at key $k_t$. This is a rank-1 approximation to preconditioned gradient descent. The product $\prod_j (I - \beta_j k_j k_j^\top)$ within a chunk is computed via the WY representation: $I - WY^\top$ where $W, Y \in \mathbb{R}^{N \times C}$ are constructed incrementally.

### 1.3 Chunkwise Decomposition

All variants share a chunkwise structure. Partition the sequence into chunks of size $C$. The output for token $t$ in chunk $c$ decomposes as:

$$y_t = \underbrace{C_t^\top S_c}_{\text{inter-chunk}} + \underbrace{\sum_{j \in \text{chunk}} L[t,j] \cdot C_t^\top B_j x_j}_{\text{intra-chunk}}$$

This involves exactly 4 matrix multiplications per chunk:

| Phase | Computation | Dimensions | Purpose |
|-------|-------------|------------|---------|
| INTRA1 | $C \times B^\top$ | $(L, N) \times (N, L) \to (L, L)$ | Raw intra-chunk attention |
| INTRA2 | $\tilde{Q} \times X$ | $(L, L) \times (L, D) \to (L, D)$ | Masked intra-chunk output |
| INTER1 | $\tilde{B}^\top \times X$ | $(N, L) \times (L, D) \to (N, D)$ | Chunk's contribution to state |
| INTER2 | $C \times S$ | $(L, N) \times (N, D) \to (L, D)$ | Inter-chunk output |

These 4 matmuls are structurally identical across all variants. What varies is the preprocessing of their inputs and postprocessing of their outputs.

### 1.4 The Covariance Gap

The TTR framework identifies a shared weakness: all variants approximate $\Phi_t^\top \Gamma_t \Phi_t \approx I$, dropping the inverse covariance. Mamba-3's three innovations are orthogonal to this—they improve discretization accuracy, feature expressiveness, and memory capacity respectively, but none incorporate any form of $(\Phi_t^\top \Gamma_t \Phi_t)^{-1}$.

A diagonal preconditioner—maintaining running estimates $\sigma_n^2 = \sum_i \gamma_i k_{i,n}^2$ in $N$ extra registers—would partially address this. The correction costs $\sim ND$ extra divisions per chunk, which is negligible. Whether this helps in practice is unknown (Wang et al. demonstrate the issue on synthetic tasks; whether it persists at scale with learned representations is an open question). We note it as a research direction the framework would make trivial to test—not as a prediction of success.

---

## 2. Architectural Hypothesis

### 2.1 The Skeleton Claim

Analysis of the Mamba-2 CuTe DSL kernel reveals a warp-specialized architecture with 16 warps in 7 groups:

| Group | Warps | Registers/Thread | Work |
|-------|-------|-----------------|------|
| TMA Load (K/Q) | 1 | 24 | Async load $B$, $C$ from HBM to SMEM |
| TMA Load (V/Δ) | 1 | 24 | Async load $X$, $\Delta$, cumsum($\Delta$) |
| MMA Intra | 1 | 24 | INTRA1 + INTRA2 on Tensor Core |
| MMA Inter | 1 | 24 | INTER1 + INTER2 on Tensor Core |
| Pre-Inter | 4 | 168 | Key preprocessing, state propagation (CUDA cores) |
| Pre-Intra | 4 | 208 | Mask computation from INTRA1 result (CUDA cores) |
| Epilogue | 4 | 112 | Combine inter+intra outputs, TMA store (CUDA cores) |

The state $S \in \mathbb{R}^{N \times D}$ is distributed across the 4 pre-inter warps' register files, persisting for the entire sequence.

**The hypothesis:** the TMA warps, MMA warps, pipeline barriers, TMEM/SMEM allocation, and tile scheduling constitute a reusable skeleton. Variant-specific behavior is confined to the 3 preprocessing/postprocessing warp groups.

**Known risks to the hypothesis:**
- SMEM layouts are computed from MMA tile shapes (`make_smem_layout_a/b`). If MIMO changes the effective tile shape, layouts must be recomputed.
- Pipeline stage counts are hardcoded (`return 2, 2, 1, 2`). If pre-inter latency varies dramatically across variants, optimal staging may too.
- TMEM offset planning assumes specific accumulator sizes. Additional TMEM consumers (e.g., WY intermediates) require extending the planner.
- The register budget of 168 for pre-inter is sufficient for SSD's trivial element-wise work. DeltaNet's WY construction requires substantially more storage.

Phase 1 is designed to test the hypothesis concretely and discover which of these risks materialize.

### 2.2 The Four Customization Points

Variant-specific logic maps to four operations with well-defined contracts:

**`key_preprocess`** (pre-inter warps). Input: raw keys in SMEM + decay parameters. Output: scaled keys in internal SMEM buffer, conforming to INTER1_MMA's A-operand layout. For SSD: element-wise exp + multiply (~100 FMAs). For DeltaNet: WY construction + $\beta$-scaling ($O(C^2 N)$ FMAs).

**`mask_apply`** (pre-intra warps). Input: INTRA1 accumulator in TMEM + decay parameters. Output: masked $\tilde{Q}$ in TMEM, conforming to INTRA2_MMA's A-operand layout. For SSD: the segmented sum $Q[i,j] = (CB^\top)[i,j] \cdot \delta_j \cdot \exp(\text{cumsum}_i - \text{cumsum}_j)$ for $i \geq j$.

**`state_propagate`** (pre-inter warps). Input: INTER1 accumulator in TMEM + persistent state in registers + decay parameters. Output: updated state + state written to SMEM for INTER2_MMA. For SSD: $S \leftarrow e^{\Delta_{\text{last}}} \cdot S + \text{INTER1\_ACC}$ (one FMA per element). For DeltaNet: $S \leftarrow (I - WY^\top)(\gamma S) + \text{INTER1\_ACC}$ (two auxiliary matmuls).

**`output_combine`** (epilogue warps). Input: INTRA2 and INTER2 accumulators in TMEM + decay parameters. Output: combined $Y$ in SMEM for TMA store. For SSD: $y = \exp(\text{cumsum}) \cdot y_{\text{inter}} + y_{\text{intra}} + D \cdot x$.

These contracts are the foundation for modularity. They specify what each customization point receives and must produce, without dictating how. Future extensions (custom mask functions, custom transitions) would target these same contracts.

---

## 3. Honest Assessment of DeltaNet Fusion

The SSD family is the easy case—pre-inter work is trivially cheap. DeltaNet is the hard case and the project's critical risk. We owe it a careful analysis.

### 3.1 Cycle Budget

At $C = 64$, $N = 128$, $D = 64$:

| Operation | FMAs | Character |
|-----------|------|-----------|
| WY construction: $\sum_{t=1}^{C} 2tN \approx C^2 N$ | 524K | **Sequential** across $C$; parallel across $N$ |
| $Y^\top S$: $C \times N \times D$ | 524K | Fully parallel |
| $W(Y^\top S)$: $N \times C \times D$ | 524K | Fully parallel |
| Decay + accumulate | 8K | Trivial |
| **Total** | **~1.6M** | |

Across 128 threads (4 warps): ~12.5K FMAs/thread. At 1 FMA/thread/cycle on SM100: ~12.5K cycles best case. Accounting for the sequential dependency chain in WY construction and likely register spills, a realistic estimate is **15–20K cycles**.

The 4 MMA phases at these dimensions take roughly **8–12K cycles** in aggregate (including pipeline overhead). So pre-inter is likely on the critical path for DeltaNet, probably by 1.5–2×.

### 3.2 Why This Isn't Fatal

Even with pre-inter as the bottleneck, the fused DeltaNet kernel still eliminates:
- Redundant HBM reads of $K$, $V$, $Q$ (each loaded once, consumed by 2 MMA phases)
- HBM materialization of the state $S$ between chunks
- Kernel launch overhead between the 3 separate kernels

In FLA's 3-kernel approach, the state write/read alone costs $2 \times N \times D \times \text{sizeof(float)} = 2 \times 128 \times 64 \times 4 = 64\text{KB}$ per chunk per head, hitting HBM at ~2TB/s → ~32ns per chunk. Over many chunks and heads this adds up. The fused kernel eliminates all of it.

The net effect is uncertain—it depends on the balance between pre-inter's extra cycles and the HBM savings—but it's plausible that the fused kernel is faster even with reduced MMA utilization. The profiling data from Phase 2 will resolve this.

### 3.3 Mitigations

If pre-inter proves to be a severe bottleneck:

1. **Warp rebalancing.** Expand pre-inter from 4 to 6–8 warps by borrowing from epilogue. For DeltaNet the epilogue is lightweight (no $Y += XD$ fusion needed). This roughly halves per-thread work.

2. **Chunk size reduction.** WY cost is $O(C^2 N)$: halving $C$ from 64 to 32 reduces WY work by 4×, at the cost of 2× more inter-chunk overhead. The auto-tuning infrastructure in Phase 3 will find the optimal point.

3. **Partial offload.** Move the WY construction into a separate lightweight preprocessing kernel that writes $W$, $Y$ to HBM. The main fused kernel reads $W$, $Y$ from SMEM (loaded by TMA) and performs only the state propagation matmuls in pre-inter. This trades some fusion benefit for removing the sequential bottleneck.

4. **Accept the cost.** If the fused kernel is 1.1× vs FLA instead of 1.5×, it's still a win. And it provides architectural value: the unified framework makes it easy to experiment with DeltaNet variants (different $\beta$ schedules, gating structures) without rewriting kernels.

---

## 4. Implementation Plan

### Phase 1: Skeleton Extraction + SSD Family (Weeks 1–6)

**Goal:** Extract the invariant skeleton from the Mamba-2 kernel and validate it by instantiating SSD (bit-exact) and GLA (new variant).

| Week | Task | Exit Criterion |
|------|------|---------------|
| 1–2 | Factor existing kernel: annotate every line as SKELETON or SSD-SPECIFIC. Document assumptions. | Assumption catalog with risk assessment per variant |
| 3–4 | Parameterize skeleton. Replace hardcoded SSD assumptions with variant-provided parameters. | SSD instantiation: bit-identical output, ≤2% perf regression |
| 5–6 | Implement GLA fragments. Test against PyTorch reference. | GLA: ≤10⁻³ relative error, performance within 15% of SSD |

**What we learn:** Which skeleton assumptions are truly invariant, which need parameterization, and how much work parameterization requires.

### Phase 2: Delta Rule Extension (Weeks 7–14)

**Goal:** Implement Gated DeltaNet within the skeleton. Characterize the pre-inter bottleneck.

| Week | Task | Exit Criterion |
|------|------|---------------|
| 7–8 | WY construction in pre-inter warps. Register pressure analysis. | Working WY + profiled register usage |
| 9–10 | Householder state propagation. GDN mask_apply + output_combine. | Correct output vs FLA reference |
| 11–12 | Profile. Measure pre-inter vs MMA critical path. Try mitigations. | Quantified speedup/regression vs FLA 3-kernel |
| 13–14 | Buffer for iteration on mitigations, or early start on Phase 3. | Documented performance characterization |

**Decision gate (week 12).** If fused GDN is ≥1.0× vs FLA: proceed as planned. If fused GDN is <1.0× and mitigations don't help: document findings, recommend 2-kernel hybrid for delta family, proceed with Phase 3 for SSD family only.

### Phase 3: Mamba-3 Features + Novel Combinations (Weeks 15–20)

**Goal:** Complete the framework with Mamba-3's features and enable novel combinations.

| Week | Task | Exit Criterion |
|------|------|---------------|
| 15–16 | Trapezoidal mask ($L_{\text{decay}} \times L_{\text{conv}}$) in pre-intra | Correct output vs Mamba-3 reference |
| 17–18 | Data-dependent RoPE on $B$, $C$. MIMO rank-$r$ INTER1 shapes. | Full Mamba-3: correct, ≥90% of hand-written kernel |
| 19–20 | Novel combinations: preconditioned GD (diagonal), DeltaNet+RoPE. Evaluation. | ≥1 novel variant benchmarked at scale |

---

## 5. The API

### 5.1 What We Build Now

A specification-level API where each component selects a hand-written code fragment:

```python
from flex_linear_attn import FlexLinearAttention, decay, feature_map, transition

# Known variants
mamba2 = FlexLinearAttention(
    decay=decay.exponential(data_dependent=True),
    features=feature_map.identity(),
    transition=transition.one_step_gd(),
)

gla = FlexLinearAttention(
    decay=decay.gated(per_head=True),
    features=feature_map.identity(),
    transition=transition.one_step_gd(),
)

gated_delta_net = FlexLinearAttention(
    decay=decay.gated(per_head=True),
    features=feature_map.identity(),
    transition=transition.delta_rule(),
)

mamba3 = FlexLinearAttention(
    decay=decay.exponential(data_dependent=True, discretization="trapezoidal"),
    features=feature_map.rope(data_dependent=True),
    transition=transition.one_step_gd(mimo_rank=4),
)

# Novel combination enabled by the framework
mamba3_preconditioned = FlexLinearAttention(
    decay=decay.exponential(data_dependent=True, discretization="trapezoidal"),
    features=feature_map.rope(data_dependent=True),
    transition=transition.preconditioned_gd(preconditioner="diagonal", mimo_rank=4),
)
```

Each component is a selection key, not a symbolic expression. `decay.gated(per_head=True)` selects the pre-written mask_apply and output_combine fragments for gated decay. `transition.delta_rule()` selects the pre-written key_preprocess and state_propagate fragments for WY construction and Householder propagation.

This is less ambitious than a general-purpose compiler but delivers concrete value: a researcher can try new *combinations* of known components without writing kernel code.

### 5.2 Where Future Automation Plugs In

The 4 customization point contracts are designed as stable interfaces that future tooling can target:

- **Custom mask tracing (Tier 2, future).** A `score_mod`-style callback compiled to `mask_apply` code. The interface contract tells the compiler exactly what it must produce. Prerequisite: validate that the contract is stable across 4+ variants before building a compiler targeting it.

- **Custom transition classes (Tier 3, future).** A structured class compiled to `key_preprocess` + `state_propagate` code. Requires exposing the register budget and cycle constraints in a way users can reason about. Phase 2's experience with DeltaNet's register pressure will inform the design.

- **Backward pass (future).** The forward kernel's customization points define what needs adjoint counterparts. A backward skeleton with its own (different) warp structure could share the variant-specific differentiation logic.

- **Auto-tuning (future).** Chunk size, warp allocation, pipeline staging as tunable parameters per variant. Phase 2's profiling data for DeltaNet will establish whether this is worthwhile or whether per-variant manual tuning suffices.

---

## 6. Performance Targets

| Variant | Target | Baseline | Confidence |
|---------|--------|----------|-----------|
| SSD (regenerated) | ≥98% throughput | Original Mamba-2 CuTe DSL | High — same code, factored |
| GLA | Establish fused baseline | FLA 3-kernel Triton on same GPU | High — trivial pre-inter, full fusion benefit |
| Gated DeltaNet | ≥1.0× vs FLA (parity) | FLA 3-kernel Triton on same GPU | Medium — pre-inter cost may offset fusion gains |
| Gated DeltaNet | ≥1.3× vs FLA (stretch) | Same | Low-medium — requires pre-inter ≤ MMA budget |
| Mamba-3 | ≥90% of hand-written | Published Mamba-3 CuTe DSL | Medium — MIMO shape changes add overhead |

The GDN target is deliberately conservative. Parity with FLA's 3-kernel approach would already validate the architecture (same speed, less memory bandwidth, simpler code structure). Exceeding it by 1.3× is the goal, not the threshold.

---

## 7. Risks

| Risk | Likelihood | Impact | Detection | Mitigation |
|------|-----------|--------|-----------|------------|
| SMEM layout assumptions encode SSD-specific shapes | Medium | Medium | Phase 1, weeks 1–2 | Parameterize layouts by variant config |
| Pre-inter bottleneck for GDN exceeds MMA by >2× | Medium-High | Medium | Phase 2, weeks 11–12 | Warp rebalancing, $C$ reduction, partial offload |
| Register pressure in pre-inter insufficient for WY | Medium | High | Phase 2, weeks 7–8 | `setmaxregister` adjustment, WY tiling to SMEM |
| MIMO changes INTER1 tile shape, cascading through skeleton | High | Medium | Phase 3, weeks 17–18 | Tile shape as skeleton parameter, accept per-shape SMEM configs |
| Pipeline staging needs per-variant tuning | Medium | Low | Phase 1–2 | Add staging to variant config |
| Diagonal preconditioner doesn't help at scale | Medium | Low | Phase 3, week 20 | Framework value doesn't depend on any single novel variant |

---

## 8. Success Criteria

**Phase 1 (weeks 1–6):**
- SSD from skeleton: bit-identical output, ≤2% performance regression
- GLA from skeleton: ≤10⁻³ relative error, working fused kernel
- A clear catalog of what's invariant and what needed parameterization

**Phase 2 (weeks 7–14):**
- Gated DeltaNet: correct output vs FLA reference
- Quantified pre-inter cost: cycle breakdown, bottleneck analysis
- Honest performance comparison vs FLA

**Phase 3 (weeks 15–20):**
- Full Mamba-3: correct output, ≥90% of hand-written kernel
- At least 1 novel combination benchmarked (even if it doesn't improve perplexity—the ability to test it is the deliverable)
- Documented interfaces ready for future Tier 2/3 extension

**Overall:** The project succeeds if the skeleton generalizes to ≥3 variants with competitive performance and clean interfaces. It provides partial value even if DeltaNet proves difficult to fuse, as long as we produce honest documentation of where the architecture's limits are.

# OLD Flex Linear Attention by Claude

**A Composable Framework for Hardware-Efficient Linear Attention Kernels**

Author: Reuben · GPU Kernel Engineering · February 2026 · Draft

---

## Executive Summary

Linear attention variants—Mamba-2, GLA, RetNet, DeltaNet, Mamba-3—have emerged as the most promising alternatives to softmax attention for long-context sequence modeling. Wang et al.'s *test-time regression* (TTR) framework reveals that all of these are instances of a single parametric family, differing only in three choices: regression weights $\gamma_i^{(t)}$, function class $\mathcal{M}$, and optimization algorithm. Despite this mathematical unity, every variant today requires its own hand-written CUDA kernel.

**Flex Linear Attention** is a composable kernel framework that lets researchers define novel linear attention variants at the mathematical level and automatically generates fully-fused, warp-specialized kernels targeting SM100 (Blackwell) and SM90 (Hopper). The framework exploits a structural insight from analysis of the Mamba-2 CuTe DSL kernel: the fused chunkwise linear attention kernel has a *fixed skeleton* (~2,800 lines of invariant infrastructure) and exactly *4 narrow customization points* that vary across variants.

::: info **Core thesis.** The entire space of linear attention variants can be compiled into a single fused kernel architecture by treating the 4 customization points as generated code regions. The user specifies mathematics; the framework produces a production-quality kernel.
:::

---

## 1. Mathematical Foundations

### 1.1 The Test-Time Regression Perspective

Wang et al. show that a broad class of sequence models can be understood as performing *regression at test time*. Given a sequence of key-value pairs $(k_1, v_1), \ldots, (k_t, v_t)$, the model maintains a memory matrix $M_t$ that solves a weighted least-squares problem:

$$M_t = \arg\min_M \sum_{i=1}^{t} \gamma_i^{(t)} \| v_i - M \phi(k_i) \|^2$$

where $\gamma_i^{(t)}$ are regression weights (controlling forgetting), $\phi$ is a feature map (defining the function class), and the optimization algorithm determines how $M_t$ is computed from $M_{t-1}$.

**Analytical solution (full regression).** Solving in closed form gives:

$$M_t = V_t^\top \Gamma_t \Phi_t \left( \Phi_t^\top \Gamma_t \Phi_t \right)^{-1}$$

where $\Phi_t = [\phi(k_1), \ldots, \phi(k_t)]^\top$, $V_t = [v_1, \ldots, v_t]^\top$, and $\Gamma_t = \text{diag}(\gamma_1^{(t)}, \ldots, \gamma_t^{(t)})$. The output for query $q_t$ is $y_t = M_t \phi(q_t)$.

**One-step gradient descent (linear attention approximation).** Starting from $M_t^{(0)} = 0$ and taking one gradient step:

$$M_t^{(1)} = V_t^\top \Gamma_t \Phi_t$$

This drops the inverse covariance term $(\Phi_t^\top \Gamma_t \Phi_t)^{-1}$, equivalent to approximating $K_t^\top \Gamma_t K_t \approx I$. This is the fundamental approximation that all linear attention variants make, and the fundamental weakness the TTR framework identifies.

### 1.2 Unified Recurrence

With geometrically decaying weights $\gamma_i^{(t)} = \prod_{j=i+1}^{t} \alpha_j$ and identity feature map, the one-step GD solution admits a recurrence:

$$M_t = \alpha_t M_{t-1} + v_t k_t^\top$$

Output: $y_t = M_t q_t$. This is the *state space duality* (SSD) form. Every SSD-family variant (Mamba-2, GLA, RetNet, RWKV-6, mLSTM) is an instance of this recurrence with different choices of $\alpha_t$.

The MIMO extension generalizes to rank-$r$ updates:

$$H_t = \alpha_t H_{t-1} + B_t X_t^\top, \quad H_t \in \mathbb{R}^{N \times P}, \quad B_t \in \mathbb{R}^{N \times r}, \quad X_t \in \mathbb{R}^{P \times r}$$

The delta rule variants modify the transition itself:

$$M_t = (I - \beta_t k_t k_t^\top) M_{t-1} + \beta_t v_t k_t^\top$$

which performs an online correction: before adding the new association, erase the old value stored at key $k_t$. This implements one step of *preconditioned* gradient descent with a rank-1 approximation to the inverse covariance.

### 1.3 Chunkwise Decomposition

All variants share a common chunkwise structure that enables hardware-efficient computation. Partition the sequence into chunks of size $C$. For chunk $c$ with tokens $\{(c-1)C+1, \ldots, cC\}$:

**Output decomposition.** The output for token $t$ in chunk $c$ decomposes as:

$$y_t = \underbrace{C_t^\top S_c}_{\text{inter-chunk}} + \underbrace{\sum_{j=(c-1)C+1}^{t} L[t,j] \cdot C_t^\top B_j x_j}_{\text{intra-chunk}}$$

where $S_c$ is the recurrent state at the boundary of chunk $c$, and $L[t,j]$ is the structured mask encoding decay between positions $t$ and $j$.

**State propagation.** The state updates across chunk boundaries:

$$S_c = f(S_{c-1}, \{k_j, v_j\}_{j \in \text{chunk } c})$$

For SSD variants: $S_c = \alpha_{\text{last}} S_{c-1} + \sum_j \tilde{B}_j X_j^\top$ (element-wise decay + rank-1 updates accumulated via matmul).

For delta rule variants: $S_c = \left(\prod_{j \in \text{chunk}} (I - \beta_j k_j k_j^\top)\right) \alpha S_{c-1} + \text{correction}$, where the Householder product is computed via the WY representation.

**Four matmuls.** The chunkwise decomposition involves exactly 4 matrix multiplications per chunk:

| MMA Phase | Computation | Dimensions | Purpose |
|-----------|-------------|------------|---------|
| INTRA1 | $C \times B^\top$ | $(L \times N) \times (N \times L) \to L \times L$ | Raw intra-chunk attention matrix |
| INTRA2 | $Q \times X$ | $(L \times L) \times (L \times D) \to L \times D$ | Intra-chunk output (after masking) |
| INTER1 | $\tilde{B}^\top \times X$ | $(N \times L) \times (L \times D) \to N \times D$ | State update contribution |
| INTER2 | $C \times S$ | $(L \times N) \times (N \times D) \to L \times D$ | Inter-chunk output |

These 4 matmuls are **invariant across all linear attention variants**. What changes is only the preprocessing applied to their inputs and postprocessing applied to their outputs.

### 1.4 How Existing Variants Map to This Structure

**Mamba-2 / SSD.** Scalar data-dependent gate $\alpha_t = e^{\Delta_t A_t}$.

- *Mask*: $L[i,j] = \delta_j \cdot \exp\left(\sum_{m=j+1}^{i} \Delta_m A_m\right)$ for $i \geq j$, zero otherwise
- *State propagation*: $S_c = e^{\Delta_{\text{last}}} S_{c-1} + \text{INTER1\_ACC}$

**Mamba-3.** Trapezoidal discretization + complex-valued state + MIMO.

- *Mask*: $L = L_{\text{decay}} \times L_{\text{conv}}$ where $L_{\text{conv}}$ is bidiagonal with entries $(\beta_t, \gamma_t)$. This factorization (Mamba-3 Eq. 5) absorbs the short convolution into the discretization scheme.
- *Feature map*: Complex SSM $\Leftrightarrow$ data-dependent RoPE on $B, C$ (Mamba-3 Proposition 3). Effective keys become $\tilde{k}_t = \left(\prod_{i \leq t} R_i^\top\right) B_t$ where $R_i$ are rotation matrices parameterized by $\theta_i$.
- *State propagation*: Same element-wise structure as SSD but with doubled (real-valued) state dimension and rotation applied to $B, C$ before the MMAs.

**Gated DeltaNet.** Per-head gating + delta rule.

- *Mask*: Same gated structure as GLA
- *State propagation*: $S_c = \left(\prod_j (I - \beta_j k_j k_j^\top)\right) \gamma_c S_{c-1} + \text{correction}$. The Householder product $\prod_j (I - \beta_j k_j k_j^\top)$ is computed via the WY representation: $\prod_j (I - \beta_j k_j k_j^\top) = I - W Y^\top$ where $W, Y \in \mathbb{R}^{N \times C}$ are constructed incrementally.

### 1.5 The Covariance Gap

The TTR framework reveals a fundamental weakness shared by *all* existing linear attention variants, including Mamba-3 and Gated DeltaNet. The analytical regression solution includes the inverse covariance term $(\Phi_t^\top \Gamma_t \Phi_t)^{-1}$. All practical variants drop this, approximating $K_t^\top \Gamma_t K_t \approx I$.

Mamba-3's three innovations (trapezoidal discretization, RoPE features, MIMO) improve the *dataset quality*, *feature space*, and *update expressiveness* respectively—but all three are orthogonal to the covariance approximation. They improve different axes of the regression:

- Trapezoidal: better discretization → less noise in the regression dataset → $O(\Delta^2)$ global error vs $O(\Delta)$
- RoPE: richer feature map → more expressive function class → cumulative rotation operators vs identity
- MIMO: rank-$r$ updates → memory capacity saturates $r\times$ faster

None incorporate $(K_t^\top \Gamma_t K_t)^{-1}$. The framework predicts that even a *diagonal* approximation to this inverse covariance—trivially cheap to maintain in registers—should yield measurable gains on top of Mamba-3's innovations. This is a key research direction that the Flex Linear Attention framework is designed to enable.

---

## 2. Kernel Architecture

### 2.1 Warp-Specialized Skeleton

The fused kernel uses 16 warps organized into 7 specialized groups. The critical architectural feature is that *preprocessing warps run on CUDA cores concurrently with MMA warps on Tensor Cores*, hiding preprocessing latency behind matmul execution.

| Warp Group | Warps | Role | Hardware |
|-----------|-------|------|----------|
| TMA Load (K/Q) | 1 | Async load keys and queries from HBM to SMEM | TMA unit |
| TMA Load (V/Δ) | 1 | Async load values and decay parameters | TMA unit |
| MMA Intra | 1 | INTRA1: $Q \times K^\top \to L \times L$; INTRA2: $\tilde{Q} \times V \to L \times D$ | Tensor Core |
| MMA Inter | 1 | INTER1: $\tilde{K}^\top \times V \to N \times D$; INTER2: $Q \times S \to L \times D$ | Tensor Core |
| Pre-Inter | 4 | Key preprocessing + state propagation | CUDA cores |
| Pre-Intra | 4 | Mask computation from INTRA1 result | CUDA cores |
| Epilogue | 4 | Combine intra+inter outputs, TMA store | CUDA cores |

The recurrent state $S \in \mathbb{R}^{N \times D}$ is distributed across the 4 pre-inter warps' register files. It never touches SMEM or HBM between chunks.

### 2.2 The Four Customization Points

Every linear attention variant is defined by the behavior at exactly 4 points within this skeleton:

**Point 1: `key_preprocess`** (pre-inter warps). Transforms keys before INTER1_MMA. For SSD, this scales $B$ by $\exp(\Delta_{\text{last}} - \text{cumsum}) \cdot \delta$. For DeltaNet, this applies $\beta$-weighting and constructs the WY representation.

**Point 2: `mask_apply`** (pre-intra warps). Computes the structured mask $Q$ from the raw INTRA1 result $C \times B^\top$. For SSD, this is the segmented sum: $Q[i,j] = (C B^\top)[i,j] \cdot \delta_j \cdot \exp(\text{cumsum}[i] - \text{cumsum}[j])$ for $i \geq j$, with $-\infty$ for $i < j$.

**Point 3: `state_propagate`** (pre-inter warps). Updates the recurrent state across chunks. For SSD: $S \leftarrow e^{\Delta_{\text{last}}} \cdot S + \text{INTER1\_ACC}$ (element-wise FMA). For DeltaNet: $S \leftarrow (I - WY^\top)(\gamma \cdot S) + \text{INTER1\_ACC}$ (two small matmuls + element-wise).

**Point 4: `output_combine`** (epilogue warps). Merges inter-chunk and intra-chunk contributions. For SSD: $y = \exp(\text{cumsum}) \cdot y_{\text{inter}} + y_{\text{intra}}$.

These are not independent. Points 1 and 3 are jointly determined by the state transition rule. Points 2 and 4 are jointly determined by the decay structure. The user defines *two* things, and the framework generates all 4.

### 2.3 Why Full Fusion Wins

Compared to Flash Linear Attention's 3-kernel chunkwise approach:

- **Single HBM load per tensor.** $K$, $V$, $Q$ are loaded once by TMA into SMEM and consumed by multiple MMA phases. A 3-kernel approach re-reads from HBM for each kernel.
- **State in registers.** $S$ is held persistently across chunks in pre-inter warp registers. No materialization to global memory between chunks.
- **Free preprocessing.** Key scaling, mask computation, and state propagation run concurrently with Tensor Core MMAs on different functional units.
- **Zero launch overhead.** The chunk loop is a software loop inside a single persistent kernel.

---

## 3. User Interface

The API mirrors the TTR framework's three-axis parameterization, creating three tiers from high-level composition to low-level customization.

### 3.1 Tier 1: Composable Specification

Predefined building blocks for known variants and novel combinations:

```python
from flex_linear_attn import FlexLinearAttention, decay, feature_map, transition

# Mamba-2 / SSD
mamba2 = FlexLinearAttention(
    decay=decay.exponential(data_dependent=True),
    features=feature_map.identity(),
    transition=transition.one_step_gd(),
)

# Gated DeltaNet
gated_delta_net = FlexLinearAttention(
    decay=decay.gated(per_head=True),
    features=feature_map.identity(),
    transition=transition.delta_rule(),
)

# Mamba-3
mamba3 = FlexLinearAttention(
    decay=decay.exponential(data_dependent=True, discretization="trapezoidal"),
    features=feature_map.rope(data_dependent=True),
    transition=transition.one_step_gd(mimo_rank=4),
)

# Novel: Mamba-3 + diagonal preconditioner
mamba3_precond = FlexLinearAttention(
    decay=decay.exponential(data_dependent=True, discretization="trapezoidal"),
    features=feature_map.rope(data_dependent=True),
    transition=transition.preconditioned_gd(preconditioner="diagonal", mimo_rank=4),
)
```

The mapping from API axes to kernel customization points:

| TTR Axis | API Surface | Kernel Target |
|----------|------------|---------------|
| Regression weights $\{\gamma_i^{(t)}\}$ | `decay.*` | Pre-Intra (mask entries) + Epilogue (output scaling) |
| Function class $\mathcal{M}$ | `features.*` | TMA preprocessing (transform $K$, $Q$ before load) |
| Optimization algorithm | `transition.*` | Pre-Inter (key preprocess + state propagation) |

### 3.2 Tier 2: Custom Mask Function

A scalar callback—analogous to FlexAttention's `score_mod`—defining the structured mask $L[i,j]$ within a chunk:

```python
@flex_linear_attn.custom_mask
def my_mask(i, j, *, delta_cumsum, delta, beta):
    """Traced symbolically → pre_intra + epilogue code generation."""
    if i < j:
        return 0.0
    return beta[j] * exp(delta_cumsum[i] - delta_cumsum[j]) * delta[j]
```

The library traces this symbolically, identifies the causal structure ($i < j \to 0$), extracts the inter-chunk decay factor (evaluating the mask at $j = 0$, $i = C-1$ to get $\exp(\text{cumsum}[-1])$), and generates the pre_intra warp code.

**Constraint.** The mask must factor as $L = L_{\text{causal}} \odot L_{\text{decay}} \odot L_{\text{local}}$ where $L_{\text{decay}}$ has extractable inter-chunk decay. If not, the library raises a clear error.

### 3.3 Tier 3: Custom State Transition

For novel update rules beyond one-step GD and delta rule:

```python
@flex_linear_attn.custom_transition
class DiagonalPreconditioned:
    """Maintain running diagonal approx to K^T Γ K, precondition state updates."""

    extra_state = {'diag_cov': ('N',)}  # persistent across chunks, held in registers

    def preprocess_keys(self, keys, decay_params):
        return keys * exp(decay_params.cumsum[-1] - decay_params.cumsum) * decay_params.delta

    def propagate(self, state, diag_cov, inter1_acc, chunk_keys, decay_factor):
        # Update running diagonal covariance: Σ_new = α² Σ_old + Σ_j ||k_j||² e_j e_j^T
        new_diag_cov = decay_factor**2 * diag_cov + sum_squared_keys_per_dim
        # Preconditioned update: (K^T Γ K)^{-1} ≈ diag(1/σ²)
        preconditioned_acc = inter1_acc / (new_diag_cov.unsqueeze(-1) + eps)
        new_state = decay_factor * state + preconditioned_acc
        return new_state, new_diag_cov
```

The library inspects `extra_state` to allocate additional register storage, compiles `preprocess_keys` into B-scaling, and compiles `propagate` into state combination. All operations must be expressible as register-level compute (element-wise, reductions, small matmuls).

---

## 4. Compilation Pipeline

```
User specification (TTR terms)
  → Variant resolver: map (decay, features, transition) → 4 customization point definitions
  → Op tracing: symbolic IR for each point
  → Cost estimation: cycle count per warp group
  → Resource planning: warp allocation, register budgets, SMEM staging
  → Code generation: CuTe DSL fragments for each warp group
  → JIT compilation: insert fragments into skeleton → fused kernel binary
```

### 4.1 Cost Estimation

The cost estimation stage compares pre-inter cycle count against expected MMA latency. This determines whether preprocessing is fully hidden (the common case) or threatens to bottleneck the pipeline.

**SSD family.** Pre-inter does element-wise $\exp$ + FMA: ~100 cycles per chunk. Always hidden.

**Delta rule family.** WY construction is $O(C^2 N)$; state propagation requires two matmuls of dimensions $(N \times C) \times (C \times D)$ and $(N \times C) \times (C \times N)$. At $C = 64$, $N = 128$, $D = 64$:
- WY construction: $C^2 N = 64^2 \times 128 \approx 524\text{K}$ FMAs
- $Y^\top S$: $C \times N \times D = 64 \times 128 \times 64 \approx 524\text{K}$ FMAs
- $W \times (Y^\top S)$: $N \times C \times D = 128 \times 64 \times 64 \approx 524\text{K}$ FMAs
- Total: ~1.6M FMAs across 128 threads (4 warps) → ~12K FMAs/thread → ~6K cycles

The 4 main MMAs at these dimensions take ~2–4K cycles each on Tensor Core. With 4 phases sequentially, that's ~8–16K cycles of MMA pipeline time. The ~6K cycles of pre-inter work fits within this budget, especially accounting for pipeline bubbles.

### 4.2 Mitigation Strategies

If pre-inter threatens to exceed MMA latency:

- **Warp rebalancing.** Expand pre-inter from 4 to 8 warps, halving per-thread work. Shrink epilogue from 4 to 2 warps (epilogue is lightweight for variants without $Y \mathrel{+}= XD$ fusion).
- **Chunk size reduction.** Smaller $C$ reduces WY cost (quadratic in $C$) at the expense of more chunks. The library auto-tunes this tradeoff.

---

## 5. Variant Coverage

### 5.1 Family 1: SSD-Compatible

Variants with scalar/diagonal state transitions $h = \alpha \cdot h + B \otimes X$. Fully composable via Tier 1.

| Variant | $\alpha_t$ | Feature Map $\phi$ | State Rank | Pre-Inter Cost |
|---------|-----------|-------------------|-----------|----------------|
| Mamba-2 | $e^{\Delta_t A_t}$ (data-dep.) | Identity | 1 | Negligible |
| GLA | $\gamma_t$ (per-head gate) | Identity | 1 | Negligible |
| RetNet | $e^{-\lambda}$ (fixed) | $e^{i\theta t}$ (fixed RoPE) | 1 | Negligible |
| RWKV-6 | Data-dependent | Identity | 1 | Negligible |
| mLSTM | $\exp(f_t)$ (exponential gate) | Identity | 1 | Negligible |
| Mamba-3 SISO | $e^{\Delta_t A_t}$ + trapezoidal | Data-dep. RoPE | 1 | Negligible |
| Mamba-3 MIMO | Same as above | Data-dep. RoPE | $r$ | Negligible |

### 5.2 Family 2: Delta Rule

Variants with state-dependent transitions $(I - \beta k k^\top)$. Requires WY representation in pre-inter.

| Variant | Gate | $\beta$ | Pre-Inter Cost |
|---------|------|---------|----------------|
| DeltaNet | None | Learned scalar | ~6K cycles |
| Gated DeltaNet | Per-head $\gamma_t$ | Learned scalar | ~6K cycles |
| DeltaProduct | Multi-step | Per-step $\beta$ | ~12K cycles |

### 5.3 Novel Combinations Enabled

| Combination | TTR Interpretation | Predicted Effect |
|------------|-------------------|-----------------|
| Preconditioned Mamba-3 | Approximate Newton step: $S \leftarrow \alpha S + \text{diag}(K^\top \Gamma K)^{-1} V^\top K$ | Addresses $K^\top K \approx I$ approximation; ~200 extra FMAs/chunk (free) |
| DeltaNet + RoPE | Delta rule in rotated feature space | State-tracking (complex eigenvalues) + covariance correction (delta rule) |
| Trapezoidal GLA | $O(\Delta^2)$ discretization with per-head gating | Better accuracy without complex state |
| MIMO DeltaNet | Rank-$r$ delta rule updates | Faster memory saturation with covariance correction |

The preconditioned Mamba-3 combination is the most promising: the TTR framework identifies the covariance approximation as the core failure mode of linear attention, while Mamba-3's three innovations are all orthogonal to it. A diagonal preconditioner maintained in $N$ extra fp32 registers closes this gap at negligible computational cost.

---

## 6. Implementation Plan

### Phase 1: Skeleton Extraction (Weeks 1–4)

Extract the invariant kernel skeleton from the Mamba-2 CuTe DSL codebase. Factor the 4 customization points into interfaces with clear input/output contracts.

- **Deliverable:** Parameterized kernel template where the 4 customization points accept generated code fragments.
- **Validation:** Instantiate with SSD-specific code → bit-identical output to original Mamba-2 kernel, no performance regression.
- **Risk:** Hidden coupling between skeleton and SSD-specific logic (e.g., SMEM layout assumptions tied to exponential decay). Mitigation: incremental refactoring.

### Phase 2: SSD Family Code Generation (Weeks 5–8)

Implement the Tier 1 API for all SSD-compatible variants. Build the full compilation pipeline.

- **Deliverable:** Code generation for Mamba-2, GLA, RetNet, Mamba-3 (without MIMO). Each produces a fused kernel from a Tier 1 specification.
- **Validation:** Numerical correctness vs reference PyTorch. Performance ≥95% of hand-written kernels.
- **Key challenge:** Mamba-3's trapezoidal mask ($L = L_{\text{decay}} \times L_{\text{conv}}$ with bidiagonal $L_{\text{conv}}$) tests generality of mask_apply code generation.

### Phase 3: Delta Rule Support (Weeks 9–12)

Extend to state-dependent transitions. Implement WY representation as built-in `transition.delta_rule()`.

- **Deliverable:** Code generation for DeltaNet and Gated DeltaNet. Fused kernels.
- **Validation:** Correctness vs FLA references. Target ≥1.3× prefill throughput vs FLA's 3-kernel approach.
- **Key challenge:** WY construction is sequential and cycle-heavy. Validate pre-inter doesn't bottleneck; implement warp rebalancing if needed.

### Phase 4: MIMO + Feature Maps (Weeks 13–16)

Add rank-$r$ MIMO updates and data-dependent feature maps (RoPE).

- **Deliverable:** Full Mamba-3 support. Novel combinations (preconditioned Mamba-3, DeltaNet+RoPE) via Tier 1.
- **Validation:** Mamba-3 kernel matches published perplexity at 1.3B scale. Novel combinations show improvements on MQAR and state-tracking benchmarks.

### Phase 5: Custom Masks + Transitions (Weeks 17–20)

Implement Tier 2 (custom mask tracing) and Tier 3 (custom transition classes).

- **Deliverable:** Symbolic tracing of user-defined masks. Class-based custom transitions. Auto-tuning of chunk size and warp allocation.
- **Validation:** A novel variant definable in <50 lines of Python, kernel performance within 10% of hand-tuned.

### Phase 6: Backward Pass + Production (Weeks 21–28)

Backward-pass code generation, gradient checkpointing, production hardening.

- **Deliverable:** Full training support (forward + backward). PyTorch autograd integration. Test suite, documentation, examples.
- **Validation:** End-to-end training at 1.3B matches baseline perplexity. Training throughput within 5% of hand-tuned kernels.

---

## 7. Performance Targets

| Metric | Target | Baseline |
|--------|--------|----------|
| SSD family: prefill throughput | ≥95% of hand-written Mamba-2/3 | Mamba-2 CuTe DSL kernel |
| SSD family: decode latency | ≥95% of hand-written kernel | Mamba-3 fused decode kernel |
| Delta family: prefill throughput | ≥1.3× vs FLA 3-kernel | Flash Linear Attention (Triton) |
| Delta family: decode latency | Parity with FLA | Flash Linear Attention (Triton) |
| Novel combinations: prefill | Within 10% of theoretical roofline | Roofline model |
| Code generation time | <10 seconds per variant | N/A (currently months of manual work) |

---

## 8. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Skeleton extraction reveals hidden SSD coupling | Medium | High | Incremental refactoring; maintain bit-exact SSD compatibility at each step |
| Pre-inter bottleneck for WY at large $N$, $C$ | Medium | Medium | Warp rebalancing (4→8 warps); chunk size auto-tuning; 2-kernel fallback |
| CuTe DSL JIT too slow for interactive iteration | Low | Medium | Cache by variant hash; precompile common variants; interpreted reference mode |
| Custom masks that don't decompose for inter-chunk decay | Medium | Low | Clear errors explaining factorization requirement; decomposability test utilities |
| Backward pass significantly harder than forward | High | High | Longest phase; may need separate backward skeleton; hybrid fused-forward/chunked-backward |

---

## 9. Success Criteria

- **Correctness.** All generated kernels produce numerically correct output (≤10⁻³ relative error vs fp32 reference) for every supported variant.
- **Performance.** SSD-family ≥95% of hand-written baselines. Delta-family ≥1.3× vs FLA 3-kernel.
- **Expressiveness.** A researcher unfamiliar with GPU programming can define and benchmark a novel linear attention variant in <50 lines of Python within a single session.
- **Adoption.** At least 2 novel combinations (not possible with existing libraries) evaluated at ≥1B scale using framework-generated kernels.
- **Sustainability.** Adding a new variant to the Tier 1 library requires <100 lines and no skeleton modifications.

