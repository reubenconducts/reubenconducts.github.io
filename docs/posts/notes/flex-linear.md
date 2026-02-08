---
title: Flex Linear Attention -  Fast and Flexible Fused Linear Attention Implementation
---

# {{ $frontmatter.title }}

FlexAttention is an elegant API for defining custom attention variants within the FlashAttention framework. It does so in two parts:
1. Modifying pre-softmax scores using a `score_mod` callable.
2. Efficient implementation of block sparsity using on-device metadata tensors.

Linear attention variants partake in a *"state space duality"* between sequential and parallel forms of updates. They are also versions of "online learning" in that they are equivalently thought of as performing gradient descent per iteration. In this proposal, we discuss a possible "FlexLinearAttention" that allows for elegant and simple implementation of linear attention variants and can be worked into a hardware-aware implementation.

## A fused kernel for linear attention

Mamba 2 has a Blackwell-optimized fused kernel. It uses a complicated warp specialization scheme to properly pipeline the intra-chunk and inter-chunk recurrence.

$$
Y_i = \alpha H_i X_i + B_iH_{i-1} X_{i-1}
$$

# Flex Linear Attention by Claude

**A Composable Framework for Hardware-Efficient Linear Attention Kernels**

Author: Reuben · GPU Kernel Engineering · February 2026 · Draft

---

## Executive Summary

Linear attention variants—Mamba-2, GLA, RetNet, DeltaNet, Mamba-3—have emerged as the most promising alternatives to softmax attention for long-context sequence modeling. Wang et al.'s *test-time regression* (TTR) framework reveals that all of these are instances of a single parametric family, differing only in three choices: regression weights $\gamma_i^{(t)}$, function class $\mathcal{M}$, and optimization algorithm. Despite this mathematical unity, every variant today requires its own hand-written CUDA kernel.

**Flex Linear Attention** is a composable kernel framework that lets researchers define novel linear attention variants at the mathematical level and automatically generates fully-fused, warp-specialized kernels targeting SM100 (Blackwell) and SM90 (Hopper). The framework exploits a structural insight from analysis of the Mamba-2 CuTe DSL kernel: the fused chunkwise linear attention kernel has a *fixed skeleton* (~2,800 lines of invariant infrastructure) and exactly *4 narrow customization points* that vary across variants.

> **Core thesis.** The entire space of linear attention variants can be compiled into a single fused kernel architecture by treating the 4 customization points as generated code regions. The user specifies mathematics; the framework produces a production-quality kernel.

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

