---
title: Vectorized mask_mod proposal
---

# Vectorized `mask_mod` proposal

This proposal is a companion to our work on vectorizing `score_mod` for FlexAttention. It consists of two separate parts:
1. Vectorized `mask_mod` *evaluation*
2. Vectorized `mask_mod` *application*

We'll treat each separately, as they are largely orthogonal to one another. We will do our best to ensure the resulting framework is compatible with existing `mask_mod`s, though there may be some unforeseen challenges.

## Current setup

Currently, `mask_mod` callables have signature
```python
def mask_mod(b_idx, h_idx, q_idx, kv_idx, seqlen, aux_tensors) -> Boolean
```
Note for later that CuTe DSL stores `Boolean` values as `Int8`.

In the kernel (specifically in [`mask.py`](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/mask.py)), this evaluation is handled one `idx` at a time:
```python
for r in range_constexpr(nrow):
    for col in range_constexpr(ncol)
        cond = mask_mod(
            batch_idx,
            head_idx,
            r,
            col,
            seqlen,
            aux_tensors,
        )
        acc_S[r, col] = acc_S[r, col] if cond else -Float32.inf
```

## Proposed changes

For the sake of simplicity, we will focus only on `__vec_size__ == 8` currently, expanding later to the more general case.

### Application

We begin with *application*, as it can function without modifying `mask_mod` itself. The key point is to pack `mask_mod` return values into a single `Int8` and then ensure the compiler emits the `R2P` instruction. In pseudocode, we have something like this, for one row:
```python
for packed_col in range_constexpr(ncol // vec_size):
    mask = Int8(0)
    for i in range_constexpr(vec_size)
        curr_col = packed_col * vec_size + i
        cond = mask_mod(
            batch_idx,
            head_idx,
            r,
            col,
            seqlen,
            aux_tensors,
        )
        mask ^= ((cond & 1) << i) # not confident about this bit manipulation
    for i in range_constexpr(vec_size):
        curr_col = packed_col * vec_size + i
        cond = cutlass.Boolean(mask & (1 << i))
        acc_S[row, curr_col] = acc_S[row, curr_col] if cond else -Float32.inf
```
Really, we want to pack `mask` into as large values as we can -- comments in the R2P code indicate 24 bits is as much as the compiler can handle. Thus, we could store `mask` as an `Int32` where the top 8 bits are unused, or potentially we store mask as an `Int4` and iterate through it in multiples of 24.

Why is this change important, anyway? I'm not even sure that the compiler *isn't* emitting the `R2P` instruction in the existing `mask_mod` setup (really need to check that)! Well, reader I am imagining, that's because of optimized *evaluation*:

### Evaluation

Simply, vectorized `mask_mod` callables will return bit-packed masks. This does mean that `__vec_size__` is not a standalone property of a `mask_mod`, but instead a descriptor of its signature. This added complexity is why I want to keep application and evaluation separate, so that existing `mask_mod`s will drop cleanly into the new framework. Vectorized `mask_mod`s will not be better unilaterally, but there are many situations I could imagine them being a win:
- `mask_mod`s that use `aux_tensors` that are indexed into contiguously in `kv_idx`, so that we can vectorize the `aux_tensor` loads, as is now allowed for `score_mod`s
- `mask_mod`s that have some bounds checking where we know, for instance, that if `kv_idx[0] > bound`, that the entire vector is masked/unmasked. This is the case in causal+local attention.

As an example, take document masking. We have our `doc_id` tensor of shape `(batch, seqlen)`. With `__vec_size__ == 8`, we could modify the `mask_mod` to be the following:
```python
def document_mask_mod(b_idx, h_idx, q_idx, kv_idx, seqlen, aux_tensors) -> Int8:
    mask = Int8(0)

    b_idx0 = b_idx[0]
    q_idx0 = q_idx[0]
    kv_idx0 = kv_idx[0]
    doc_id = aux_tensors[0]
    doc_id_q = doc_id[b_idx0, q_idx0]
    vec_size = cute.size(kv_idx.shape)

    doc_id_cur = doc_id[b_idx, None]
    doc_id_cur = cute.flat_divide(doc_id_cur, (vec_size,))
    vec_start = kv_idx0 // vec_size
    gdoc_id = doc_id_cur[None, vec_start]
    doc_id_vec = cute.make_rmem_tensor_like(gdoc_id)

    cute.autovec_copy(gdoc_id, doc_id_vec)

    for idx in range_constexpr(vec_size):
        doc_id_idx = doc_id_vec[i]
        cond = doc_id_idx == doc_id_q
        mask ^= ((cond & 1) << idx)

    return mask

```

### Combining the two

Suppose we have a `mask_mod` with `__vec_size__ == 32`, so the return type is `Int32` (or whatever 32-bit dtype). The application loop would look something like this:
```python
for row in range_constexpr(nrows):
    masks = cute.make_rmem_tensor((4), dtype=Int32)
    for col_start in range_constexpr(ncols // vec_size): # probably == 4
        b_idx = ...
        h_idx = ...
        q_idx = ...
        kv_idx = ...
        masks[col_start] = mask_mod(
            b_idx,
            h_idx,
            q_idx,
            kv_idx,
            seqlen,
            aux_tensors,
        )
    for s in range_constexpr(cute.ceil_div(ncols, 24)):
        # have to be careful about cross-value iterations
        for i in range_constexpr(min(24, ncol - s * 24)):
            curr_col = s * 24 + i
            curr_mask = curr_col // 32
            curr_bit = curr_col % 32
            cond = (masks[curr_mask] >> curr_bit) & 1
            acc_S[curr_col] = acc_S[curr_col] if Boolean(cond) else -Float32.inf
```
