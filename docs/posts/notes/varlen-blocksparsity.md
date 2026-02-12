---
title: Variable Sequence Length Blocksparsity for FlexAttention
---

# {{ $frontmatter.title }}

This proposal outlines an approach to blocksparsity for variable sequence length attention. To set some notation, `B` is batch, `H` is heads, `M` is number of $m$ blocks, and `N` is number of $n$ blocks.
Recall that we have four blocksparse metadata tensors:
- `mask_block_cnt: [B, H, M]` counting the number of blocks that need `mask_mod` application in a given row
- `mask_block_idx: [B, H, M, N]` listing the indices of said blocks
- `full_block_cnt: [B, H, M]` counting the number of fully-computed blocks in a given row
- `full_block_idx: [B, H, M, N]` listing the indices of said blocks.

In the variable sequence length setting, we concatenate sequences along the batch dimension. As a result, we propose to reshape our metadata tensors to `[H, total_M]` or `[H, total_N, max_N]` where `total_M` is the total number of $m$ blocks across all sequences and `max_N` is the maximum number of $n$ blocks processed by any one batch.

::: info
We could choose to have the `*_idx` tensors have shape `[H, total_N]`, in line with the "varlen philosophy". However, the added complexity may not be worth the memory savings. That being said, it wouldn't be terrible to make that change.
:::

We now need one more metadata tensor to accurately index into the `total_M` dimension: `cu_block_cnt: [B+1]`. This is similar to the `cu_seqlens` tensors that are already present in the varlen setting. It is necessary because when sequences in our batch are not divisible by tile size, it is not possible to easily compute the block offset from the seqlen offset. This can be created in the blocksparsity computation kernel or even in a separate preprocessing step, as it is very lightweight.

## What needs to change to support this

The bulk of the logic needed to deal with varlen blocksparsity will be wrapped into the `block_sparse_utils` file and/or `seqlen_info`.

```python
@cute.jit
# in block_sparse_utils.py
def get_curr_blocksparse_tensors(
    batch_idx: cutlass.Int32,
    head_idx: cutlass.Int32,
    m_block: cutlass.Int32,
    blocksparse_tensors: BlockSparseTensors,
    seqlen_info: SeqlenInfoQK,
) -> Tuple[cutlass.Int32, cute.Tensor, cutlass.Int32, Optional[cute.Tensor]]:
    mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx = blocksparse_tensors
    if const_expr(len(mask_block_cnt.shape) == 2):
        # In the case where we are varlen_q, blocksparse tensors have shape
        # [nheads, total_m_block] and [nheads, total_m_block, max_n_block]
        curr_m_block = seqlen_info.block_offset + m_block
        curr_mask_block_cnt = mask_block_cnt[head_idx, curr_m_block]
        curr_mask_block_idx = mask_block_idx[head_idx, curr_m_block, None]
        if const_expr(full_block_cnt is not None):
            curr_full_block_cnt = full_block_cnt[head_idx, curr_m_block]
            curr_full_block_idx = full_block_idx[head_idx, curr_m_block, None]
        else:
            curr_full_block_cnt = Int32(0)
            curr_full_block_idx = None
    else:
        curr_mask_block_cnt = mask_block_cnt[batch_idx, head_idx, m_block]
        curr_mask_block_idx = mask_block_idx[batch_idx, head_idx, m_block, None]
        if const_expr(full_block_cnt is not None):
            curr_full_block_cnt = full_block_cnt[batch_idx, head_idx, m_block]
            curr_full_block_idx = full_block_idx[batch_idx, head_idx, m_block, None]
        else:
            curr_full_block_cnt = Int32(0)
            curr_full_block_idx = None

    return (curr_mask_block_cnt, curr_mask_block_idx, curr_full_block_cnt, curr_full_block_idx)
```
And in `seqlen_info.py`
```python
@dataclass(frozen=True)
class SeqlenInfoQK:
    # ... existing params ...
    block_offset: cutlass.Int32 # NEW
    # ... existing params ...

    @staticmethod
    def create(
        # ... existing args ...
        mCuTotalMBlocks: Optional[cute.Tensor] = None, # NEW
        # ... existing args ...
    ):

        # ... existing setup ...

        # NEW
        block_offset = 0 if const_expr(mCuTotalMBlocks is None) else mCuTotalMBlocks[batch_idx]

        # ... existing setup ...
        return SeqlenInfoQK(
            # ...
            seqlen_q,
            seqlen_k,
            block_offset,
            has_cu_seqlens_q,
            has_cu_seqlens_k,
            # ...
        )

# Rest of file kept the same
```

The greatest difficulty is ensuring that the blocksparsity computation kernel emits the proper layout, but that's not a huge challenge and is in-progress already. We will have to carefully comb through all other files to ensure that any blocksparsity logic is updated with the varlen-flexible options (i.e. `get_curr_blocksparse_tensors`), but I believe everything is already compartmentalized.

## Updated approach to index tensors

For a varlen batch with sequence lengths contained in $\mathsf{seqlens}_Q$ and $\mathsf{seqlens}_K$ and batch index $b$, we let

$$
\mathsf{num}_m(b) = \mathsf{ceildiv}(\mathsf{seqlens}_Q[b], \mathsf{tile}_m)
\qquad
\mathsf{num}_n(b) = \mathsf{ceildiv}(\mathsf{seqlens}_K[b], \mathsf{tile}_n)
$$

We then define

$$
\mathsf{total}_m = \sum_{b \in B} \mathsf{num}_m(b)
\qquad
\mathsf{total}_n = \sum_{b \in B} \mathsf{num}_m(b) \cdot \mathsf{num}_n(b)
$$

We then have `*_block_cnt: [H, total_m]` and `*_block_idx: [H, total_n]`.
