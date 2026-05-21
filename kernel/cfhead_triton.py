from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _cfhead_coarse_kernel(
        lut_ptr,
        index_t_ptr,
        out_ptr,
        V: tl.constexpr,
        N: tl.constexpr,
        C: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        block_id = tl.program_id(0)
        batch_id = tl.program_id(1)
        offsets = block_id * BLOCK_V + tl.arange(0, BLOCK_V)
        mask = offsets < V

        acc = tl.zeros((BLOCK_V,), dtype=tl.float32)
        for tile_id in tl.range(0, N):
            cluster = tl.load(index_t_ptr + tile_id * V + offsets, mask=mask, other=0).to(tl.int32)
            values = tl.load(
                lut_ptr + (batch_id * N + tile_id) * C + cluster,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            acc += values

        tl.store(out_ptr + batch_id * V + offsets, acc, mask=mask)


    @triton.jit
    def _cfhead_coarse_2bit_kernel(
        lut_ptr,
        packed_index_t_ptr,
        out_ptr,
        V: tl.constexpr,
        PACKED_V: tl.constexpr,
        N: tl.constexpr,
        C: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        block_id = tl.program_id(0)
        batch_id = tl.program_id(1)
        offsets = block_id * BLOCK_V + tl.arange(0, BLOCK_V)
        mask = offsets < V
        packed_offsets = offsets // 4
        bit_shifts = (offsets & 3) * 2

        acc = tl.zeros((BLOCK_V,), dtype=tl.float32)
        for tile_id in tl.range(0, N):
            packed = tl.load(
                packed_index_t_ptr + tile_id * PACKED_V + packed_offsets,
                mask=mask,
                other=0,
            ).to(tl.int32)
            cluster = (packed >> bit_shifts) & 3
            values = tl.load(
                lut_ptr + (batch_id * N + tile_id) * C + cluster,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            acc += values

        tl.store(out_ptr + batch_id * V + offsets, acc, mask=mask)


    @triton.jit
    def _cfhead_coarse_2bit_partial_kernel(
        lut_ptr,
        packed_index_t_ptr,
        partial_ptr,
        V: tl.constexpr,
        PACKED_V: tl.constexpr,
        N: tl.constexpr,
        C: tl.constexpr,
        NUM_TILE_BLOCKS: tl.constexpr,
        TILE_BLOCK: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        vocab_block_id = tl.program_id(0)
        batch_id = tl.program_id(1)
        tile_block_id = tl.program_id(2)

        offsets = vocab_block_id * BLOCK_V + tl.arange(0, BLOCK_V)
        vocab_mask = offsets < V
        packed_offsets = offsets // 4
        bit_shifts = (offsets & 3) * 2

        acc = tl.zeros((BLOCK_V,), dtype=tl.float32)
        tile_start = tile_block_id * TILE_BLOCK
        for local_tile in tl.range(0, TILE_BLOCK):
            tile_id = tile_start + local_tile
            tile_mask = tile_id < N
            packed = tl.load(
                packed_index_t_ptr + tile_id * PACKED_V + packed_offsets,
                mask=vocab_mask & tile_mask,
                other=0,
            ).to(tl.int32)
            cluster = (packed >> bit_shifts) & 3
            values = tl.load(
                lut_ptr + (batch_id * N + tile_id) * C + cluster,
                mask=vocab_mask & tile_mask,
                other=0.0,
            ).to(tl.float32)
            acc += values

        partial_base = (batch_id * NUM_TILE_BLOCKS + tile_block_id) * V
        tl.store(partial_ptr + partial_base + offsets, acc, mask=vocab_mask)


    @triton.jit
    def _cfhead_coarse_reduce_kernel(
        partial_ptr,
        out_ptr,
        V: tl.constexpr,
        NUM_TILE_BLOCKS: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        vocab_block_id = tl.program_id(0)
        batch_id = tl.program_id(1)

        offsets = vocab_block_id * BLOCK_V + tl.arange(0, BLOCK_V)
        mask = offsets < V
        acc = tl.zeros((BLOCK_V,), dtype=tl.float32)
        for tile_block_id in tl.range(0, NUM_TILE_BLOCKS):
            partial_base = (batch_id * NUM_TILE_BLOCKS + tile_block_id) * V
            acc += tl.load(partial_ptr + partial_base + offsets, mask=mask, other=0.0).to(tl.float32)

        tl.store(out_ptr + batch_id * V + offsets, acc, mask=mask)


    @triton.jit
    def _cfhead_fine_kernel(
        hidden_ptr,
        weight_ptr,
        candidate_ptr,
        out_ptr,
        K: tl.constexpr,
        H: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        candidate_block_id = tl.program_id(0)
        batch_id = tl.program_id(1)

        k_offsets = candidate_block_id * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K
        token_ids = tl.load(candidate_ptr + batch_id * K + k_offsets, mask=k_mask, other=0).to(tl.int64)

        acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for h_start in tl.range(0, H, BLOCK_H):
            h_offsets = h_start + tl.arange(0, BLOCK_H)
            h_mask = h_offsets < H
            h_vals = tl.load(hidden_ptr + batch_id * H + h_offsets, mask=h_mask, other=0.0).to(tl.float32)
            w_vals = tl.load(
                weight_ptr + token_ids[:, None] * H + h_offsets[None, :],
                mask=k_mask[:, None] & h_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(w_vals * h_vals[None, :], axis=1)

        tl.store(out_ptr + batch_id * K + k_offsets, acc, mask=k_mask)


def cfhead_coarse_forward(lut: torch.Tensor, index_t: torch.Tensor) -> torch.Tensor:
    """Compute CFHead coarse logits without materializing [B, chunk, V] gathers.

    Args:
        lut: [B, N_tile, C] float tensor.
        index_t: [N_tile, V] uint8 contiguous index map.

    Returns:
        [B, V] float32 coarse logits.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not lut.is_cuda or not index_t.is_cuda:
        raise RuntimeError("cfhead_coarse_forward requires CUDA tensors")
    if lut.dim() != 3 or index_t.dim() != 2:
        raise ValueError(f"expected lut [B,N,C] and index_t [N,V], got {tuple(lut.shape)} and {tuple(index_t.shape)}")

    batch_size, num_tiles, cluster_size = lut.shape
    index_tiles, vocab_size = index_t.shape
    if index_tiles != num_tiles:
        raise ValueError(f"index_t tiles={index_tiles} does not match lut tiles={num_tiles}")

    lut = lut.contiguous()
    index_t = index_t.contiguous()
    out = torch.empty((batch_size, vocab_size), device=lut.device, dtype=torch.float32)

    block_v = 256
    grid = (triton.cdiv(vocab_size, block_v), batch_size)
    _cfhead_coarse_kernel[grid](
        lut,
        index_t,
        out,
        vocab_size,
        num_tiles,
        cluster_size,
        BLOCK_V=block_v,
        num_warps=8,
    )
    return out


def cfhead_coarse_forward_2bit(lut: torch.Tensor, packed_index_t: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Compute CFHead coarse logits from a [N_tile, ceil(V/4)] 2-bit packed index map."""
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not lut.is_cuda or not packed_index_t.is_cuda:
        raise RuntimeError("cfhead_coarse_forward_2bit requires CUDA tensors")
    if lut.dim() != 3 or packed_index_t.dim() != 2:
        raise ValueError(
            f"expected lut [B,N,C] and packed_index_t [N,ceil(V/4)], got {tuple(lut.shape)} and {tuple(packed_index_t.shape)}"
        )

    batch_size, num_tiles, cluster_size = lut.shape
    index_tiles, packed_vocab = packed_index_t.shape
    if index_tiles != num_tiles:
        raise ValueError(f"packed index tiles={index_tiles} does not match lut tiles={num_tiles}")
    if cluster_size > 4:
        raise ValueError("2-bit CFHead kernel requires cluster_size <= 4")

    lut = lut.contiguous()
    packed_index_t = packed_index_t.contiguous()
    out = torch.empty((batch_size, vocab_size), device=lut.device, dtype=torch.float32)

    block_v = 256
    grid = (triton.cdiv(vocab_size, block_v), batch_size)
    _cfhead_coarse_2bit_kernel[grid](
        lut,
        packed_index_t,
        out,
        vocab_size,
        packed_vocab,
        num_tiles,
        cluster_size,
        BLOCK_V=block_v,
        num_warps=8,
    )
    return out


def cfhead_coarse_forward_2bit_parallel(lut: torch.Tensor, packed_index_t: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Parallel 2-bit CFHead coarse path using tile-block partial sums."""
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not lut.is_cuda or not packed_index_t.is_cuda:
        raise RuntimeError("cfhead_coarse_forward_2bit_parallel requires CUDA tensors")
    if lut.dim() != 3 or packed_index_t.dim() != 2:
        raise ValueError(
            f"expected lut [B,N,C] and packed_index_t [N,ceil(V/4)], got {tuple(lut.shape)} and {tuple(packed_index_t.shape)}"
        )

    batch_size, num_tiles, cluster_size = lut.shape
    index_tiles, packed_vocab = packed_index_t.shape
    if index_tiles != num_tiles:
        raise ValueError(f"packed index tiles={index_tiles} does not match lut tiles={num_tiles}")
    if cluster_size > 4:
        raise ValueError("2-bit CFHead kernel requires cluster_size <= 4")

    lut = lut.contiguous()
    packed_index_t = packed_index_t.contiguous()
    out = torch.empty((batch_size, vocab_size), device=lut.device, dtype=torch.float32)

    block_v = 256
    tile_block = 16
    num_tile_blocks = triton.cdiv(num_tiles, tile_block)
    partial = torch.empty(
        (batch_size, num_tile_blocks, vocab_size),
        device=lut.device,
        dtype=torch.float32,
    )

    partial_grid = (triton.cdiv(vocab_size, block_v), batch_size, num_tile_blocks)
    _cfhead_coarse_2bit_partial_kernel[partial_grid](
        lut,
        packed_index_t,
        partial,
        vocab_size,
        packed_vocab,
        num_tiles,
        cluster_size,
        num_tile_blocks,
        TILE_BLOCK=tile_block,
        BLOCK_V=block_v,
        num_warps=8,
    )

    reduce_grid = (triton.cdiv(vocab_size, block_v), batch_size)
    _cfhead_coarse_reduce_kernel[reduce_grid](
        partial,
        out,
        vocab_size,
        num_tile_blocks,
        BLOCK_V=block_v,
        num_warps=8,
    )
    return out


def cfhead_fine_forward(hidden: torch.Tensor, weight: torch.Tensor, candidate_idx: torch.Tensor) -> torch.Tensor:
    """Compute exact candidate logits without materializing selected_weight."""
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not hidden.is_cuda or not weight.is_cuda or not candidate_idx.is_cuda:
        raise RuntimeError("cfhead_fine_forward requires CUDA tensors")
    if hidden.dim() != 2 or weight.dim() != 2 or candidate_idx.dim() != 2:
        raise ValueError(
            f"expected hidden [B,H], weight [V,H], candidate_idx [B,K], got {tuple(hidden.shape)}, {tuple(weight.shape)}, {tuple(candidate_idx.shape)}"
        )

    batch_size, hidden_size = hidden.shape
    cand_batch, top_k = candidate_idx.shape
    if cand_batch != batch_size:
        raise ValueError(f"candidate batch={cand_batch} does not match hidden batch={batch_size}")
    if weight.shape[1] != hidden_size:
        raise ValueError(f"weight hidden={weight.shape[1]} does not match hidden size={hidden_size}")

    hidden = hidden.contiguous()
    weight = weight.contiguous()
    candidate_idx = candidate_idx.contiguous()
    out = torch.empty((batch_size, top_k), device=hidden.device, dtype=torch.float32)

    block_k = 32
    block_h = 64
    grid = (triton.cdiv(top_k, block_k), batch_size)
    _cfhead_fine_kernel[grid](
        hidden,
        weight,
        candidate_idx,
        out,
        top_k,
        hidden_size,
        BLOCK_K=block_k,
        BLOCK_H=block_h,
        num_warps=8,
    )
    return out
