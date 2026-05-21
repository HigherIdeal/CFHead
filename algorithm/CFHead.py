# -*- coding: utf-8 -*-
"""
CFHead.py

Coarse-Fine LM Head selector for the current evaluation framework.

Default configuration:
    tile_size     = 8
    cluster_size  = 64
    top_k         = 256
    kmeans_iters  = 20
    seed          = 99

The cache is reused only when all relevant metadata match:
    vocab size, hidden size, tile size, cluster size, top-k,
    k-means iterations, seed, source dtype, and a lightweight
    fingerprint of the current LM Head weight.

If any setting changes, or if the LM Head weight fingerprint changes,
the codebook and index map are rebuilt automatically.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

try:
    from kernel.cfhead_cuda import (
        CUDA_KERNEL_AVAILABLE,
        cfhead_coarse_cuda,
        cfhead_fine_cuda,
    )
except Exception:
    CUDA_KERNEL_AVAILABLE = False
    cfhead_coarse_cuda = None
    cfhead_fine_cuda = None

try:
    from kernel.cfhead_triton import (
        TRITON_AVAILABLE,
        cfhead_coarse_forward,
        cfhead_coarse_forward_2bit,
        cfhead_coarse_forward_2bit_parallel,
        cfhead_fine_forward,
    )
except Exception:
    TRITON_AVAILABLE = False
    cfhead_coarse_forward = None
    cfhead_coarse_forward_2bit = None
    cfhead_coarse_forward_2bit_parallel = None
    cfhead_fine_forward = None


IMPLEMENTED = True

DEFAULT_TILE_SIZE = 8
DEFAULT_CLUSTER_SIZE = 4
DEFAULT_TOP_K = 8192
DEFAULT_KMEANS_ITERS = 10
DEFAULT_SEED = 99
CACHE_VERSION = 3
COARSE_CHUNK_TILES = 32


try:
    from algorithm.Original import NextTokenResult
except Exception:
    @dataclass
    class NextTokenResult:
        token_ids: torch.Tensor
        logits: torch.Tensor
        lm_head_time_ms: float


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got: {value}") from exc


def build_selector(model) -> "CFHeadSelector":
    """Factory used by eval_common.build_selector()."""
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise RuntimeError("model.get_output_embeddings() returned None")

    return CFHeadSelector(
        lm_head=lm_head,
        tile_size=_env_int("CFHEAD_TILE_SIZE", DEFAULT_TILE_SIZE),
        cluster_size=_env_int("CFHEAD_CLUSTER_SIZE", DEFAULT_CLUSTER_SIZE),
        top_k=_env_int("CFHEAD_TOP_K", DEFAULT_TOP_K),
        kmeans_iters=_env_int("CFHEAD_KMEANS_ITERS", DEFAULT_KMEANS_ITERS),
        seed=_env_int("CFHEAD_SEED", DEFAULT_SEED),
        force_rebuild=os.environ.get("CFHEAD_FORCE_REBUILD", "0").strip() == "1",
        debug=os.environ.get("CFHEAD_DEBUG", "0").strip() == "1",
    )


class CFHeadSelector:
    """Tile-wise clustered coarse-fine LM Head selector.

    Algorithm:
      1. Offline/cache stage:
         W_lm: [V, H]
         split H into N_tile = H / T tiles.
         For each tile t, cluster [V, T] vectors into C centroids.
         Store:
             codebook:  [N_tile, C, T]
             index_map: [V, N_tile] uint8 when C <= 256

      2. Online stage:
         hidden h: [B, H] -> [B, N_tile, T]
         LUT = <h_tile, codebook_tile>: [B, N_tile, C]
         coarse_logit[v] = sum_t LUT[t, index_map[v, t]]
         candidates = top_k(coarse_logit)
         exact logits are computed only for candidate tokens using original W_lm.
    """

    def __init__(
        self,
        lm_head: torch.nn.Module,
        tile_size: int = DEFAULT_TILE_SIZE,
        cluster_size: int = DEFAULT_CLUSTER_SIZE,
        top_k: int = DEFAULT_TOP_K,
        kmeans_iters: int = DEFAULT_KMEANS_ITERS,
        seed: int = DEFAULT_SEED,
        cache_dir: str | Path | None = None,
        force_rebuild: bool = False,
        debug: bool = False,
    ):
        self.lm_head = lm_head
        self.tile_size = int(tile_size)
        self.cluster_size = int(cluster_size)
        self.top_k = int(top_k)
        self.kmeans_iters = int(kmeans_iters)
        self.seed = int(seed)
        self.force_rebuild = bool(force_rebuild)
        self.debug = bool(debug)
        self.use_triton = os.environ.get("CFHEAD_USE_TRITON", "1").strip() != "0" and TRITON_AVAILABLE
        self.use_triton_coarse = (
            self.use_triton
            and os.environ.get("CFHEAD_USE_TRITON_COARSE", "0").strip() == "1"
        )
        self.use_triton_fine = (
            self.use_triton
            and os.environ.get("CFHEAD_USE_TRITON_FINE", "0").strip() == "1"
        )
        self.use_cuda_kernel = (
            os.environ.get("CFHEAD_USE_CUDA_KERNEL", "1").strip() != "0"
            and CUDA_KERNEL_AVAILABLE
        )
        self._runtime_buffers: dict[str, torch.Tensor] = {}

        if self.tile_size <= 0:
            raise ValueError("tile_size must be positive")
        if self.cluster_size <= 1:
            raise ValueError("cluster_size must be larger than 1")
        if self.cluster_size > 256:
            raise ValueError("This implementation stores index_map as uint8, so cluster_size must be <= 256")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")

        root_dir = Path(__file__).resolve().parents[1]
        default_cache_dir = root_dir / "cache" / "cfhead"
        self.cache_dir = Path(cache_dir or os.environ.get("CFHEAD_CACHE_DIR", default_cache_dir))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        weight = self._weight()
        if weight.dim() != 2:
            raise ValueError(f"LM Head weight must be 2-D [V, H], got shape={tuple(weight.shape)}")

        self.vocab_size = int(weight.shape[0])
        self.hidden_size = int(weight.shape[1])

        if self.hidden_size % self.tile_size != 0:
            raise ValueError(
                f"hidden_size={self.hidden_size} must be divisible by tile_size={self.tile_size}"
            )
        if self.top_k > self.vocab_size:
            raise ValueError(f"top_k={self.top_k} exceeds vocab_size={self.vocab_size}")

        self.num_tiles = self.hidden_size // self.tile_size
        self.index_bits = (self.cluster_size - 1).bit_length()

        self.weight_fingerprint = self._fingerprint_weight(weight)
        self.cache_path = self._make_cache_path()

        self.codebook: torch.Tensor
        self.index_map: torch.Tensor
        self.index_t_u8: torch.Tensor
        self.index_t_long: torch.Tensor | None
        self.index_t_packed2: torch.Tensor | None

        self._load_or_build()
        self._prepare_runtime_index()

        self._print_storage_info()

    def _weight(self) -> torch.Tensor:
        weight = getattr(self.lm_head, "weight", None)
        if weight is None:
            raise RuntimeError("lm_head has no .weight")
        return weight.detach()

    def _bias(self) -> torch.Tensor | None:
        bias = getattr(self.lm_head, "bias", None)
        if bias is None:
            return None
        return bias.detach()

    def _prepare_runtime_index(self) -> None:
        self.index_t_u8 = self.index_map.t().contiguous()
        self.index_t_long = None
        self.index_t_packed2 = None
        if not self.use_cuda_kernel:
            if self.use_triton_coarse and self.cluster_size <= 4:
                self.index_t_packed2 = self._pack_index_t_2bit(self.index_t_u8)
            elif not self.use_triton_coarse:
                self.index_t_long = self.index_t_u8.to(dtype=torch.long)

    def _pack_index_t_2bit(self, index_t: torch.Tensor) -> torch.Tensor:
        num_tiles, vocab_size = index_t.shape
        packed_vocab = (vocab_size + 3) // 4
        padded_vocab = packed_vocab * 4
        if padded_vocab != vocab_size:
            padded = torch.zeros((num_tiles, padded_vocab), device=index_t.device, dtype=torch.uint8)
            padded[:, :vocab_size] = index_t
        else:
            padded = index_t

        packed = (
            (padded[:, 0::4] & 3)
            | ((padded[:, 1::4] & 3) << 2)
            | ((padded[:, 2::4] & 3) << 4)
            | ((padded[:, 3::4] & 3) << 6)
        )
        return packed.contiguous()

    def _buffer(self, name: str, shape: tuple[int, ...], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        buffer = self._runtime_buffers.get(name)
        if buffer is None or tuple(buffer.shape) != shape or buffer.device != device or buffer.dtype != dtype:
            buffer = torch.empty(shape, device=device, dtype=dtype)
            self._runtime_buffers[name] = buffer
        return buffer

    def _metadata(self) -> dict[str, Any]:
        weight = self._weight()
        return {
            "version": CACHE_VERSION,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "tile_size": self.tile_size,
            "cluster_size": self.cluster_size,
            "top_k": self.top_k,
            "kmeans_iters": self.kmeans_iters,
            "seed": self.seed,
            "source_dtype": str(weight.dtype),
            "weight_fingerprint": self.weight_fingerprint,
            "index_dtype": "uint8",
        }

    def _make_cache_path(self) -> Path:
        dtype_tag = str(self._weight().dtype).replace("torch.", "")
        filename = (
            f"cfhead_v{CACHE_VERSION}"
            f"_V{self.vocab_size}_H{self.hidden_size}"
            f"_T{self.tile_size}_C{self.cluster_size}_K{self.top_k}"
            f"_iters{self.kmeans_iters}_seed{self.seed}_{dtype_tag}.pt"
        )
        return self.cache_dir / filename

    def _fingerprint_weight(self, weight: torch.Tensor, samples: int = 4096) -> str:
        """Cheap fingerprint to avoid reusing artifacts from a different LM Head.

        This intentionally does not hash the full 500 MiB matrix. It samples
        deterministic positions across the flattened weight.
        """
        flat = weight.detach().reshape(-1)
        total = flat.numel()
        if total == 0:
            return "empty"

        n = min(samples, total)
        if n == 1:
            idx = torch.zeros(1, dtype=torch.long, device=flat.device)
        else:
            # Avoid float-based linspace on large CUDA tensors. With large flattened
            # LM-head sizes, float rounding can produce an out-of-range final index.
            idx = torch.arange(n, dtype=torch.long, device=flat.device)
            idx = idx * (total - 1) // (n - 1)
        sample = flat.index_select(0, idx).float().cpu().contiguous()
        digest = hashlib.sha1(sample.numpy().tobytes()).hexdigest()
        return digest[:16]

    def _load_or_build(self) -> None:
        if not self.force_rebuild and self.cache_path.exists():
            try:
                obj = torch.load(self.cache_path, map_location="cpu")
                meta = obj.get("metadata", {})
                if self._metadata_matches(meta):
                    print(f"[CFHead] loaded cache: {self.cache_path}")
                    self.codebook = obj["codebook"].to(device=self._weight().device, dtype=self._weight().dtype).contiguous()
                    self.index_map = obj["index_map"].to(device=self._weight().device, dtype=torch.uint8).contiguous()
                    return
                print("[CFHead] cache metadata mismatch. Rebuilding codebook/index.")
                print(f"[CFHead] cache: {self.cache_path}")
            except Exception as exc:
                print(f"[CFHead] failed to load cache. Rebuilding. reason={exc}")

        print("[CFHead] clustering LM Head from current weight.")
        print(
            f"[CFHead] config: tile_size={self.tile_size}, cluster_size={self.cluster_size}, "
            f"top_k={self.top_k}, kmeans_iters={self.kmeans_iters}, seed={self.seed}"
        )
        codebook, index_map = self._build_codebook_and_index()
        self.codebook = codebook.to(device=self._weight().device, dtype=self._weight().dtype).contiguous()
        self.index_map = index_map.to(device=self._weight().device, dtype=torch.uint8).contiguous()

        obj = {
            "metadata": self._metadata(),
            "codebook": codebook.cpu().contiguous(),
            "index_map": index_map.cpu().contiguous(),
        }
        torch.save(obj, self.cache_path)
        print(f"[CFHead] saved cache: {self.cache_path}")

    def _metadata_matches(self, meta: dict[str, Any]) -> bool:
        expected = self._metadata()
        keys = [
            "version",
            "vocab_size",
            "hidden_size",
            "tile_size",
            "cluster_size",
            "top_k",
            "kmeans_iters",
            "seed",
            "source_dtype",
            "weight_fingerprint",
            "index_dtype",
        ]
        for key in keys:
            if meta.get(key) != expected.get(key):
                print(f"[CFHead] mismatch: {key}: cache={meta.get(key)} current={expected.get(key)}")
                return False
        return True

    @torch.no_grad()
    def _build_codebook_and_index(self) -> tuple[torch.Tensor, torch.Tensor]:
        weight = self._weight()
        device = weight.device
        dtype = weight.dtype

        V = self.vocab_size
        T = self.tile_size
        C = self.cluster_size
        N = self.num_tiles

        codebook = torch.empty((N, C, T), device=device, dtype=torch.float32)
        index_map = torch.empty((V, N), device=device, dtype=torch.uint8)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)

        for tile_id in range(N):
            start = tile_id * T
            end = start + T
            x = weight[:, start:end].float().contiguous()  # [V, T]

            # Deterministic random row initialization.
            init_idx_cpu = torch.randperm(V, generator=generator)[:C]
            init_idx = init_idx_cpu.to(device=device)
            centroids = x.index_select(0, init_idx).contiguous()  # [C, T]

            assign = None
            for _ in range(self.kmeans_iters):
                # dist: [V, C]. T is small, so cdist is acceptable.
                dist = torch.cdist(x, centroids, p=2)
                assign = torch.argmin(dist, dim=1)

                new_centroids = torch.zeros_like(centroids)
                counts = torch.bincount(assign, minlength=C).to(device=device, dtype=torch.float32)
                new_centroids.index_add_(0, assign, x)

                nonempty = counts > 0
                new_centroids[nonempty] = new_centroids[nonempty] / counts[nonempty].unsqueeze(1)
                new_centroids[~nonempty] = centroids[~nonempty]
                centroids = new_centroids.contiguous()

            if assign is None:
                dist = torch.cdist(x, centroids, p=2)
                assign = torch.argmin(dist, dim=1)

            codebook[tile_id] = centroids
            index_map[:, tile_id] = assign.to(torch.uint8)

            if (tile_id + 1) % max(1, N // 16) == 0 or tile_id == N - 1:
                print(f"[CFHead] clustering tile {tile_id + 1}/{N}")

        return codebook.to(dtype=dtype), index_map

    def _print_storage_info(self) -> None:
        original_bytes = self.vocab_size * self.hidden_size * 2  # BF16/FP16 reference
        codebook_bytes = self.num_tiles * self.cluster_size * self.tile_size * 2
        theoretical_index_bytes = self.vocab_size * self.num_tiles * self.index_bits / 8.0
        theoretical_total = codebook_bytes + theoretical_index_bytes
        runtime_index_bytes = self.vocab_size * self.num_tiles  # uint8

        mib = 1024.0 * 1024.0
        pct = 100.0 * theoretical_total / original_bytes

        print(
            f"[CFHead] theoretical coarse artifact = {theoretical_total / mib:.3f} MiB "
            f"({pct:.3f}% of BF16 LM Head); "
            f"index_bits={self.index_bits}, tile_size={self.tile_size}, "
            f"cluster_size={self.cluster_size}, top_k={self.top_k}"
        )
        print(
            f"[CFHead] runtime index_map uses uint8 for torch.gather = "
            f"{runtime_index_bytes / mib:.3f} MiB"
        )

    @torch.no_grad()
    def select(self, last_hidden: torch.Tensor) -> NextTokenResult:
        if last_hidden.dim() == 1:
            last_hidden = last_hidden.unsqueeze(0)
        if last_hidden.dim() != 2:
            raise ValueError(f"last_hidden must be [B, H], got shape={tuple(last_hidden.shape)}")
        if last_hidden.shape[-1] != self.hidden_size:
            raise ValueError(
                f"last_hidden hidden size mismatch: got {last_hidden.shape[-1]}, expected {self.hidden_size}"
            )

        device = last_hidden.device
        if self.codebook.device != device:
            self.codebook = self.codebook.to(device=device, dtype=last_hidden.dtype).contiguous()
        if self.index_map.device != device:
            self.index_map = self.index_map.to(device=device, dtype=torch.uint8).contiguous()
            self._prepare_runtime_index()
        elif not hasattr(self, "index_t_u8") or self.index_t_u8.device != device:
            self._prepare_runtime_index()

        if last_hidden.is_cuda:
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        token_ids, returned_logits, debug_info = self._select_impl(last_hidden)

        if last_hidden.is_cuda:
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if self.debug and debug_info is not None:
            print(
                f"[CFHead DEBUG] hit={debug_info['hit']} "
                f"orig_top1={debug_info['orig_top1']} cf_token={debug_info['cf_token']}"
            )

        return NextTokenResult(
            token_ids=token_ids,
            logits=returned_logits,
            lm_head_time_ms=elapsed_ms,
        )

    def _select_impl(self, last_hidden: torch.Tensor):
        B = last_hidden.shape[0]
        V = self.vocab_size
        N = self.num_tiles
        T = self.tile_size

        hidden = last_hidden.to(dtype=self.codebook.dtype)
        hidden_tiles = hidden.reshape(B, N, T)

        # LUT: [B, N, C]
        # LUT[b, tile, c] = dot(hidden_tile[b, tile], codebook[tile, c])
        lut = torch.einsum("bnt,nct->bnc", hidden_tiles.float(), self.codebook.float())

        if self.use_cuda_kernel and last_hidden.is_cuda:
            coarse = cfhead_coarse_cuda(lut, self.index_t_u8)
        elif self.use_triton_coarse and last_hidden.is_cuda and self.index_t_packed2 is not None:
            coarse = cfhead_coarse_forward_2bit_parallel(lut, self.index_t_packed2, V)
        elif self.use_triton_coarse and last_hidden.is_cuda:
            coarse = cfhead_coarse_forward(lut, self.index_t_u8)
        else:
            coarse = self._buffer("coarse", (B, V), last_hidden.device, torch.float32)
            coarse.zero_()

            # coarse[b, v] = sum_tile LUT[b, tile, index_map[v, tile]]
            # Runtime uses [tile, vocab] long indices to avoid strided column reads
            # and repeated uint8->long conversions on every generated token.
            for start in range(0, N, COARSE_CHUNK_TILES):
                end = min(start + COARSE_CHUNK_TILES, N)
                chunk_tiles = end - start
                if self.index_t_long is None:
                    self.index_t_long = self.index_t_u8.to(dtype=torch.long)
                idx = self.index_t_long[start:end].unsqueeze(0).expand(B, chunk_tiles, V)
                vals = self._buffer(
                    "coarse_vals",
                    (B, COARSE_CHUNK_TILES, V),
                    last_hidden.device,
                    lut.dtype,
                )[:, :chunk_tiles, :]
                torch.gather(lut[:, start:end, :], dim=2, index=idx, out=vals)
                chunk_sum = self._buffer("coarse_chunk_sum", (B, V), last_hidden.device, torch.float32)
                torch.sum(vals, dim=1, out=chunk_sum)
                coarse.add_(chunk_sum)

        bias = self._bias()
        if bias is not None:
            coarse.add_(bias.to(device=coarse.device, dtype=coarse.dtype).unsqueeze(0))

        topk = min(self.top_k, V)
        topk_values = self._buffer("topk_values", (B, topk), last_hidden.device, coarse.dtype)
        candidate_idx = self._buffer("candidate_idx", (B, topk), last_hidden.device, torch.long)
        torch.topk(coarse, k=topk, dim=-1, largest=True, sorted=False, out=(topk_values, candidate_idx))

        weight = self._weight().to(device=last_hidden.device, dtype=hidden.dtype)
        if self.use_cuda_kernel and last_hidden.is_cuda:
            fine_logits = cfhead_fine_cuda(hidden, weight, candidate_idx)
        elif self.use_triton_fine and last_hidden.is_cuda:
            fine_logits = cfhead_fine_forward(hidden, weight, candidate_idx)
        else:
            flat_candidate_idx = candidate_idx.reshape(-1)
            selected_weight_flat = self._buffer(
                "selected_weight_flat",
                (B * topk, self.hidden_size),
                last_hidden.device,
                hidden.dtype,
            )
            torch.index_select(weight, dim=0, index=flat_candidate_idx, out=selected_weight_flat)
            selected_weight = selected_weight_flat.reshape(B, topk, self.hidden_size)
            fine_logits_3d = self._buffer("fine_logits_3d", (B, topk, 1), last_hidden.device, hidden.dtype)
            torch.bmm(selected_weight, hidden.unsqueeze(-1), out=fine_logits_3d)
            fine_logits = fine_logits_3d.squeeze(-1)
        if bias is not None:
            selected_bias = bias.to(device=last_hidden.device, dtype=fine_logits.dtype)[candidate_idx]
            fine_logits = fine_logits + selected_bias

        max_fine = torch.max(fine_logits, dim=-1, keepdim=True).values
        tie_candidates = candidate_idx.masked_fill(fine_logits != max_fine, self.vocab_size)
        token_ids = torch.min(tie_candidates, dim=-1).values

        debug_info = None
        if self.debug:
            # Debug path only. This computes the original full LM Head and should
            # not be used for timing experiments.
            original_logits = self.lm_head(last_hidden)
            orig_top1 = torch.argmax(original_logits, dim=-1)
            hit = (candidate_idx == orig_top1.unsqueeze(1)).any(dim=1)
            debug_info = {
                "hit": hit.detach().cpu().tolist(),
                "orig_top1": orig_top1.detach().cpu().tolist(),
                "cf_token": token_ids.detach().cpu().tolist(),
            }

        # Returning coarse logits keeps the interface shape [B, V].
        # The selected token is determined by exact fine-stage logits.
        return token_ids, coarse, debug_info
