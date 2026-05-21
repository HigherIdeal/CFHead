# -*- coding: utf-8 -*-
"""
FlashHead.py

FlashHead-like cluster-level retrieval selector for the current evaluation framework.

Default configuration matched to CFHead(T=8, C=4, top_k=2048):
    num_clusters = 2004
    num_probes   = 32
    tokens/cluster = 64 for LLaMA3.2-1B vocab 128256
    candidate tokens = 32 * 64 = 2048

Interface:
    eval_common.py imports algorithm.FlashHead and calls build_selector(model).
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


IMPLEMENTED = True

DEFAULT_NUM_CLUSTERS = 8016
DEFAULT_NUM_PROBES = 512
DEFAULT_CLUSTER_ITERS = 5
DEFAULT_SEED = 99
DEFAULT_ASSIGN_CHUNK = 512
DEFAULT_TOP_CHOICES = 16
CACHE_VERSION = 2


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


def build_selector(model) -> "FlashHeadSelector":
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise RuntimeError("model.get_output_embeddings() returned None")

    return FlashHeadSelector(
        lm_head=lm_head,
        num_clusters=_env_int("FLASHHEAD_NUM_CLUSTERS", DEFAULT_NUM_CLUSTERS),
        num_probes=_env_int("FLASHHEAD_NUM_PROBES", DEFAULT_NUM_PROBES),
        cluster_iters=_env_int("FLASHHEAD_CLUSTER_ITERS", DEFAULT_CLUSTER_ITERS),
        seed=_env_int("FLASHHEAD_SEED", DEFAULT_SEED),
        assign_chunk=_env_int("FLASHHEAD_ASSIGN_CHUNK", DEFAULT_ASSIGN_CHUNK),
        top_choices=_env_int("FLASHHEAD_TOP_CHOICES", DEFAULT_TOP_CHOICES),
        force_rebuild=os.environ.get("FLASHHEAD_FORCE_REBUILD", "0").strip() == "1",
        debug=os.environ.get("FLASHHEAD_DEBUG", "0").strip() == "1",
    )


class FlashHeadSelector:
    """FlashHead-like cluster-level retrieval LM Head selector.

    Offline/cache:
        W_lm: [V, H]
        Cluster token rows into c balanced clusters.
        Store:
            centroids: [c, H]
            c2t:       [c, b] token ids

    Online:
        cluster_scores = hidden @ centroids.T
        top-p clusters are selected.
        Candidate tokens are union of selected clusters.
        Exact LM Head GEMV is performed only for candidate tokens.
    """

    def __init__(
        self,
        lm_head: torch.nn.Module,
        num_clusters: int = DEFAULT_NUM_CLUSTERS,
        num_probes: int = DEFAULT_NUM_PROBES,
        cluster_iters: int = DEFAULT_CLUSTER_ITERS,
        seed: int = DEFAULT_SEED,
        assign_chunk: int = DEFAULT_ASSIGN_CHUNK,
        top_choices: int = DEFAULT_TOP_CHOICES,
        cache_dir: str | Path | None = None,
        force_rebuild: bool = False,
        debug: bool = False,
    ):
        self.lm_head = lm_head
        self.num_clusters = int(num_clusters)
        self.num_probes = int(num_probes)
        self.cluster_iters = int(cluster_iters)
        self.seed = int(seed)
        self.assign_chunk = int(assign_chunk)
        self.top_choices = int(top_choices)
        self.force_rebuild = bool(force_rebuild)
        self.debug = bool(debug)

        if self.num_clusters <= 1:
            raise ValueError("num_clusters must be larger than 1")
        if self.num_probes <= 0:
            raise ValueError("num_probes must be positive")
        if self.num_probes > self.num_clusters:
            raise ValueError("num_probes must be <= num_clusters")
        if self.cluster_iters <= 0:
            raise ValueError("cluster_iters must be positive")
        if self.assign_chunk <= 0:
            raise ValueError("assign_chunk must be positive")
        if self.top_choices <= 0:
            raise ValueError("top_choices must be positive")

        root_dir = Path(__file__).resolve().parents[1]
        default_cache_dir = root_dir / "cache" / "flashhead"
        self.cache_dir = Path(cache_dir or os.environ.get("FLASHHEAD_CACHE_DIR", default_cache_dir))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        weight = self._weight()
        if weight.dim() != 2:
            raise ValueError(f"LM Head weight must be [V, H], got shape={tuple(weight.shape)}")

        self.vocab_size = int(weight.shape[0])
        self.hidden_size = int(weight.shape[1])

        self.tokens_per_cluster = (self.vocab_size + self.num_clusters - 1) // self.num_clusters
        self.candidate_tokens = self.num_probes * self.tokens_per_cluster

        self.weight_fingerprint = self._fingerprint_weight(weight)
        self.cache_path = self._make_cache_path()

        self.centroids: torch.Tensor
        self.c2t: torch.Tensor
        self.valid_mask: torch.Tensor

        self._load_or_build()
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

    def _metadata(self) -> dict[str, Any]:
        weight = self._weight()
        return {
            "version": CACHE_VERSION,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_clusters": self.num_clusters,
            "num_probes": self.num_probes,
            "tokens_per_cluster": self.tokens_per_cluster,
            "candidate_tokens": self.candidate_tokens,
            "cluster_iters": self.cluster_iters,
            "seed": self.seed,
            "assign_chunk": self.assign_chunk,
            "top_choices": self.top_choices,
            "source_dtype": str(weight.dtype),
            "weight_fingerprint": self.weight_fingerprint,
        }

    def _make_cache_path(self) -> Path:
        dtype_tag = str(self._weight().dtype).replace("torch.", "")
        filename = (
            f"flashhead_v{CACHE_VERSION}"
            f"_V{self.vocab_size}_H{self.hidden_size}"
            f"_C{self.num_clusters}_P{self.num_probes}_B{self.tokens_per_cluster}"
            f"_iters{self.cluster_iters}_seed{self.seed}_{dtype_tag}.pt"
        )
        return self.cache_dir / filename

    def _fingerprint_weight(self, weight: torch.Tensor, samples: int = 4096) -> str:
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

    def _metadata_matches(self, meta: dict[str, Any]) -> bool:
        expected = self._metadata()
        keys = [
            "version",
            "vocab_size",
            "hidden_size",
            "num_clusters",
            "num_probes",
            "tokens_per_cluster",
            "candidate_tokens",
            "cluster_iters",
            "seed",
            "assign_chunk",
            "top_choices",
            "source_dtype",
            "weight_fingerprint",
        ]
        for key in keys:
            if meta.get(key) != expected.get(key):
                print(f"[FlashHead] mismatch: {key}: cache={meta.get(key)} current={expected.get(key)}")
                return False
        return True

    def _load_or_build(self) -> None:
        if not self.force_rebuild and self.cache_path.exists():
            try:
                obj = torch.load(self.cache_path, map_location="cpu")
                meta = obj.get("metadata", {})
                if self._metadata_matches(meta):
                    print(f"[FlashHead] loaded cache: {self.cache_path}")
                    device = self._weight().device
                    dtype = self._weight().dtype
                    self.centroids = obj["centroids"].to(device=device, dtype=dtype).contiguous()
                    self.c2t = obj["c2t"].to(device=device, dtype=torch.long).contiguous()
                    self.valid_mask = obj["valid_mask"].to(device=device, dtype=torch.bool).contiguous()
                    return
                print("[FlashHead] cache metadata mismatch. Rebuilding centroids/C2T.")
            except Exception as exc:
                print(f"[FlashHead] failed to load cache. Rebuilding. reason={exc}")

        print("[FlashHead] clustering LM Head from current weight.")
        print(
            f"[FlashHead] config: c={self.num_clusters}, p={self.num_probes}, "
            f"tokens_per_cluster={self.tokens_per_cluster}, candidate_tokens={self.candidate_tokens}, "
            f"cluster_iters={self.cluster_iters}, seed={self.seed}"
        )
        centroids, c2t, valid_mask = self._build_centroids_and_c2t()
        device = self._weight().device
        dtype = self._weight().dtype
        self.centroids = centroids.to(device=device, dtype=dtype).contiguous()
        self.c2t = c2t.to(device=device, dtype=torch.long).contiguous()
        self.valid_mask = valid_mask.to(device=device, dtype=torch.bool).contiguous()

        obj = {
            "metadata": self._metadata(),
            "centroids": centroids.cpu().contiguous(),
            "c2t": c2t.cpu().contiguous(),
            "valid_mask": valid_mask.cpu().contiguous(),
        }
        torch.save(obj, self.cache_path)
        print(f"[FlashHead] saved cache: {self.cache_path}")

    @torch.no_grad()
    def _build_centroids_and_c2t(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Approximate balanced spherical k-means.

        This implementation is designed to be self-contained.
        It performs capacity-constrained assignment using top choices per row.
        """
        weight = self._weight()
        device = weight.device
        V = self.vocab_size
        H = self.hidden_size
        C = self.num_clusters
        B = self.tokens_per_cluster

        x = F.normalize(weight.float(), dim=1).contiguous()

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)

        # Balanced random initialization.
        perm = torch.randperm(V, generator=generator)
        assignment = torch.empty((V,), dtype=torch.long, device="cpu")
        for pos, token_id in enumerate(perm.tolist()):
            cluster_id = min(pos // B, C - 1)
            assignment[token_id] = cluster_id

        centroids = torch.empty((C, H), device=device, dtype=torch.float32)

        for it in range(self.cluster_iters):
            centroids.zero_()
            assign_dev = assignment.to(device=device)
            centroids.index_add_(0, assign_dev, x)
            counts = torch.bincount(assign_dev, minlength=C).to(device=device, dtype=torch.float32).clamp_min_(1.0)
            centroids = centroids / counts.unsqueeze(1)
            centroids = F.normalize(centroids, dim=1).contiguous()

            if it < self.cluster_iters - 1:
                assignment = self._balanced_assign(x, centroids, B, seed=self.seed + it + 1)

            print(f"[FlashHead] clustering iteration {it + 1}/{self.cluster_iters}")

        c2t = torch.full((C, B), -1, dtype=torch.long, device="cpu")
        valid_mask = torch.zeros((C, B), dtype=torch.bool, device="cpu")
        fill = [0 for _ in range(C)]

        for token_id, cluster_id in enumerate(assignment.tolist()):
            pos = fill[cluster_id]
            if pos < B:
                c2t[cluster_id, pos] = token_id
                valid_mask[cluster_id, pos] = True
                fill[cluster_id] += 1

        # If numerical/assignment edge cases leave holes, fill them with nearest unfilled tokens.
        # This rarely triggers when V is divisible by C*B, but keeps the structure robust.
        missing = int((~valid_mask).sum().item())
        if missing > 0:
            print(f"[FlashHead] warning: filling {missing} empty C2T slots by token 0 padding")
            c2t[~valid_mask] = 0

        return centroids.to(dtype=weight.dtype), c2t, valid_mask

    @torch.no_grad()
    def _balanced_assign(self, x: torch.Tensor, centroids: torch.Tensor, capacity: int, seed: int) -> torch.Tensor:
        V = x.shape[0]
        C = centroids.shape[0]
        top_m = min(self.top_choices, C)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        order = torch.randperm(V, generator=generator)

        capacities = [capacity for _ in range(C)]
        assignment = torch.empty((V,), dtype=torch.long, device="cpu")

        for start in range(0, V, self.assign_chunk):
            token_cpu = order[start:start + self.assign_chunk]
            token_dev = token_cpu.to(device=x.device)
            scores = x.index_select(0, token_dev) @ centroids.t()
            choices = torch.topk(scores, k=top_m, dim=1).indices.cpu()

            for row, token_id in enumerate(token_cpu.tolist()):
                chosen = -1
                for c in choices[row].tolist():
                    if capacities[c] > 0:
                        chosen = c
                        break
                if chosen < 0:
                    # Fallback: choose cluster with the largest remaining capacity.
                    chosen = max(range(C), key=lambda cid: capacities[cid])
                assignment[token_id] = chosen
                capacities[chosen] -= 1

        return assignment

    def _print_storage_info(self) -> None:
        original_bytes = self.vocab_size * self.hidden_size * 2
        centroid_bytes = self.num_clusters * self.hidden_size * 2
        c2t_bytes_int32 = self.num_clusters * self.tokens_per_cluster * 4
        artifact_bytes = centroid_bytes + c2t_bytes_int32
        fine_bytes = self.candidate_tokens * self.hidden_size * 2

        mib = 1024.0 * 1024.0
        print(
            f"[FlashHead] artifact = {artifact_bytes / mib:.3f} MiB "
            f"({100.0 * artifact_bytes / original_bytes:.3f}% of BF16 LM Head); "
            f"centroid={centroid_bytes / mib:.3f} MiB, C2T(int32)={c2t_bytes_int32 / mib:.3f} MiB"
        )
        print(
            f"[FlashHead] candidate_tokens={self.candidate_tokens} "
            f"(p={self.num_probes} clusters x b={self.tokens_per_cluster} tokens), "
            f"fine weight read={fine_bytes / mib:.3f} MiB"
        )

    @torch.no_grad()
    def select(self, last_hidden: torch.Tensor) -> NextTokenResult:
        if last_hidden.dim() == 1:
            last_hidden = last_hidden.unsqueeze(0)
        if last_hidden.dim() != 2:
            raise ValueError(f"last_hidden must be [B, H], got shape={tuple(last_hidden.shape)}")
        if last_hidden.shape[-1] != self.hidden_size:
            raise ValueError(
                f"hidden size mismatch: got {last_hidden.shape[-1]}, expected {self.hidden_size}"
            )

        device = last_hidden.device
        dtype = last_hidden.dtype
        if self.centroids.device != device:
            self.centroids = self.centroids.to(device=device, dtype=dtype).contiguous()
        if self.c2t.device != device:
            self.c2t = self.c2t.to(device=device, dtype=torch.long).contiguous()
        if self.valid_mask.device != device:
            self.valid_mask = self.valid_mask.to(device=device, dtype=torch.bool).contiguous()

        if last_hidden.is_cuda:
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        token_ids, cluster_scores, debug_info = self._select_impl(last_hidden)

        if last_hidden.is_cuda:
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if self.debug and debug_info is not None:
            print(
                f"[FlashHead DEBUG] hit={debug_info['hit']} "
                f"orig_top1={debug_info['orig_top1']} flash_token={debug_info['flash_token']}"
            )

        return NextTokenResult(
            token_ids=token_ids,
            logits=cluster_scores,
            lm_head_time_ms=elapsed_ms,
        )

    def _select_impl(self, last_hidden: torch.Tensor):
        hidden = last_hidden.to(dtype=self.centroids.dtype)
        B = hidden.shape[0]

        # Stage 1: cluster-level retrieval.
        cluster_scores = hidden.float() @ self.centroids.float().t()  # [B, C]
        selected_clusters = torch.topk(cluster_scores, k=self.num_probes, dim=-1).indices  # [B, p]

        # Expand selected clusters to candidate token ids.
        candidate = self.c2t[selected_clusters]  # [B, p, b]
        valid = self.valid_mask[selected_clusters]  # [B, p, b]
        candidate = candidate.reshape(B, -1)
        valid = valid.reshape(B, -1)

        candidate_safe = candidate.clamp_min(0)

        # Stage 2: exact LM Head only for selected candidate tokens.
        weight = self._weight().to(device=last_hidden.device, dtype=hidden.dtype)
        selected_weight = weight[candidate_safe]  # [B, p*b, H]
        fine_logits = torch.einsum("bh,bkh->bk", hidden, selected_weight)

        bias = self._bias()
        if bias is not None:
            selected_bias = bias.to(device=last_hidden.device, dtype=fine_logits.dtype)[candidate_safe]
            fine_logits = fine_logits + selected_bias

        fine_logits = fine_logits.masked_fill(~valid, float("-inf"))

        best_pos = torch.argmax(fine_logits, dim=-1)
        token_ids = candidate_safe.gather(dim=1, index=best_pos.unsqueeze(1)).squeeze(1)

        debug_info = None
        if self.debug:
            original_logits = self.lm_head(last_hidden)
            orig_top1 = torch.argmax(original_logits, dim=-1)
            hit = (candidate_safe == orig_top1.unsqueeze(1)).logical_and(valid).any(dim=1)
            debug_info = {
                "hit": hit.detach().cpu().tolist(),
                "orig_top1": orig_top1.detach().cpu().tolist(),
                "flash_token": token_ids.detach().cpu().tolist(),
            }

        return token_ids, cluster_scores, debug_info
