# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

IMPLEMENTED = True

DEFAULT_RETAINED_VOCAB_SIZE = 8192
DEFAULT_SEED = 99
DEFAULT_FALLBACK_MODE = "low_id"
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
    return int(value)


def build_selector(model) -> "VocabTrimSelector":
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise RuntimeError("model.get_output_embeddings() returned None")
    return VocabTrimSelector(
        lm_head=lm_head,
        retained_vocab_size=_env_int("VOCABTRIM_SIZE", DEFAULT_RETAINED_VOCAB_SIZE),
        seed=_env_int("VOCABTRIM_SEED", DEFAULT_SEED),
        freq_path=os.environ.get("VOCABTRIM_FREQ_PATH", "").strip() or None,
        fallback_mode=os.environ.get("VOCABTRIM_FALLBACK", DEFAULT_FALLBACK_MODE).strip() or DEFAULT_FALLBACK_MODE,
        force_rebuild=os.environ.get("VOCABTRIM_FORCE_REBUILD", "0").strip() == "1",
        debug=os.environ.get("VOCABTRIM_DEBUG", "0").strip() == "1",
    )


class VocabTrimSelector:
    """VocabTrim-style retained-vocabulary LM Head.

    S is a retained vocabulary set. The selector computes only:
        logits_S = h @ W[S, :].T
    and maps the local argmax back to the original token id:
        token_id = S[argmax(logits_S)]

    If VOCABTRIM_FREQ_PATH is provided, S is built from target-token frequency.
    Otherwise, S is built by a deterministic fallback prior.
    """

    def __init__(
        self,
        lm_head: torch.nn.Module,
        retained_vocab_size: int = DEFAULT_RETAINED_VOCAB_SIZE,
        seed: int = DEFAULT_SEED,
        freq_path: str | Path | None = None,
        fallback_mode: str = DEFAULT_FALLBACK_MODE,
        cache_dir: str | Path | None = None,
        force_rebuild: bool = False,
        debug: bool = False,
    ):
        self.lm_head = lm_head
        self.retained_vocab_size = int(retained_vocab_size)
        self.seed = int(seed)
        self.freq_path = Path(freq_path).expanduser().resolve() if freq_path else None
        self.fallback_mode = str(fallback_mode)
        self.force_rebuild = bool(force_rebuild)
        self.debug = bool(debug)

        if self.retained_vocab_size <= 0:
            raise ValueError("retained_vocab_size must be positive")

        root_dir = Path(__file__).resolve().parents[1]
        default_cache_dir = root_dir / "cache" / "vocabtrim"
        self.cache_dir = Path(cache_dir or os.environ.get("VOCABTRIM_CACHE_DIR", default_cache_dir))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        weight = self._weight()
        if weight.dim() != 2:
            raise ValueError(f"LM Head weight must be [V, H], got {tuple(weight.shape)}")

        self.vocab_size = int(weight.shape[0])
        self.hidden_size = int(weight.shape[1])
        if self.retained_vocab_size > self.vocab_size:
            raise ValueError("retained_vocab_size exceeds vocab_size")

        self.weight_fingerprint = self._fingerprint_weight(weight)
        self.freq_fingerprint = self._fingerprint_file(self.freq_path) if self.freq_path else "none"
        self.cache_path = self._make_cache_path()

        self.retained_ids: torch.Tensor
        self.trimmed_weight: torch.Tensor
        self.trimmed_bias: torch.Tensor | None

        self._load_or_build()
        self._materialize_trimmed_head()
        self._print_storage_info()

    def _weight(self) -> torch.Tensor:
        weight = getattr(self.lm_head, "weight", None)
        if weight is None:
            raise RuntimeError("lm_head has no .weight")
        return weight.detach()

    def _bias(self) -> torch.Tensor | None:
        bias = getattr(self.lm_head, "bias", None)
        return None if bias is None else bias.detach()

    def _metadata(self) -> dict[str, Any]:
        weight = self._weight()
        return {
            "version": CACHE_VERSION,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "retained_vocab_size": self.retained_vocab_size,
            "seed": self.seed,
            "fallback_mode": self.fallback_mode,
            "freq_path": str(self.freq_path) if self.freq_path else "",
            "freq_fingerprint": self.freq_fingerprint,
            "source_dtype": str(weight.dtype),
            "weight_fingerprint": self.weight_fingerprint,
        }

    def _make_cache_path(self) -> Path:
        dtype_tag = str(self._weight().dtype).replace("torch.", "")
        mode_tag = self.fallback_mode.replace("/", "_")
        return self.cache_dir / (
            f"vocabtrim_v{CACHE_VERSION}_V{self.vocab_size}_H{self.hidden_size}"
            f"_M{self.retained_vocab_size}_mode{mode_tag}_seed{self.seed}"
            f"_{dtype_tag}_freq{self.freq_fingerprint[:8]}.pt"
        )

    def _fingerprint_weight(self, weight: torch.Tensor, samples: int = 4096) -> str:
        flat = weight.detach().reshape(-1)
        total = flat.numel()
        if total == 0:
            return "empty"
        n = min(samples, total)
        if n == 1:
            idx = torch.zeros(1, dtype=torch.long, device=flat.device)
        else:
            # Float linspace can round the last CUDA index past the flattened
            # tensor size for large LM heads. Use integer arithmetic instead.
            idx = torch.arange(n, dtype=torch.long, device=flat.device)
            idx = idx * (total - 1) // (n - 1)
        sample = flat.index_select(0, idx).float().cpu().contiguous()
        return hashlib.sha1(sample.numpy().tobytes()).hexdigest()[:16]

    def _fingerprint_file(self, path: Path | None) -> str:
        if path is None:
            return "none"
        if not path.exists():
            raise FileNotFoundError(f"VOCABTRIM_FREQ_PATH not found: {path}")
        h = hashlib.sha1()
        with path.open("rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()[:16]

    def _metadata_matches(self, meta: dict[str, Any]) -> bool:
        expected = self._metadata()
        for key, value in expected.items():
            if meta.get(key) != value:
                print(f"[VocabTrim] mismatch: {key}: cache={meta.get(key)} current={value}")
                return False
        return True

    def _load_or_build(self) -> None:
        if not self.force_rebuild and self.cache_path.exists():
            try:
                obj = torch.load(self.cache_path, map_location="cpu")
                if self._metadata_matches(obj.get("metadata", {})):
                    print(f"[VocabTrim] loaded cache: {self.cache_path}")
                    self.retained_ids = obj["retained_ids"].to(device=self._weight().device, dtype=torch.long).contiguous()
                    return
                print("[VocabTrim] cache metadata mismatch. Rebuilding retained vocabulary.")
            except Exception as exc:
                print(f"[VocabTrim] failed to load cache. Rebuilding. reason={exc}")

        self.retained_ids = self._build_retained_ids().to(device=self._weight().device, dtype=torch.long).contiguous()
        torch.save({"metadata": self._metadata(), "retained_ids": self.retained_ids.cpu().contiguous()}, self.cache_path)
        print(f"[VocabTrim] saved cache: {self.cache_path}")

    @torch.no_grad()
    def _build_retained_ids(self) -> torch.Tensor:
        M = self.retained_vocab_size
        if self.freq_path is not None:
            print(
                f"[VocabTrim] config: retained_vocab={M}, fallback={self.fallback_mode}, "
                f"seed={self.seed}"
            )
            print(f"[VocabTrim] building retained vocabulary from frequency file: {self.freq_path}")
            ranked_ids = self._load_frequency_rank(self.freq_path)
            retained = self._take_valid_unique(ranked_ids, M)
            if len(retained) < M:
                print(f"[VocabTrim] warning: frequency file has only {len(retained)} valid ids. Filling by fallback.")
                seen = set(retained)
                for token_id in self._fallback_rank():
                    if token_id not in seen:
                        retained.append(token_id)
                        seen.add(token_id)
                    if len(retained) >= M:
                        break
            return torch.tensor(retained[:M], dtype=torch.long)

        print(
            f"[VocabTrim] config: retained_vocab={M}, fallback={self.fallback_mode}, "
            f"seed={self.seed}"
        )
        print(f"[VocabTrim] no VOCABTRIM_FREQ_PATH. Using fallback S: {self.fallback_mode}")
        return torch.tensor(self._fallback_rank()[:M], dtype=torch.long)

    def _take_valid_unique(self, ids: list[int], limit: int) -> list[int]:
        retained, seen = [], set()
        for token_id in ids:
            token_id = int(token_id)
            if 0 <= token_id < self.vocab_size and token_id not in seen:
                retained.append(token_id)
                seen.add(token_id)
            if len(retained) >= limit:
                break
        return retained

    @torch.no_grad()
    def _fallback_rank(self) -> list[int]:
        mode = self.fallback_mode.lower()
        if mode == "low_id":
            return list(range(self.vocab_size))
        if mode == "random":
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.seed)
            return torch.randperm(self.vocab_size, generator=gen).tolist()
        if mode == "row_norm":
            norms = torch.linalg.vector_norm(self._weight().float(), ord=2, dim=1)
            return torch.argsort(norms, descending=True).cpu().tolist()
        raise ValueError("VOCABTRIM_FALLBACK must be low_id, row_norm, or random")

    def _load_frequency_rank(self, path: Path) -> list[int]:
        suffix = path.suffix.lower()
        if suffix in {".pt", ".pth"}:
            return self._rank_from_object(torch.load(path, map_location="cpu"))
        if suffix == ".json":
            return self._rank_from_object(json.loads(path.read_text(encoding="utf-8")))
        if suffix == ".jsonl":
            pairs = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        token = row.get("token_id", row.get("token", row.get("id")))
                        count = row.get("count", row.get("freq", row.get("frequency", 1)))
                        if token is not None:
                            pairs.append((int(token), float(count)))
            pairs.sort(key=lambda x: (-x[1], x[0]))
            return [t for t, _ in pairs]
        if suffix == ".csv":
            pairs = []
            with path.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    token = row.get("token_id", row.get("token", row.get("id")))
                    count = row.get("count", row.get("freq", row.get("frequency", 1)))
                    if token is not None:
                        pairs.append((int(token), float(count)))
            pairs.sort(key=lambda x: (-x[1], x[0]))
            return [t for t, _ in pairs]

        ids = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    ids.append(int(line.strip().split()[0]))
        return ids

    def _rank_from_object(self, obj: Any) -> list[int]:
        if isinstance(obj, torch.Tensor):
            if obj.dim() != 1:
                raise ValueError("Tensor object must be a 1-D token id list")
            return [int(x) for x in obj.tolist()]
        if isinstance(obj, (list, tuple)):
            return [int(x) for x in obj]
        if isinstance(obj, dict):
            for key in ("token_ids", "tokens"):
                if key in obj:
                    value = obj[key]
                    if isinstance(value, torch.Tensor):
                        value = value.tolist()
                    return [int(x) for x in value]
            pairs = []
            for k, v in obj.items():
                try:
                    pairs.append((int(k), float(v)))
                except Exception:
                    pass
            if not pairs:
                raise ValueError("Could not parse token frequency dictionary")
            pairs.sort(key=lambda x: (-x[1], x[0]))
            return [t for t, _ in pairs]
        raise TypeError(f"Unsupported frequency object type: {type(obj)}")

    def _materialize_trimmed_head(self) -> None:
        weight = self._weight()
        ids = self.retained_ids.to(device=weight.device, dtype=torch.long)
        self.trimmed_weight = weight.index_select(0, ids).contiguous()
        bias = self._bias()
        self.trimmed_bias = None
        if bias is not None:
            self.trimmed_bias = bias.index_select(0, ids.to(device=bias.device)).contiguous()

    def _print_storage_info(self) -> None:
        original_bytes = self.vocab_size * self.hidden_size * 2
        trimmed_weight_bytes = self.retained_vocab_size * self.hidden_size * 2
        id_map_bytes = self.retained_vocab_size * 4
        total = trimmed_weight_bytes + id_map_bytes
        mib = 1024.0 * 1024.0
        print(f"[VocabTrim] retained_vocab={self.retained_vocab_size}/{self.vocab_size} ({100.0*self.retained_vocab_size/self.vocab_size:.3f}% of vocab)")
        print(f"[VocabTrim] artifact/read = {total/mib:.3f} MiB ({100.0*total/original_bytes:.3f}% of BF16 LM Head); trimmed_weight={trimmed_weight_bytes/mib:.3f} MiB, id_map={id_map_bytes/mib:.3f} MiB")

    @torch.no_grad()
    def select(self, last_hidden: torch.Tensor) -> NextTokenResult:
        if last_hidden.dim() == 1:
            last_hidden = last_hidden.unsqueeze(0)
        if last_hidden.dim() != 2:
            raise ValueError(f"last_hidden must be [B, H], got {tuple(last_hidden.shape)}")
        if last_hidden.shape[-1] != self.hidden_size:
            raise ValueError(f"hidden size mismatch: got {last_hidden.shape[-1]}, expected {self.hidden_size}")

        device = last_hidden.device
        dtype = last_hidden.dtype

        if self.retained_ids.device != device:
            self.retained_ids = self.retained_ids.to(device=device, dtype=torch.long).contiguous()
        if self.trimmed_weight.device != device or self.trimmed_weight.dtype != dtype:
            self.trimmed_weight = self.trimmed_weight.to(device=device, dtype=dtype).contiguous()
        if self.trimmed_bias is not None and (self.trimmed_bias.device != device or self.trimmed_bias.dtype != dtype):
            self.trimmed_bias = self.trimmed_bias.to(device=device, dtype=dtype).contiguous()

        if last_hidden.is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        logits_s = last_hidden.to(dtype=self.trimmed_weight.dtype) @ self.trimmed_weight.t()
        if self.trimmed_bias is not None:
            logits_s = logits_s + self.trimmed_bias.unsqueeze(0)

        local_ids = torch.argmax(logits_s, dim=-1)
        token_ids = self.retained_ids.gather(dim=0, index=local_ids)

        if last_hidden.is_cuda:
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if self.debug:
            original_logits = self.lm_head(last_hidden)
            orig_top1 = torch.argmax(original_logits, dim=-1)
            hit = (orig_top1.unsqueeze(1) == self.retained_ids.unsqueeze(0)).any(dim=1)
            print(
                f"[VocabTrim DEBUG] retained_hit={hit.detach().cpu().tolist()} "
                f"orig_top1={orig_top1.detach().cpu().tolist()} "
                f"vocabtrim_token={token_ids.detach().cpu().tolist()}"
            )

        return NextTokenResult(token_ids=token_ids, logits=logits_s, lm_head_time_ms=elapsed_ms)
