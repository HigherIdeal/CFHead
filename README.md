# CFHead: Coarse-Fine Approximation of the LM Head for Efficient Inference

> **[Paper]** · **[arXiv]** *(coming soon)*

---

## Overview

The **LM Head** (the final linear projection from hidden states to vocabulary logits) accounts for a disproportionate share of inference latency in modern LLMs — its weight matrix is `[V × H]` (e.g. 128,256 × 2,048 ≈ 500 MiB for Llama-3.2-1B), and it must be read in full for every generated token.

**CFHead** replaces this full matrix–vector product with a two-stage coarse-to-fine lookup:

1. **Offline** — Tile the hidden dimension into `N = H / T` tiles. For each tile, cluster the `V` vocabulary rows into `C` centroids (k-means). Store a compact codebook `[N, C, T]` and an integer index map `[V, N]`.
2. **Online** — Build a look-up table (LUT) `[N, C]` from the hidden state, accumulate coarse scores for all `V` tokens in O(V) without touching the full weight, select the top-k candidates, then compute exact logits **only** for those candidates.

This reduces the amount of weight data read per token from ~500 MiB (full LM head) to ~31 MiB (codebook + index map), while preserving top-1 accuracy.

---

## Method

```
W_lm : [V, H]   (LM Head weight, ~500 MiB BF16)

── Offline ───────────────────────────────────────────────────────
Split H into N tiles of size T.
For each tile t:
    cluster W[:, t*T:(t+1)*T]  →  codebook[t] : [C, T]
                                   index_map[:, t] : [V]  (uint8)

── Online (per token) ────────────────────────────────────────────
hidden h : [H]
  →  h_tiles : [N, T]
  →  LUT[t, c] = ⟨h_tiles[t], codebook[t, c]⟩   (N·C multiplies)
  →  coarse[v] = Σ_t LUT[t, index_map[v, t]]      (V·N lookups, no GEMM)
  →  candidates = top-k(coarse)                    (k ≪ V)
  →  exact logits = h @ W[candidates].T            (k·H multiplies)
  →  next token = argmax(exact logits)
```

**Key properties:**
| | Original | CFHead (ours) |
|---|:---:|:---:|
| Weight bytes read / token | ~500 MiB | ~31 MiB |
| Arithmetic | Full GEMV | LUT + small GEMV |
| Offline artifact | — | codebook + index map |
| Top-1 accuracy | 100% | ≈ 100% (top-k dependent) |

---

## Baselines

| Method | Description |
|---|---|
| **Original** | Full BF16 LM Head (`h @ W.T`) |
| **CFHead** (ours) | Coarse-Fine tile-clustered LM Head |
| **FlashHead** | Cluster-level retrieval (centroid + C2T table) |
| **VocabTrim** | Static retained-vocabulary subset |

---

## Results

Evaluated on `meta-llama/Llama-3.2-1B-Instruct` (BF16, greedy, batch size 1).

| Method | Memory | Arc-Challenge | HellaSwag | IFEval | GSM8K | LM/sample | TPOT |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Original | 500 MiB | — | — | — | — | 1.00× | 1.00× |
| VocabTrim | ~32 MiB | — | — | — | — | — | — |
| FlashHead | ~32 MiB | — | — | — | — | — | — |
| **CFHead** (ours) | ~31 MiB | — | — | — | — | — | — |

*Full results will be updated upon paper release.*

---

## Requirements

```
Python >= 3.10
PyTorch >= 2.1 (CUDA)
transformers
```

Optional (for Triton/CUDA kernels):
```
triton
ninja
```

Install:
```bash
pip install torch transformers
# optional
pip install triton ninja
```

---

## Quickstart

```bash
# ARC-Challenge — all four methods
bash sh/arc_challenge.sh --path /path/to/Llama-3.2-1B-Instruct

# Specific methods only
bash sh/arc_challenge.sh --path /path/to/Llama-3.2-1B-Instruct --algorithms CFHead,Original

# Original baseline only
bash sh/arc_challenge.sh --path /path/to/Llama-3.2-1B-Instruct --origin_only
```

Available task scripts: `arc_challenge`, `hellaswag`, `gsm8k`, `ifeval`

Results are written to `result/` and the live leaderboard to `result/leaderboard.html`.

### CFHead hyperparameters

| Parameter | Default | Description |
|---|:---:|---|
| `CFHEAD_TILE_SIZE` | 8 | Hidden-dim tile size `T` |
| `CFHEAD_CLUSTER_SIZE` | 4 | Centroids per tile `C` |
| `CFHEAD_TOP_K` | 8192 | Candidate tokens |
| `CFHEAD_KMEANS_ITERS` | 10 | k-means iterations |

Set via environment variables:
```bash
CFHEAD_TOP_K=4096 bash sh/arc_challenge.sh --path /path/to/model
```

---

## Repository Structure

```
.
├── algorithm/          # LM Head selectors
│   ├── CFHead.py       # Proposed method
│   ├── FlashHead.py    # Baseline: cluster-level retrieval
│   ├── VocabTrim.py    # Baseline: static vocab subset
│   └── Original.py     # Full BF16 LM Head
├── kernel/             # CUDA / Triton kernels for CFHead
├── script/             # Evaluation harness
│   ├── eval_common.py  # Shared runner, leaderboard writer
│   ├── arc_challenge.py
│   ├── hellaswag.py
│   ├── gsm8k.py
│   └── ifeval.py
├── sh/                 # Convenience shell scripts
├── data/               # Benchmark datasets (JSONL)
└── result/             # Outputs and leaderboard
```

---

## Citation

```bibtex
@article{cfhead2025,
  title   = {CFHead: Coarse-Fine Approximation of the LM Head for Efficient Inference},
  author  = {},
  journal = {},
  year    = {2025}
}
```

---

## License

This repository is released for research purposes.
The model weights (Llama-3.2) are subject to the [Meta Llama 3 Community License](https://llama.meta.com/llama3/license/).
