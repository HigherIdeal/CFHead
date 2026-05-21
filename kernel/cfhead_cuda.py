from __future__ import annotations

import os
from pathlib import Path

import torch


def _candidate_cuda_homes() -> list[Path]:
    torch_cuda = torch.version.cuda or ""
    candidates: list[Path] = []
    if torch_cuda.startswith("12."):
        candidates.extend([Path("/usr/local/cuda-12.1"), Path("/usr/local/cuda")])
    elif torch_cuda.startswith("11."):
        candidates.extend([Path("/usr/local/cuda-11.8"), Path("/usr/local/cuda")])
    candidates.extend([Path("/usr/local/cuda-12.1"), Path("/usr/local/cuda-11.8"), Path("/usr/local/cuda")])
    return candidates


def _configure_cuda_home() -> None:
    current = os.environ.get("CUDA_HOME")
    torch_cuda = torch.version.cuda or ""
    if (
        current
        and (Path(current) / "bin" / "nvcc").exists()
        and (not torch_cuda.startswith("12.") or "cuda-12" in str(Path(current).resolve()))
        and (not torch_cuda.startswith("11.") or "cuda-11" in str(Path(current).resolve()))
    ):
        return

    for cuda_home in _candidate_cuda_homes():
        if (cuda_home / "bin" / "nvcc").exists():
            os.environ["CUDA_HOME"] = str(cuda_home)
            os.environ["PATH"] = f"{cuda_home / 'bin'}:{os.environ.get('PATH', '')}"
            lib64 = cuda_home / "lib64"
            if lib64.exists():
                os.environ["LD_LIBRARY_PATH"] = f"{lib64}:{os.environ.get('LD_LIBRARY_PATH', '')}"
            return


def _load_extension():
    if not torch.cuda.is_available():
        return None

    _configure_cuda_home()
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"

    try:
        from torch.utils.cpp_extension import load

        this_dir = Path(__file__).resolve().parent
        build_dir = this_dir / "build"
        build_dir.mkdir(parents=True, exist_ok=True)

        return load(
            name="cfhead_cuda_ext",
            sources=[
                str(this_dir / "cfhead_cuda.cpp"),
                str(this_dir / "cfhead_cuda_kernel.cu"),
            ],
            build_directory=str(build_dir),
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=os.environ.get("CFHEAD_CUDA_VERBOSE", "0").strip() == "1",
        )
    except Exception as exc:
        print(f"[CFHead CUDA] extension unavailable, falling back to torch path. reason={exc}")
        return None


_EXT = _load_extension()
CUDA_KERNEL_AVAILABLE = _EXT is not None


def cfhead_coarse_cuda(lut: torch.Tensor, index_t: torch.Tensor) -> torch.Tensor:
    if _EXT is None:
        raise RuntimeError("CFHead CUDA extension is not available")
    return _EXT.coarse_forward(lut.contiguous(), index_t.contiguous())


def cfhead_fine_cuda(hidden: torch.Tensor, weight: torch.Tensor, candidate_idx: torch.Tensor) -> torch.Tensor:
    if _EXT is None:
        raise RuntimeError("CFHead CUDA extension is not available")
    return _EXT.fine_forward(hidden.contiguous(), weight.contiguous(), candidate_idx.contiguous())
