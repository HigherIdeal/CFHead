#!/usr/bin/env python3
import argparse
import gc
import importlib
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from algorithm.Original import OriginalHeadSelector


def clear_cuda_memory() -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

def resolve_model_path(explicit_path: str | None) -> str:
    if not explicit_path:
        raise ValueError("--path is required: specify the folder containing Llama-3.2-1B-Instruct weights.")
    if not Path(explicit_path).exists():
        raise FileNotFoundError(f"Model path does not exist: {explicit_path}")
    return explicit_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple chat inference for Llama3.2-1B-Instruct")
    parser.add_argument("--path", type=str, default=None, help="Folder containing Llama-3.2-1B-Instruct weights (required)")
    parser.add_argument("--prompt", type=str, default="안녕하세요. 자기소개를 간단히 해주세요.")
    parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--algorithm", type=str, default=None, help="Single algorithm name, e.g. Original or CFHead.")
    parser.add_argument(
        "--algorithms",
        type=str,
        default=None,
        help="Comma-separated algorithm names to run sequentially, e.g. CFHead or Original,CFHead.",
    )
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto, cpu, cuda, or cuda:<index>. auto/cuda select the CUDA GPU with the most free memory.",
    )
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    return parser


def resolve_device_and_dtype(device_arg: str, dtype_arg: str):
    device = resolve_device(device_arg)

    if dtype_arg == "auto":
        if is_cuda_device(device):
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
    else:
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype_arg]

    return device, dtype


def is_cuda_device(device: str) -> bool:
    return device == "cuda" or device.startswith("cuda:")


def resolve_device(device_arg: str) -> str:
    requested = device_arg.strip().lower()
    if requested == "cpu":
        return "cpu"
    if requested == "auto":
        if not torch.cuda.is_available():
            return "cpu"
        return select_cuda_device_with_most_free_memory()
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available.")
        return select_cuda_device_with_most_free_memory()
    if requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"--device {device_arg} was requested, but CUDA is not available.")
        try:
            index = int(requested.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid CUDA device: {device_arg}") from exc
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index out of range: {device_arg}")
        return f"cuda:{index}"
    raise ValueError("--device must be one of: auto, cpu, cuda, cuda:<index>")


def select_cuda_device_with_most_free_memory() -> str:
    best_index = 0
    best_free = -1
    for index in range(torch.cuda.device_count()):
        try:
            free_bytes, _ = torch.cuda.mem_get_info(index)
        except TypeError:
            with torch.cuda.device(index):
                free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes > best_free:
            best_index = index
            best_free = free_bytes
    return f"cuda:{best_index}"


def fix_seed(seed: int):
    random.seed(seed)
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_no_offload(model_path: str, device: str, torch_dtype: torch.dtype):
    """Load the model on one device only.

    device_map="auto" can silently offload modules to CPU when GPU memory is
    tight. That invalidates latency experiments, so CUDA runs must either keep
    the whole model on CUDA or fail with OOM.
    """
    clear_cuda_memory()
    model = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        return model.to(device)
    except torch.OutOfMemoryError as exc:
        if model is not None:
            del model
        clear_cuda_memory()
        if is_cuda_device(device):
            raise SystemExit(
                "\n[ERROR] CUDA out of memory while loading the model on a single GPU.\n"
                f"[ERROR] selected_device={device}, {cuda_memory_status(device)}\n"
                "[ERROR] CPU offload is disabled because it makes latency comparisons invalid.\n"
                "[HINT] Free GPU memory with `nvidia-smi`, stop the occupying process, "
                "or pass a specific free GPU with `--device cuda:<index>`."
            ) from exc
        raise


def cuda_memory_status(device: str) -> str:
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(torch.device(device))
    except TypeError:
        with torch.cuda.device(torch.device(device)):
            free_bytes, total_bytes = torch.cuda.mem_get_info()
    mib = 1024 * 1024
    return f"free={free_bytes / mib:.0f}MiB / total={total_bytes / mib:.0f}MiB"


def main():
    args = build_parser().parse_args()
    fix_seed(args.seed)
    model_path = resolve_model_path(args.path)
    device, torch_dtype = resolve_device_and_dtype(args.device, args.dtype)
    methods = resolve_methods(args)

    print(f"[INFO] model_path={model_path}")
    print(f"[INFO] device={device}, dtype={torch_dtype}, batch_size={args.batch_size}, seed={args.seed}")
    if is_cuda_device(device):
        print(f"[INFO] cuda_memory={cuda_memory_status(device)}")
    print(f"[INFO] methods={', '.join(methods)}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model_no_offload(model_path, device, torch_dtype)
    model.eval()

    prompts = [args.prompt for _ in range(args.batch_size)]
    messages_batch = [
        [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": prompt},
        ]
        for prompt in prompts
    ]

    input_texts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in messages_batch
    ]
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    for method in methods:
        clear_cuda_memory()
        selector = build_selector(method, model)

        try:
            start_time = time.perf_counter()
            generated_ids, lm_head_time_ms = generate_with_selector(
                model=model,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                selector=selector,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
            )
            e2e_time_ms = (time.perf_counter() - start_time) * 1000.0
            responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        finally:
            del selector
            clear_cuda_memory()

        print(f"\n[INFO] algorithm={method}, decoding=greedy, temperature=0")
        print(f"[INFO] lm_head_time_ms={lm_head_time_ms:.3f}, e2e_time_ms={e2e_time_ms:.3f}")

        for idx, response in enumerate(responses, start=1):
            print(f"\n[{method} Response {idx}]")
            print(response.strip())


def resolve_methods(args) -> list[str]:
    if args.algorithms:
        methods = [normalize_method_name(name.strip()) for name in args.algorithms.split(",") if name.strip()]
    elif args.algorithm:
        methods = [normalize_method_name(args.algorithm.strip())]
    else:
        methods = ["Original"]

    if not methods:
        raise ValueError("At least one algorithm must be provided.")
    return methods


def normalize_method_name(method: str) -> str:
    if method.lower() == "flashhead":
        return "FlashHead"
    return method


def build_selector(method: str, model):
    if method == "Original":
        return OriginalHeadSelector(
            lm_head=model.get_output_embeddings(),
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
        )

    module = importlib.import_module(f"algorithm.{method}")
    if not hasattr(module, "build_selector"):
        raise NotImplementedError(f"algorithm/{method}.py does not define build_selector(model)")
    if getattr(module, "IMPLEMENTED", True) is False:
        raise NotImplementedError(f"{method} is not implemented yet.")
    return module.build_selector(model)


@torch.no_grad()
def generate_with_selector(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    selector: OriginalHeadSelector,
    max_new_tokens: int,
    eos_token_id: int,
):
    past_key_values = None
    generated = []
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
    lm_head_time_ms = 0.0

    for step in range(max_new_tokens):
        model_inputs = {
            "input_ids": input_ids if step == 0 else next_token_ids.unsqueeze(-1),
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": True,
            "return_dict": True,
        }
        outputs = forward_backbone(model, **model_inputs)
        past_key_values = outputs.past_key_values
        last_hidden = outputs.last_hidden_state[:, -1, :]

        selected = selector.select(last_hidden)
        lm_head_time_ms += selected.lm_head_time_ms
        next_token_ids = selected.token_ids.masked_fill(finished, eos_token_id)
        generated.append(next_token_ids)

        finished = finished | (next_token_ids == eos_token_id)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((attention_mask.shape[0], 1), device=attention_mask.device, dtype=attention_mask.dtype)],
            dim=1,
        )
        if torch.all(finished):
            break

    return torch.stack(generated, dim=1), lm_head_time_ms


def forward_backbone(model, **model_inputs):
    """Run only the transformer body, not the CausalLM lm_head.

    The selector owns the LM-head step. Calling AutoModelForCausalLM.forward()
    would also materialize full-vocabulary logits that are immediately thrown
    away, which is both slower and can OOM on long prompts.
    """
    backbone = getattr(model, "model", None)
    if backbone is None:
        model_inputs["output_hidden_states"] = True
        outputs = model(**model_inputs)
        last_hidden_state = outputs.hidden_states[-1]
        return type("BackboneOutput", (), {
            "past_key_values": outputs.past_key_values,
            "last_hidden_state": last_hidden_state,
        })()
    return backbone(**model_inputs)


if __name__ == "__main__":
    main()
