#!/usr/bin/env python3
import argparse
import csv
import gc
import hashlib
import html
import importlib
import json
import random
import re
import sys
import time
import types
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer, set_seed

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from algorithm.Original import OriginalHeadSelector
from script.chat import (
    cuda_memory_status,
    generate_with_selector,
    is_cuda_device,
    load_model_no_offload,
    resolve_device_and_dtype,
    resolve_model_path,
)


DEFAULT_ALGORITHMS = ["Original", "CFHead", "FlashHead", "VocabTrim"]
DEFAULT_SEED = 99
DEFAULT_SAMPLE_COUNT = 128
DEFAULT_NUM_SHOTS = 5


def clear_cuda_memory() -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

TASK_DEFAULTS = {
    "arc_challenge": {"max_new_tokens": 4, "dataset_path": "data/arc_challenge.jsonl", "num_shots": DEFAULT_NUM_SHOTS},
    "hellaswag": {"max_new_tokens": 4, "dataset_path": "data/hellaswag.jsonl", "num_shots": DEFAULT_NUM_SHOTS},
    "ifeval": {"max_new_tokens": 1280, "dataset_path": "data/ifeval.jsonl", "num_shots": 0},
    "gsm8k": {"max_new_tokens": 512, "dataset_path": "data/gsm8k.jsonl", "num_shots": DEFAULT_NUM_SHOTS},
}


def build_parser(task: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"{task} evaluation runner")
    parser.add_argument("--task", type=str, default=task)
    parser.add_argument("--path", type=str, default=None, help="Folder containing Llama-3.2-1B-Instruct weights (required)")
    parser.add_argument(
        "--algorithms",
        type=str,
        default=",".join(DEFAULT_ALGORITHMS),
        help="Comma-separated algorithm modules to evaluate. Default: CFHead,FlashHead,VocabTrim.",
    )
    parser.add_argument("--origin_only", action="store_true", help="Evaluate only the Original BF16 LM head.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="JSONL file with prompt/target fields. Defaults to data/<task>.jsonl.",
    )
    parser.add_argument("--output_dir", type=str, default="result")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=DEFAULT_SAMPLE_COUNT,
        help=f"Number of samples to evaluate after seed-fixed shuffle. Default: {DEFAULT_SAMPLE_COUNT}.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=TASK_DEFAULTS[task]["max_new_tokens"])
    parser.add_argument("--num_shots", type=int, default=TASK_DEFAULTS[task].get("num_shots", DEFAULT_NUM_SHOTS))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no_chat_template", action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto, cpu, cuda, or cuda:<index>. auto/cuda select the CUDA GPU with the most free memory.",
    )
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    return parser


def load_jsonl(path: Path, max_samples: int | None, seed: int):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    rng = random.Random(seed)
    rng.shuffle(rows)
    if max_samples is not None:
        rows = rows[:max_samples]
    return rows


def batched(items, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start:start + batch_size]


def normalize_answer(text: str) -> str:
    return text.strip()


def extract_answer(text: str, answer_type: str) -> str:
    normalized = normalize_answer(text)
    if answer_type == "choice":
        match = re.search(r"\b([A-E])\b", normalized.upper())
        return match.group(1) if match else normalized[:1].upper()
    if answer_type == "gsm8k":
        match = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", normalized)
        if match:
            return match.group(1).replace(",", "")
        numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", normalized)
        return numbers[-1].replace(",", "") if numbers else normalized
    return normalized


def score_answer(prediction_text: str, target_text: str, answer_type: str) -> bool | None:
    if not target_text:
        return None
    return extract_answer(prediction_text, answer_type) == extract_answer(target_text, answer_type)


@dataclass
class EvalSummary:
    method: str
    status: str
    samples: int = 0
    accuracy: float | None = None
    total_lm_head_ms: float = 0.0
    total_e2e_ms: float = 0.0
    total_tokens: int = 0
    output_jsonl: str = ""
    prompt_level_strict_acc: float | None = None
    inst_level_strict_acc: float | None = None
    prompt_level_loose_acc: float | None = None
    inst_level_loose_acc: float | None = None
    memory_mib: float | None = None
    algo_config: dict = field(default_factory=dict)

    @property
    def avg_lm_head_ms(self) -> float:
        return self.total_lm_head_ms / max(self.samples, 1)

    @property
    def avg_e2e_ms(self) -> float:
        return self.total_e2e_ms / max(self.samples, 1)

    @property
    def tpot_ms(self) -> float:
        return self.total_e2e_ms / max(self.total_tokens, 1)


def run_task(task: str):
    args = build_parser(task).parse_args()
    fix_seed(args.seed)
    model_path = resolve_model_path(args.path)
    device, torch_dtype = resolve_device_and_dtype(args.device, args.dtype)
    dataset_path = resolve_dataset_path(args.task, args.dataset_path, args.num_shots)
    methods = resolve_methods(args)
    runnable_methods, summaries = preflight_methods(methods)
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{args.task}_summary.csv"
    leaderboard_path = output_dir / "leaderboard.csv"
    leaderboard_html_path = output_dir / "leaderboard.html"
    total_dataset_samples = count_jsonl(dataset_path)

    print(f"[INFO] task={args.task}, methods={', '.join(methods)}")
    print(f"[INFO] model_path={model_path}")
    print(f"[INFO] device={device}, dtype={torch_dtype}, batch_size={args.batch_size}")
    if is_cuda_device(device):
        print(f"[INFO] cuda_memory={cuda_memory_status(device)}")
    print(f"[INFO] decoding=greedy, temperature=0, seed={args.seed}")

    if not runnable_methods:
        write_summary_csv(summary_path, args.task, summaries)
        update_leaderboard(
            leaderboard_path=leaderboard_path,
            html_path=leaderboard_html_path,
            task=args.task,
            summaries=summaries,
            settings=build_settings(args, model_path, dataset_path, total_dataset_samples, total_dataset_samples, None),
        )
        print(f"[DONE] wrote {summary_path}")
        print(f"[DONE] updated {leaderboard_path}")
        print(f"[DONE] updated {leaderboard_html_path}")
        print_summary_table(args.task, summaries)
        return

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    samples = load_jsonl(dataset_path, args.max_samples, args.seed)
    prompt_stats = estimate_prompt_stats(tokenizer, samples, args.no_chat_template)
    scoring_mode = detect_scoring_mode(samples)
    print_setting_table(
        build_settings(args, model_path, dataset_path, total_dataset_samples, len(samples), prompt_stats, scoring_mode)
    )

    model = load_model_no_offload(model_path, device, torch_dtype)
    model.eval()

    print(f"[INFO] dataset={dataset_path}, samples={len(samples)}")

    for method in runnable_methods:
        clear_cuda_memory()
        selector = None
        try:
            selector = build_selector(method, model)
        except NotImplementedError as exc:
            print(f"[SKIP] {method}: {exc}")
            summaries.append(EvalSummary(method=method, status="not_implemented"))
            continue

        mem_mib = compute_selector_memory_mib(method, selector)
        algo_cfg = extract_selector_config(method, selector)
        try:
            summary = evaluate_method(
                task=args.task,
                method=method,
                model=model,
                tokenizer=tokenizer,
                selector=selector,
                samples=samples,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                no_chat_template=args.no_chat_template,
                output_dir=output_dir,
            )
            summary.memory_mib = mem_mib
            summary.algo_config = algo_cfg
            summaries.append(summary)
        finally:
            del selector
            clear_cuda_memory()

    write_summary_csv(summary_path, args.task, summaries)
    update_leaderboard(
        leaderboard_path=leaderboard_path,
        html_path=leaderboard_html_path,
        task=args.task,
        summaries=summaries,
        settings=build_settings(args, model_path, dataset_path, total_dataset_samples, len(samples), prompt_stats, scoring_mode),
    )
    print(f"[DONE] wrote {summary_path}")
    print(f"[DONE] updated {leaderboard_path}")
    print(f"[DONE] updated {leaderboard_html_path}")
    print_summary_table(args.task, summaries)


def resolve_methods(args) -> list[str]:
    if args.origin_only:
        return ["Original"]
    methods = [normalize_method_name(name.strip()) for name in args.algorithms.split(",") if name.strip()]
    return methods or DEFAULT_ALGORITHMS


def normalize_method_name(method: str) -> str:
    if method.lower() == "flashhead":
        return "FlashHead"
    return method


def build_selector(method: str, model):
    if method == "Original":
        return OriginalHeadSelector(lm_head=model.get_output_embeddings(), do_sample=False, temperature=0.0)

    module = importlib.import_module(f"algorithm.{method}")
    if not hasattr(module, "build_selector"):
        raise NotImplementedError(f"algorithm/{method}.py does not define build_selector(model)")
    return module.build_selector(model)


def compute_selector_memory_mib(method: str, selector) -> float | None:
    mib = 1024.0 * 1024.0
    try:
        if method == "Original":
            w = selector.lm_head.weight
            return w.numel() * w.element_size() / mib
        if method == "CFHead":
            codebook = selector.codebook.numel() * selector.codebook.element_size()
            index_map = selector.index_map.numel()  # uint8 = 1 byte each
            return (codebook + index_map) / mib
        if method == "FlashHead":
            centroids = selector.centroids.numel() * selector.centroids.element_size()
            c2t = selector.c2t.numel() * 4  # int32
            return (centroids + c2t) / mib
        if method == "VocabTrim":
            weight = selector.trimmed_weight.numel() * selector.trimmed_weight.element_size()
            ids = selector.retained_ids.numel() * 4  # int32
            return (weight + ids) / mib
    except Exception:
        pass
    return None


def extract_selector_config(method: str, selector) -> dict:
    try:
        if method == "Original":
            w = selector.lm_head.weight
            return {"vocab_size": int(w.shape[0]), "hidden_size": int(w.shape[1])}
        if method == "CFHead":
            return {
                "tile_size": selector.tile_size,
                "num_tiles": selector.num_tiles,
                "cluster_size": selector.cluster_size,
                "top_k": selector.top_k,
                "kmeans_iters": selector.kmeans_iters,
            }
        if method == "FlashHead":
            return {
                "num_clusters": selector.num_clusters,
                "num_probes": selector.num_probes,
                "tokens_per_cluster": selector.tokens_per_cluster,
                "candidate_tokens": selector.candidate_tokens,
                "cluster_iters": selector.cluster_iters,
            }
        if method == "VocabTrim":
            return {
                "retained_vocab": selector.retained_vocab_size,
                "vocab_size": selector.vocab_size,
                "fallback_mode": selector.fallback_mode,
            }
    except Exception:
        pass
    return {}


def preflight_methods(methods: list[str]) -> tuple[list[str], list[EvalSummary]]:
    runnable = []
    summaries = []
    for method in methods:
        if method == "Original":
            runnable.append(method)
            continue
        module = importlib.import_module(f"algorithm.{method}")
        if getattr(module, "IMPLEMENTED", True):
            runnable.append(method)
        else:
            print(f"[SKIP] {method}: not implemented yet")
            summaries.append(EvalSummary(method=method, status="not_implemented"))
    return runnable, summaries


def evaluate_method(
    task: str,
    method: str,
    model,
    tokenizer,
    selector,
    samples,
    batch_size: int,
    max_new_tokens: int,
    no_chat_template: bool,
    output_dir: Path,
) -> EvalSummary:
    if task == "ifeval":
        return evaluate_ifeval(
            method=method,
            model=model,
            tokenizer=tokenizer,
            selector=selector,
            samples=samples,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            no_chat_template=no_chat_template,
            output_dir=output_dir,
        )

    jsonl_path = output_dir / f"{task}_{method.lower()}.jsonl"
    correct = 0
    total = 0
    total_lm_head_ms = 0.0
    total_e2e_ms = 0.0
    total_tokens = 0
    batch_count = (len(samples) + batch_size - 1) // batch_size

    print(f"[RUN] {task}/{method}: {len(samples)} samples, {batch_count} batches")

    with jsonl_path.open("w", encoding="utf-8") as out:
        for batch_idx, (start_idx, batch) in enumerate(batched(samples, batch_size), start=1):
            prompts = [format_prompt(tokenizer, row, no_chat_template) for row in batch]
            targets = [str(row.get("target", "")) for row in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True)
            inputs = {key: value.to(model.device) for key, value in inputs.items()}

            start_time = time.perf_counter()
            generated_ids, lm_head_ms = generate_with_selector(
                model=model,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                selector=selector,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
            )
            e2e_ms = (time.perf_counter() - start_time) * 1000.0
            outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            total_lm_head_ms += lm_head_ms
            total_e2e_ms += e2e_ms
            total_tokens += generated_ids.numel()

            for offset, (row, output, target) in enumerate(zip(batch, outputs, targets)):
                sample_id = row.get("sample_id", start_idx + offset)
                answer_type = row.get("answer_type", "text")
                prediction = extract_answer(output, answer_type)
                target_answer = extract_answer(target, answer_type)
                is_correct = score_answer(output, target, answer_type)
                if is_correct:
                    correct += 1
                total += 1

                record = {
                    "task": task,
                    "sample_id": sample_id,
                    "prompt_hash": hashlib.sha1(row["prompt"].encode("utf-8")).hexdigest(),
                    "target": target,
                    "target_answer": target_answer,
                    "original_output": output if method == "Original" else None,
                    "method_output": output,
                    "original_answer": prediction if method == "Original" else None,
                    "method_answer": prediction,
                    "is_correct_original": is_correct if method == "Original" else None,
                    "is_correct_method": is_correct,
                    "token_agreement": 1.0 if method == "Original" else None,
                    "containment_count": None,
                    "generated_token_count": generated_ids.shape[1],
                    "lm_head_time_ms": lm_head_ms / max(len(batch), 1),
                    "e2e_time_ms": e2e_ms / max(len(batch), 1),
                    "tpot_ms": e2e_ms / max(generated_ids.numel(), 1),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

            print_progress(
                method=method,
                batch_idx=batch_idx,
                batch_count=batch_count,
                total=total,
                correct=correct,
                total_e2e_ms=total_e2e_ms,
                total_tokens=total_tokens,
            )

    print(f"\n[DONE] wrote {jsonl_path}")
    return EvalSummary(
        method=method,
        status="done",
        samples=total,
        accuracy=correct / total if total else 0.0,
        total_lm_head_ms=total_lm_head_ms,
        total_e2e_ms=total_e2e_ms,
        total_tokens=total_tokens,
        output_jsonl=str(jsonl_path),
    )


def print_progress(method: str, batch_idx: int, batch_count: int, total: int, correct: int, total_e2e_ms: float, total_tokens: int):
    percent = 100.0 * batch_idx / max(batch_count, 1)
    accuracy = 100.0 * correct / max(total, 1)
    tpot = total_e2e_ms / max(total_tokens, 1)
    print(
        f"\r[{method}] {batch_idx}/{batch_count} batches ({percent:6.2f}%) | "
        f"samples={total} | acc={accuracy:6.2f}% | TPOT={format_ms(tpot)}",
        end="",
        flush=True,
    )


def evaluate_ifeval(
    method: str,
    model,
    tokenizer,
    selector,
    samples,
    batch_size: int,
    max_new_tokens: int,
    no_chat_template: bool,
    output_dir: Path,
) -> EvalSummary:
    ifeval_utils = load_ifeval_utils()

    jsonl_path = output_dir / f"ifeval_{method.lower()}.jsonl"
    prompt_strict = []
    prompt_loose = []
    inst_strict = []
    inst_loose = []
    total_lm_head_ms = 0.0
    total_e2e_ms = 0.0
    total_tokens = 0
    batch_count = (len(samples) + batch_size - 1) // batch_size

    print(f"[RUN] ifeval/{method}: {len(samples)} samples, {batch_count} batches")

    with jsonl_path.open("w", encoding="utf-8") as out:
        for batch_idx, (start_idx, batch) in enumerate(batched(samples, batch_size), start=1):
            prompts = [format_prompt(tokenizer, row, no_chat_template) for row in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True)
            inputs = {key: value.to(model.device) for key, value in inputs.items()}

            start_time = time.perf_counter()
            generated_ids, lm_head_ms = generate_with_selector(
                model=model,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                selector=selector,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
            )
            e2e_ms = (time.perf_counter() - start_time) * 1000.0
            outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            total_lm_head_ms += lm_head_ms
            total_e2e_ms += e2e_ms
            total_tokens += generated_ids.numel()

            for offset, (row, response) in enumerate(zip(batch, outputs)):
                metadata = row.get("metadata", {})
                doc = {
                    "key": row.get("sample_id", start_idx + offset),
                    "prompt": row["prompt"],
                    "instruction_id_list": metadata.get("instruction_id_list", []),
                    "kwargs": metadata.get("kwargs", []),
                }
                metrics = ifeval_utils.process_results(doc, [response])

                prompt_strict.append(bool(metrics["prompt_level_strict_acc"]))
                prompt_loose.append(bool(metrics["prompt_level_loose_acc"]))
                inst_strict.extend([bool(item) for item in metrics["inst_level_strict_acc"]])
                inst_loose.extend([bool(item) for item in metrics["inst_level_loose_acc"]])

                record = {
                    "task": "ifeval",
                    "sample_id": doc["key"],
                    "prompt_hash": hashlib.sha1(row["prompt"].encode("utf-8")).hexdigest(),
                    "method_output": response,
                    "prompt_level_strict_acc": bool(metrics["prompt_level_strict_acc"]),
                    "prompt_level_loose_acc": bool(metrics["prompt_level_loose_acc"]),
                    "inst_level_strict_acc": metrics["inst_level_strict_acc"],
                    "inst_level_loose_acc": metrics["inst_level_loose_acc"],
                    "instruction_id_list": doc["instruction_id_list"],
                    "generated_token_count": generated_ids.shape[1],
                    "lm_head_time_ms": lm_head_ms / max(len(batch), 1),
                    "e2e_time_ms": e2e_ms / max(len(batch), 1),
                    "tpot_ms": e2e_ms / max(generated_ids.numel(), 1),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

            print_progress_ifeval(
                method=method,
                batch_idx=batch_idx,
                batch_count=batch_count,
                prompt_strict=mean_bool(prompt_strict),
                inst_strict=mean_bool(inst_strict),
                prompt_loose=mean_bool(prompt_loose),
                inst_loose=mean_bool(inst_loose),
                total_e2e_ms=total_e2e_ms,
                total_tokens=total_tokens,
            )

    p_strict = mean_bool(prompt_strict)
    i_strict = mean_bool(inst_strict)
    p_loose = mean_bool(prompt_loose)
    i_loose = mean_bool(inst_loose)
    avg_score = (p_strict + i_strict + p_loose + i_loose) / 4.0

    print(f"\n[DONE] wrote {jsonl_path}")
    return EvalSummary(
        method=method,
        status="done",
        samples=len(samples),
        accuracy=avg_score,
        total_lm_head_ms=total_lm_head_ms,
        total_e2e_ms=total_e2e_ms,
        total_tokens=total_tokens,
        output_jsonl=str(jsonl_path),
        prompt_level_strict_acc=p_strict,
        inst_level_strict_acc=i_strict,
        prompt_level_loose_acc=p_loose,
        inst_level_loose_acc=i_loose,
    )


def print_progress_ifeval(
    method: str,
    batch_idx: int,
    batch_count: int,
    prompt_strict: float,
    inst_strict: float,
    prompt_loose: float,
    inst_loose: float,
    total_e2e_ms: float,
    total_tokens: int,
):
    percent = 100.0 * batch_idx / max(batch_count, 1)
    avg_score = (prompt_strict + inst_strict + prompt_loose + inst_loose) / 4.0
    tpot = total_e2e_ms / max(total_tokens, 1)
    print(
        f"\r[{method}] {batch_idx}/{batch_count} batches ({percent:6.2f}%) | "
        f"avg={to_pct(avg_score):6.2f}% | "
        f"strict(P/I)={to_pct(prompt_strict):5.1f}/{to_pct(inst_strict):5.1f}% | "
        f"loose(P/I)={to_pct(prompt_loose):5.1f}/{to_pct(inst_loose):5.1f}% | "
        f"TPOT={format_ms(tpot)}",
        end="",
        flush=True,
    )


def mean_bool(items: list[bool]) -> float:
    return sum(items) / len(items) if items else 0.0


def load_ifeval_utils():
    module_name = "lm_eval.tasks.ifeval.utils"
    if module_name in sys.modules:
        return sys.modules[module_name]

    base_dir = Path("/home/iclab/JW/EVAL/lm-evaluation-harness/lm_eval/tasks/ifeval")
    if not base_dir.exists():
        raise FileNotFoundError(f"Could not find local IFEval checker directory: {base_dir}")

    ensure_package("lm_eval", base_dir.parents[2])
    ensure_package("lm_eval.tasks", base_dir.parent)
    ensure_package("lm_eval.tasks.ifeval", base_dir)
    ensure_immutabledict_fallback()

    for filename in ["instructions_util.py", "instructions.py", "instructions_registry.py", "utils.py"]:
        load_module_from_path(f"lm_eval.tasks.ifeval.{filename[:-3]}", base_dir / filename)

    return sys.modules[module_name]


def ensure_immutabledict_fallback():
    if importlib.util.find_spec("immutabledict") is not None:
        return
    module = types.ModuleType("immutabledict")
    module.immutabledict = dict
    sys.modules["immutabledict"] = module


def ensure_package(name: str, path: Path):
    if name in sys.modules:
        return
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package


def load_module_from_path(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    parent_name, attr = name.rsplit(".", 1)
    setattr(sys.modules[parent_name], attr, module)
    return module


def write_summary_csv(summary_path: Path, task: str, summaries: list[EvalSummary]):
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "task",
            "method",
            "status",
            "samples",
            "accuracy_pct",
            "prompt_level_strict_acc",
            "inst_level_strict_acc",
            "prompt_level_loose_acc",
            "inst_level_loose_acc",
            "avg_lm_head_ms",
            "avg_e2e_ms",
            "tpot_ms",
            "output_jsonl",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "task": task,
                    "method": summary.method,
                    "status": summary.status,
                    "samples": summary.samples,
                    "accuracy_pct": to_pct(summary.accuracy) if summary.accuracy is not None else "",
                    "prompt_level_strict_acc": to_pct(summary.prompt_level_strict_acc) if summary.prompt_level_strict_acc is not None else "",
                    "inst_level_strict_acc": to_pct(summary.inst_level_strict_acc) if summary.inst_level_strict_acc is not None else "",
                    "prompt_level_loose_acc": to_pct(summary.prompt_level_loose_acc) if summary.prompt_level_loose_acc is not None else "",
                    "inst_level_loose_acc": to_pct(summary.inst_level_loose_acc) if summary.inst_level_loose_acc is not None else "",
                    "avg_lm_head_ms": f"{summary.avg_lm_head_ms:.4f}" if summary.samples else "",
                    "avg_e2e_ms": f"{summary.avg_e2e_ms:.4f}" if summary.samples else "",
                    "tpot_ms": f"{summary.tpot_ms:.4f}" if summary.samples else "",
                    "output_jsonl": summary.output_jsonl,
                }
            )


def print_summary_table(task: str, summaries: list[EvalSummary]):
    rows = []
    for summary in summaries:
        rows.append(
            [
                summary.method,
                summary.status,
                str(summary.samples) if summary.samples else "-",
                format_pct(summary.accuracy),
                format_pct(summary.prompt_level_strict_acc),
                format_pct(summary.inst_level_strict_acc),
                format_pct(summary.prompt_level_loose_acc),
                format_pct(summary.inst_level_loose_acc),
                format_ms(summary.avg_lm_head_ms) if summary.samples else "-",
                format_ms(summary.avg_e2e_ms) if summary.samples else "-",
                format_ms(summary.tpot_ms) if summary.samples else "-",
            ]
        )

    print(f"\n[SUMMARY] {task}")
    print_table(["Method", "Status", "N", "Acc", "PStrict", "IStrict", "PLoose", "ILoose", "LM/sample", "E2E/sample", "TPOT"], rows)


def print_table(headers: list[str], rows: list[list[str]]):
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]
    line = " | ".join(header.ljust(width) for header, width in zip(headers, widths))
    sep = "-+-".join("-" * width for width in widths)
    print(line)
    print(sep)
    for row in rows:
        print(" | ".join(cell.ljust(width) for cell, width in zip(row, widths)))


def to_pct(value: float) -> float:
    return value * 100.0


def format_pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{to_pct(value):.2f}%"


def format_ms(value: float) -> str:
    if value >= 1000.0:
        return f"{value / 1000.0:.2f}s"
    if value >= 10.0:
        return f"{value:.1f}ms"
    return f"{value:.2f}ms"


def fix_seed(seed: int):
    random.seed(seed)
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_output_dir(output_dir: str) -> Path:
    path = Path(output_dir)
    return path if path.is_absolute() else ROOT_DIR / path


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def estimate_prompt_stats(tokenizer, samples, no_chat_template: bool) -> dict:
    if not samples:
        return {"avg_prompt_tokens": 0, "max_prompt_tokens": 0}
    lengths = []
    for row in samples:
        prompt = format_prompt(tokenizer, row, no_chat_template)
        lengths.append(len(tokenizer(prompt, add_special_tokens=False)["input_ids"]))
    return {
        "avg_prompt_tokens": sum(lengths) / len(lengths),
        "max_prompt_tokens": max(lengths),
    }


def detect_scoring_mode(samples) -> str:
    if samples and samples[0].get("metric_type"):
        return samples[0]["metric_type"]
    return "generate_until"


def expected_metric_type(task: str) -> str:
    if task in {"arc_challenge", "hellaswag"}:
        return "fewshot_choice_generation"
    if task == "gsm8k":
        return "fewshot_generation"
    return "generate_until"


def _resolve_gpu_name(device: str) -> str:
    try:
        if torch.cuda.is_available() and "cuda" in str(device):
            idx = int(str(device).split(":")[1]) if ":" in str(device) else 0
            return torch.cuda.get_device_name(idx)
    except Exception:
        pass
    return str(device)


def build_settings(
    args,
    model_path: str,
    dataset_path: Path,
    total_samples: int,
    used_samples: int,
    prompt_stats: dict | None,
    scoring_mode: str = "generate_until",
) -> dict:
    prompt_stats = prompt_stats or {"avg_prompt_tokens": None, "max_prompt_tokens": None}
    gpu_name = _resolve_gpu_name(args.device)
    return {
        "task": args.task,
        "model": model_path,
        "dataset": str(dataset_path),
        "dataset_total": total_samples,
        "samples_used": used_samples,
        "sample_selection": f"seed_shuffle_first_{used_samples}",
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "scoring_mode": scoring_mode,
        "num_shots": args.num_shots,
        "avg_prompt_tokens": prompt_stats["avg_prompt_tokens"],
        "max_prompt_tokens": prompt_stats["max_prompt_tokens"],
        "decoding": "greedy",
        "temperature": 0,
        "seed": args.seed,
        "dtype": args.dtype,
        "device": args.device,
        "gpu_name": gpu_name,
        "chat_template": "off" if args.no_chat_template else "on",
    }


def print_setting_table(settings: dict):
    rows = [
        ["Task", settings["task"]],
        ["Dataset", settings["dataset"]],
        ["Dataset samples", str(settings["dataset_total"])],
        ["Used samples", str(settings["samples_used"])],
        ["Sample selection", settings["sample_selection"]],
        ["Batch size", str(settings["batch_size"])],
        ["Scoring", settings["scoring_mode"]],
        ["Num shots", str(settings["num_shots"])],
        ["Max new tokens", str(settings["max_new_tokens"])],
        ["Avg prompt tokens", format_number(settings["avg_prompt_tokens"])],
        ["Max prompt tokens", format_number(settings["max_prompt_tokens"])],
        ["Decoding", "greedy / temperature=0"],
        ["Seed", str(settings["seed"])],
    ]
    print("\n[SETTING]")
    print_table(["Item", "Value"], rows)


def format_number(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def update_leaderboard(leaderboard_path: Path, html_path: Path, task: str, summaries: list[EvalSummary], settings: dict):
    rows = read_csv_rows(leaderboard_path)
    existing = {(row.get("experiment_id", legacy_experiment_id(row)), row["method"]): row for row in rows if row.get("method")}
    updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    experiment_id = make_experiment_id(task, settings)

    for summary in summaries:
        row = {
            "experiment_id": experiment_id,
            "task": task,
            "scoring_mode": settings["scoring_mode"],
            "dataset": settings["dataset"],
            "dataset_total": str(settings["dataset_total"]),
            "samples_used": str(summary.samples or settings["samples_used"]),
            "sample_selection": settings["sample_selection"],
            "num_shots": str(settings["num_shots"]),
            "max_new_tokens": str(settings["max_new_tokens"]),
            "batch_size": str(settings["batch_size"]),
            "seed": str(settings["seed"]),
            "decoding": settings["decoding"],
            "temperature": str(settings["temperature"]),
            "dtype": settings["dtype"],
            "device": settings["device"],
            "method": summary.method,
            "status": summary.status,
            "score_pct": f"{to_pct(summary.accuracy):.2f}" if summary.accuracy is not None else "",
            "prompt_strict_pct": f"{to_pct(summary.prompt_level_strict_acc):.2f}" if summary.prompt_level_strict_acc is not None else "",
            "inst_strict_pct": f"{to_pct(summary.inst_level_strict_acc):.2f}" if summary.inst_level_strict_acc is not None else "",
            "prompt_loose_pct": f"{to_pct(summary.prompt_level_loose_acc):.2f}" if summary.prompt_level_loose_acc is not None else "",
            "inst_loose_pct": f"{to_pct(summary.inst_level_loose_acc):.2f}" if summary.inst_level_loose_acc is not None else "",
            "avg_prompt_tokens": format_number(settings["avg_prompt_tokens"]),
            "max_prompt_tokens": format_number(settings["max_prompt_tokens"]),
            "avg_lm_head": format_ms(summary.avg_lm_head_ms) if summary.samples else "",
            "avg_e2e": format_ms(summary.avg_e2e_ms) if summary.samples else "",
            "tpot": format_ms(summary.tpot_ms) if summary.samples else "",
            "memory_mib": f"{summary.memory_mib:.3f}" if summary.memory_mib is not None else "",
            "gpu_name": settings.get("gpu_name", ""),
            "algo_config": json.dumps(summary.algo_config) if summary.algo_config else "",
            "output_jsonl": summary.output_jsonl,
            "updated_at": updated_at,
        }
        existing[(experiment_id, summary.method)] = row

    ordered_rows = sorted(existing.values(), key=lambda row: (row.get("experiment_id", legacy_experiment_id(row)), row["method"]))
    write_leaderboard_csv(leaderboard_path, ordered_rows)
    write_leaderboard_html(html_path, read_csv_rows(leaderboard_path))


def make_experiment_id(task: str, settings: dict) -> str:
    return "|".join(
        [
            task,
            settings["scoring_mode"],
            f"shots={settings['num_shots']}",
            f"samples={settings['samples_used']}",
            f"select={settings['sample_selection']}",
            f"max_new={settings['max_new_tokens']}",
            f"seed={settings['seed']}",
        ]
    )


def legacy_experiment_id(row: dict) -> str:
    task = row.get("task", "")
    return "|".join(
        [
            task,
            row.get("scoring_mode", ""),
            f"shots={row.get('num_shots', '')}",
            f"samples={row.get('samples_used', '')}",
            f"select={row.get('sample_selection', '')}",
            f"max_new={row.get('max_new_tokens', '')}",
            f"seed={row.get('seed', '')}",
        ]
    )


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_leaderboard_csv(path: Path, rows: list[dict]):
    fieldnames = [
        "experiment_id",
        "task",
        "scoring_mode",
        "dataset",
        "dataset_total",
        "samples_used",
        "sample_selection",
        "num_shots",
        "max_new_tokens",
        "batch_size",
        "seed",
        "decoding",
        "temperature",
        "dtype",
        "device",
        "method",
        "status",
        "score_pct",
        "prompt_strict_pct",
        "inst_strict_pct",
        "prompt_loose_pct",
        "inst_loose_pct",
        "avg_prompt_tokens",
        "max_prompt_tokens",
        "avg_lm_head",
        "avg_e2e",
        "tpot",
        "memory_mib",
        "gpu_name",
        "algo_config",
        "output_jsonl",
        "updated_at",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_leaderboard_html(path: Path, rows: list[dict]):
    table = build_pivot_table(rows)
    settings = build_settings_section(rows)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="1">
  <title>Experiment Leaderboard</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4f5f0;
      --ink: #1e211d;
      --muted: #6b6f65;
      --line: #d8dace;
      --head: #e6e9dc;
      --accent: #2a6b6f;
      --surface: #ffffff;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
      line-height: 1.5;
    }}
    main {{
      max-width: 1100px;
      margin: 44px auto;
      padding: 0 28px;
    }}
    h1 {{
      font-size: 24px;
      font-weight: 700;
      letter-spacing: -0.3px;
      margin-bottom: 4px;
    }}
    .meta {{
      color: var(--muted);
      font-size: 12.5px;
      margin-bottom: 28px;
    }}
    .wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 1px 4px rgba(0,0,0,.06);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--surface);
      font-size: 13.5px;
    }}
    th, td {{
      padding: 10px 15px;
      border-bottom: 1px solid var(--line);
      text-align: center;
      white-space: nowrap;
    }}
    th:first-child, td:first-child {{
      text-align: left;
      font-weight: 600;
      padding-left: 18px;
    }}
    th {{
      background: var(--head);
      font-weight: 700;
      font-size: 12px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    tbody tr:last-child td {{ border-bottom: none; }}
    tbody tr:hover td {{ background: #eef1e8; }}
    td.score {{ color: var(--accent); font-weight: 700; }}
    td.empty {{ color: var(--muted); }}
    td.best {{ font-weight: 900; text-decoration: underline; }}
    td.score.best {{ color: var(--accent); font-weight: 900; text-decoration: underline; }}
    .env-pills {{
      display: flex; flex-wrap: wrap; gap: 8px;
      margin: 12px 0 20px;
    }}
    .pill {{
      background: white; border: 1px solid var(--line);
      border-radius: 4px; padding: 3px 10px;
      font-size: 12px; color: var(--muted);
    }}
    .settings-heading {{
      font-size: 16px;
      font-weight: 700;
      margin: 44px 0 16px;
      color: var(--ink);
    }}
    .settings-block {{
      margin-bottom: 28px;
    }}
    .settings-block h3 {{
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .settings-block table {{
      font-size: 12.5px;
    }}
    .settings-block th, .settings-block td {{
      padding: 7px 12px;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Experiment Leaderboard</h1>
    <div class="meta">Generated at {html.escape(generated_at)} &nbsp;·&nbsp; auto-refreshes every 1 s</div>
    <div class="wrap">{table}</div>
    {settings}
  </main>
</body>
</html>
"""
    path.write_text(content, encoding="utf-8")


TASK_DISPLAY_NAMES: dict[str, str] = {
    "arc_challenge": "Arc-Challenge",
    "hellaswag": "HellaSwag",
    "winogrande": "WinoGrande",
    "boolq": "BoolQ",
    "truthfulqa": "TruthfulQA",
    "ifeval": "IFEval",
    "gsm8k": "GSM8K",
}

TASK_ORDER = list(TASK_DISPLAY_NAMES.keys())

METHOD_DISPLAY: dict[str, str] = {
    "Original": "Original",
    "VocabTrim": "VocabTrim",
    "FlashHead": "FlashHead",
    "CFHead": "Ours",
}

METHOD_ORDER = ["Original", "VocabTrim", "FlashHead", "CFHead"]


def _parse_ms(s: str) -> float | None:
    s = s.strip()
    if s.endswith("ms"):
        try:
            return float(s[:-2])
        except ValueError:
            return None
    return None


def _build_latest(rows: list[dict]) -> dict[tuple[str, str], dict]:
    latest: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (row.get("method", ""), row.get("task", ""))
        prev = latest.get(key)
        if prev is None or row.get("updated_at", "") >= prev.get("updated_at", ""):
            latest[key] = row
    return latest


def build_pivot_table(rows: list[dict]) -> str:
    if not rows:
        return "<p>No leaderboard rows yet.</p>"

    present_tasks = [t for t in TASK_ORDER if any(r.get("task") == t for r in rows)]
    extra_tasks = sorted({r.get("task", "") for r in rows} - set(TASK_ORDER) - {""})
    all_tasks = present_tasks + extra_tasks

    present_methods = {r.get("method", "") for r in rows if r.get("method")}
    extra_methods = sorted(present_methods - set(METHOD_ORDER))
    ordered_methods = [m for m in METHOD_ORDER if m in present_methods] + extra_methods

    latest = _build_latest(rows)

    def th(text: str) -> str:
        return f"<th>{html.escape(text)}</th>"

    score_cols = [TASK_DISPLAY_NAMES.get(t, t) for t in all_tasks]
    header_cells = "".join(th(c) for c in ["Method", "Memory (MiB)"] + score_cols + ["LM/sample", "E2E/sample", "TPOT"])
    thead = f"<thead><tr>{header_cells}</tr></thead>"

    def _avg_timing(method: str, timing_field: str) -> float | None:
        vals = [
            v for task in all_tasks
            for v in [_parse_ms((latest.get((method, task)) or {}).get(timing_field, ""))]
            if v is not None
        ]
        return sum(vals) / len(vals) if vals else None

    timing_fields = ("avg_lm_head", "avg_e2e", "tpot")
    method_timing: dict[str, list[float | None]] = {
        m: [_avg_timing(m, f) for f in timing_fields] for m in ordered_methods
    }
    baseline_timing = method_timing.get("Original", [None, None, None])

    # Collect raw values for best-per-column detection
    score_vals: dict[str, list[float | None]] = {t: [] for t in all_tasks}
    mem_vals: list[float | None] = []
    speedup_vals: list[list[float | None]] = [[], [], []]

    for method in ordered_methods:
        # Memory
        sample_row = next((latest.get((method, t)) for t in all_tasks if latest.get((method, t))), None)
        mem = float(sample_row["memory_mib"]) if sample_row and sample_row.get("memory_mib") else None
        mem_vals.append(mem)
        # Scores
        for task in all_tasks:
            r = latest.get((method, task))
            score_vals[task].append(float(r["score_pct"]) if r and r.get("score_pct") else None)
        # Speedup
        m_t = method_timing[method]
        for i, (mv, bv) in enumerate(zip(m_t, baseline_timing)):
            ratio = (bv / mv) if (mv is not None and bv is not None and mv > 0) else None
            speedup_vals[i].append(ratio)

    best_score: dict[str, float | None] = {
        t: max((v for v in score_vals[t] if v is not None), default=None) for t in all_tasks
    }
    best_mem = min((v for v in mem_vals if v is not None), default=None)
    best_speedup = [max((v for v in col if v is not None), default=None) for col in speedup_vals]

    def cell(value: str, is_best: bool, extra_cls: str = "") -> str:
        cls_parts = []
        if extra_cls:
            cls_parts.append(extra_cls)
        if is_best:
            cls_parts.append("best")
        cls = " ".join(cls_parts)
        attr = f' class="{cls}"' if cls else ""
        return f"<td{attr}>{html.escape(str(value))}</td>"

    tbody_rows = []
    for idx, method in enumerate(ordered_methods):
        display = METHOD_DISPLAY.get(method, method)
        cells = [f"<td>{html.escape(display)}</td>"]

        # Memory
        mem = mem_vals[idx]
        if mem is not None:
            cells.append(cell(f"{mem:.1f}", mem == best_mem))
        else:
            cells.append('<td class="empty">—</td>')

        # Score columns
        for t_idx, task in enumerate(all_tasks):
            sv = score_vals[task][idx]
            r = latest.get((method, task))
            if sv is not None:
                cells.append(cell(f"{sv:.2f}%", sv == best_score[task], "score"))
            elif r and r.get("status") == "running":
                cells.append('<td class="empty">…</td>')
            else:
                cells.append('<td class="empty">—</td>')

        # Speedup columns
        m_t = method_timing[method]
        for i, (mv, bv) in enumerate(zip(m_t, baseline_timing)):
            if mv is None:
                cells.append('<td class="empty">—</td>')
            elif bv is None or mv == 0:
                cells.append('<td class="empty">—</td>')
            else:
                ratio = bv / mv
                sp = speedup_vals[i][idx]
                cls = "score" if ratio > 1.0 else ("empty" if ratio < 1.0 else "")
                is_best = (sp is not None and sp == best_speedup[i])
                cells.append(cell(f"{ratio:.2f}×", is_best, cls))

        tbody_rows.append("<tr>" + "".join(cells) + "</tr>")

    tbody = "<tbody>" + "\n".join(tbody_rows) + "</tbody>"
    return f'<table>{thead}{tbody}</table>'


def build_settings_section(rows: list[dict]) -> str:
    if not rows:
        return ""

    latest = _build_latest(rows)
    present_tasks = [t for t in TASK_ORDER if any(r.get("task") == t for r in rows)]
    extra_tasks = sorted({r.get("task", "") for r in rows} - set(TASK_ORDER) - {""})
    all_tasks = present_tasks + extra_tasks

    present_methods = {r.get("method", "") for r in rows if r.get("method")}
    extra_methods = sorted(present_methods - set(METHOD_ORDER))
    ordered_methods = [m for m in METHOD_ORDER if m in present_methods] + extra_methods

    def th(text: str) -> str:
        return f"<th>{html.escape(text)}</th>"

    def td(text: str) -> str:
        return f"<td>{html.escape(str(text))}</td>"

    # --- Algorithm configuration table ---
    algo_repr: dict[str, dict] = {}
    gpu_name = ""
    dtype_str = ""
    for method in ordered_methods:
        for task in all_tasks:
            row = latest.get((method, task))
            if row is None:
                continue
            if not gpu_name:
                gpu_name = row.get("gpu_name", "")
            if not dtype_str:
                dtype_str = row.get("dtype", "")
            if method not in algo_repr and row.get("algo_config"):
                try:
                    algo_repr[method] = json.loads(row["algo_config"])
                except Exception:
                    pass

    # Collect all algo config keys across all methods
    all_config_keys: list[str] = []
    seen_keys: set[str] = set()
    for method in ordered_methods:
        for k in algo_repr.get(method, {}):
            if k not in seen_keys:
                all_config_keys.append(k)
                seen_keys.add(k)

    algo_rows = []
    for method in ordered_methods:
        if method not in algo_repr and not any(latest.get((method, t)) for t in all_tasks):
            continue
        display = METHOD_DISPLAY.get(method, method)
        cfg = algo_repr.get(method, {})
        cells = [td(display)]
        for k in all_config_keys:
            cells.append(td(str(cfg[k]) if k in cfg else "—"))
        algo_rows.append("<tr>" + "".join(cells) + "</tr>")

    algo_section = ""
    if algo_rows and all_config_keys:
        algo_thead = "<thead><tr>" + "".join(th(c) for c in ["Method"] + all_config_keys) + "</tr></thead>"
        algo_tbody = "<tbody>" + "\n".join(algo_rows) + "</tbody>"
        algo_section = (
            f'<div class="settings-block">'
            f'<h3>Algorithm Config</h3>'
            f'<div class="wrap"><table>{algo_thead}{algo_tbody}</table></div>'
            f'</div>'
        )

    # --- Per-task experiment settings table ---
    task_setting_keys = [
        ("scoring_mode", "Scoring"),
        ("num_shots", "Shots"),
        ("samples_used", "Samples"),
        ("max_new_tokens", "Max tokens"),
        ("seed", "Seed"),
    ]
    task_rows = []
    for task in all_tasks:
        rep = next((latest.get((m, task)) for m in ordered_methods if latest.get((m, task))), None)
        if rep is None:
            continue
        task_display = TASK_DISPLAY_NAMES.get(task, task)
        cells = [td(task_display)]
        for key, _ in task_setting_keys:
            cells.append(td(rep.get(key, "—") or "—"))
        task_rows.append("<tr>" + "".join(cells) + "</tr>")

    task_section = ""
    if task_rows:
        task_thead = "<thead><tr>" + "".join(th(c) for c in ["Task"] + [l for _, l in task_setting_keys]) + "</tr></thead>"
        task_tbody = "<tbody>" + "\n".join(task_rows) + "</tbody>"
        task_section = (
            f'<div class="settings-block">'
            f'<h3>Task Settings</h3>'
            f'<div class="wrap"><table>{task_thead}{task_tbody}</table></div>'
            f'</div>'
        )

    if not algo_section and not task_section:
        return ""

    env_pills = ""
    if dtype_str:
        env_pills += f'<span class="pill">dtype: {html.escape(dtype_str)}</span>'
    if gpu_name:
        env_pills += f'<span class="pill">GPU: {html.escape(gpu_name)}</span>'
    env_row = f'<div class="env-pills">{env_pills}</div>' if env_pills else ""

    return (
        f'<h2 class="settings-heading">Experiment Settings</h2>'
        f'{env_row}'
        f'{algo_section}'
        f'{task_section}'
    )



def resolve_dataset_path(task: str, explicit_path: str | None, num_shots: int) -> Path:
    if explicit_path:
        path = Path(explicit_path)
    else:
        path = ROOT_DIR / TASK_DEFAULTS[task]["dataset_path"]

    if not path.exists():
        if explicit_path:
            raise FileNotFoundError(f"Dataset file not found: {path}")
        build_default_dataset(task, path, num_shots)
    elif not explicit_path and dataset_needs_rebuild(task, path, num_shots):
        print(f"[INFO] rebuilding {task} dataset with the current evaluation schema -> {path}")
        build_default_dataset(task, path, num_shots)

    return path


def dataset_needs_rebuild(task: str, path: Path, num_shots: int) -> bool:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if task == "ifeval":
                return False
            return row.get("num_shots") != num_shots or row.get("metric_type") != expected_metric_type(task)
    return True


def format_prompt(tokenizer, row: dict, no_chat_template: bool) -> str:
    prompt = row.get("prompt", row.get("query", ""))
    if row.get("messages") and not no_chat_template and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
    if no_chat_template or not hasattr(tokenizer, "apply_chat_template"):
        return prompt
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_default_dataset(task: str, output_path: Path, num_shots: int):
    print(f"[INFO] dataset not found. downloading/building {task} -> {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    builders = {
        "arc_challenge": build_arc_challenge,
        "hellaswag": build_hellaswag,
        "ifeval": build_ifeval,
        "gsm8k": build_gsm8k,
    }
    rows = builders[task](num_shots)
    write_jsonl(output_path, rows)
    print(f"[INFO] wrote dataset: {output_path} ({len(rows)} samples)")


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_hf_dataset(*args, **kwargs):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Please install Hugging Face datasets: pip install datasets") from exc
    return load_dataset(*args, **kwargs)


def choice_prompt(question: str, choices: list[str]) -> str:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    options = "\n".join(f"{labels[i]}. {choice}" for i, choice in enumerate(choices))
    return f"{question}\n\n{options}\n\nAnswer with only the letter of the correct option."


def make_choice_messages(demos: list[dict], question: str) -> list[dict]:
    messages = []
    for demo in demos:
        messages.append({"role": "user", "content": demo["question"]})
        messages.append({"role": "assistant", "content": demo["answer"]})
    messages.append({"role": "user", "content": question})
    return messages


def make_choice_prompt(demos: list[dict], question: str) -> str:
    parts = []
    for demo in demos:
        parts.append(f"User:\n{demo['question']}\nAssistant: {demo['answer']}")
    parts.append(f"User:\n{question}\nAssistant:")
    return "\n\n".join(parts)


def preprocess_hellaswag(text: str) -> str:
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text


def hellaswag_context(sample: dict) -> str:
    return preprocess_hellaswag(sample["activity_label"] + ": " + sample["ctx_a"] + " " + sample["ctx_b"].capitalize())


def hellaswag_choices(sample: dict) -> list[str]:
    return [preprocess_hellaswag(str(choice)) for choice in sample["endings"]]


def format_hellaswag_question(context: str, choices: list[str]) -> str:
    labels = "ABCD"
    options = "\n".join(f"{labels[idx]}. {choice}" for idx, choice in enumerate(choices))
    return (
        "Choose the most plausible ending for the context.\n\n"
        f"Context: {context}\n{options}\n\n"
        "Answer with only A, B, C, or D."
    )


def format_hellaswag_demo(sample: dict) -> dict:
    target = chr(ord("A") + int(sample["label"]))
    return {
        "question": format_hellaswag_question(hellaswag_context(sample), hellaswag_choices(sample)),
        "answer": target,
    }


def build_arc_challenge(num_shots: int):
    train = load_hf_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    dataset = load_hf_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    demos = []
    for sample in train.select(range(min(num_shots, len(train)))):
        labels = [str(label) for label in sample["choices"]["label"]]
        texts = [str(text) for text in sample["choices"]["text"]]
        answer_key = str(sample["answerKey"])
        target = {label: chr(ord("A") + i) for i, label in enumerate(labels)}.get(answer_key, answer_key)
        demos.append({"question": choice_prompt(sample["question"], texts), "answer": target})

    rows = []
    for idx, sample in enumerate(dataset):
        labels = [str(label) for label in sample["choices"]["label"]]
        texts = [str(text) for text in sample["choices"]["text"]]
        label_to_letter = {label: chr(ord("A") + i) for i, label in enumerate(labels)}
        answer_key = str(sample["answerKey"])
        target = label_to_letter.get(answer_key, answer_key)
        rows.append(
            {
                "task": "arc_challenge",
                "sample_id": sample.get("id", idx),
                "prompt": make_choice_prompt(demos, choice_prompt(sample["question"], texts)),
                "messages": make_choice_messages(demos, choice_prompt(sample["question"], texts)),
                "target": target,
                "answer_type": "choice",
                "metric_type": "fewshot_choice_generation",
                "num_shots": len(demos),
            }
        )
    return rows


def build_hellaswag(num_shots: int):
    train = load_hf_dataset("Rowan/hellaswag", split="train")
    dataset = load_hf_dataset("Rowan/hellaswag", split="validation")
    demos = [format_hellaswag_demo(sample) for sample in train.select(range(min(num_shots, len(train))))]
    rows = []
    for idx, sample in enumerate(dataset):
        context = hellaswag_context(sample)
        choices = hellaswag_choices(sample)
        target_index = int(sample["label"])
        question = format_hellaswag_question(context, choices)
        rows.append(
            {
                "task": "hellaswag",
                "sample_id": sample.get("ind", idx),
                "prompt": make_choice_prompt(demos, question),
                "query": context,
                "messages": make_choice_messages(demos, question),
                "choices": choices,
                "target": chr(ord("A") + target_index),
                "target_index": target_index,
                "answer_type": "choice",
                "metric_type": "fewshot_choice_generation",
                "num_shots": len(demos),
            }
        )
    return rows


def build_ifeval(num_shots: int):
    dataset = load_hf_dataset("google/IFEval", split="train")
    rows = []
    for idx, sample in enumerate(dataset):
        rows.append(
            {
                "task": "ifeval",
                "sample_id": sample.get("prompt_id", idx),
                "prompt": sample["prompt"],
                "target": "",
                "answer_type": "text",
                "metric_type": "generate_until",
                "num_shots": 0,
                "metadata": {
                    "instruction_id_list": sample.get("instruction_id_list", []),
                    "kwargs": sample.get("kwargs", []),
                },
            }
        )
    return rows


def build_gsm8k(num_shots: int):
    train = load_hf_dataset("openai/gsm8k", "main", split="train")
    dataset = load_hf_dataset("openai/gsm8k", "main", split="test")
    demos = [
        {"question": sample["question"], "answer": sample["answer"]}
        for sample in train.select(range(min(num_shots, len(train))))
    ]
    rows = []
    for idx, sample in enumerate(dataset):
        messages = []
        prompt_parts = []
        for demo in demos:
            messages.append({"role": "user", "content": f"Solve the problem. Give the final answer after ####.\n\n{demo['question']}"})
            messages.append({"role": "assistant", "content": demo["answer"]})
            prompt_parts.append(
                f"User:\nSolve the problem. Give the final answer after ####.\n\n{demo['question']}\nAssistant: {demo['answer']}"
            )
        question = f"Solve the problem. Give the final answer after ####.\n\n{sample['question']}"
        messages.append({"role": "user", "content": question})
        prompt_parts.append(f"User:\n{question}\nAssistant:")
        rows.append(
            {
                "task": "gsm8k",
                "sample_id": idx,
                "prompt": "\n\n".join(prompt_parts),
                "messages": messages,
                "target": sample["answer"],
                "answer_type": "gsm8k",
                "metric_type": "fewshot_generation",
                "num_shots": len(demos),
            }
        )
    return rows


if __name__ == "__main__":
    run_task("arc_challenge")
