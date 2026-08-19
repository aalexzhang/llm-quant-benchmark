"""
Usage:
    python run_benchmark.py            # full sweep from config.py
    python run_benchmark.py --quick    # 1 model, 1 quant mode, smoke test
"""

import argparse
import gc
import json
import os
import sys
import traceback

import pandas as pd
import torch
import pynvml

import config as cfg
from src.quantization import load_model_and_tokenizer
from src.benchmark import get_gpu_mem_used_mb, get_gpu_mem_free_mb, run_latency_suite
from src.metrics import compute_perplexity


def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def run_one(model_spec, quant_mode, handle, prompt_lengths, args):
    print(f"\n=== {model_spec.short_name} | {quant_mode} ===")

    free_mb = get_gpu_mem_free_mb(handle)
    if free_mb < cfg.MIN_FREE_VRAM_MB:
        print(f"  skip: only {free_mb:.0f} MB free, below {cfg.MIN_FREE_VRAM_MB} MB floor")
        return None

    row = {
        "model": model_spec.short_name,
        "params_b": model_spec.approx_params_b,
        "quant_mode": quant_mode,
        "status": "ok",
    }

    try:
        mem_before = get_gpu_mem_used_mb(handle)
        model, tokenizer = load_model_and_tokenizer(model_spec.name, quant_mode)
        torch.cuda.synchronize()
        row["load_vram_mb"] = get_gpu_mem_used_mb(handle) - mem_before

        latency_results = run_latency_suite(
            model, tokenizer, prompt_lengths, cfg.MAX_NEW_TOKENS,
            cfg.N_WARMUP_RUNS, cfg.N_TIMED_RUNS,
        )
        row["latency_by_prompt_len"] = latency_results
        row["peak_vram_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)

        if not args.skip_ppl:
            ppl = compute_perplexity(
                model, tokenizer, cfg.PPL_DATASET, cfg.PPL_DATASET_CONFIG,
                cfg.PPL_SPLIT, cfg.PPL_MAX_SAMPLES, cfg.PPL_STRIDE,
            )
            row["perplexity"] = ppl["perplexity"]
            row["ppl_tokens_eval"] = ppl["n_tokens_eval"]

        del model, tokenizer
        free_gpu()

    except torch.cuda.OutOfMemoryError:
        row["status"] = "oom"
        print("  OOM — skipping, clearing cache")
        free_gpu()
    except Exception as e:
        row["status"] = f"error: {e}"
        print(f"  error: {e}")
        traceback.print_exc()
        free_gpu()

    return row


def flatten_latency(row):
    flat = {k: v for k, v in row.items() if k != "latency_by_prompt_len"}
    for entry in row.get("latency_by_prompt_len", []):
        p = entry["prompt_len"]
        flat[f"latency_p{p}_mean_sec"] = entry["mean_latency_sec"]
        flat[f"tps_p{p}_mean"] = entry["mean_tokens_per_sec"]
    return flat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-ppl", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available. This benchmark expects a GPU (GTX 1650 Ti).")
        sys.exit(1)

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024**2):.0f} MB")

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    models = cfg.MODELS[:1] if args.quick else cfg.MODELS
    quant_modes = cfg.QUANT_MODES[:1] if args.quick else cfg.QUANT_MODES
    prompt_lengths = cfg.PROMPT_TOKEN_LENGTHS[:1] if args.quick else cfg.PROMPT_TOKEN_LENGTHS

    all_rows = []
    for model_spec in models:
        for quant_mode in quant_modes:
            torch.cuda.reset_peak_memory_stats()
            row = run_one(model_spec, quant_mode, handle, prompt_lengths, args)
            if row is not None:
                all_rows.append(row)
                with open(cfg.RESULTS_JSON, "w") as f:
                    json.dump(all_rows, f, indent=2)

    df = pd.DataFrame([flatten_latency(r) for r in all_rows])
    df.to_csv(cfg.RESULTS_CSV, index=False)

    print(f"\nDone. Results: {cfg.RESULTS_CSV}, {cfg.RESULTS_JSON}")
    if not df.empty:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()