"""Latency / throughput / VRAM measurement for a loaded model."""

import time
import torch
import pynvml


def get_gpu_mem_used_mb(handle) -> float:
    return pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024 ** 2)


def get_gpu_mem_free_mb(handle) -> float:
    return pynvml.nvmlDeviceGetMemoryInfo(handle).free / (1024 ** 2)


def make_prompt(tokenizer, target_len: int) -> str:
    filler = "The quick brown fox jumps over the lazy dog. "
    text = filler * (target_len // 8 + 1)
    ids = tokenizer(text, return_tensors="pt").input_ids[0][:target_len]
    return tokenizer.decode(ids, skip_special_tokens=True)


@torch.inference_mode()
def time_generation(model, tokenizer, prompt: str, max_new_tokens: int, device="cuda"):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[1]

    torch.cuda.synchronize()
    start = time.perf_counter()

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    new_tokens = out.shape[1] - input_len
    tok_per_sec = new_tokens / elapsed if elapsed > 0 else float("nan")
    return {
        "input_tokens": input_len,
        "new_tokens": new_tokens,
        "elapsed_sec": elapsed,
        "tokens_per_sec": tok_per_sec,
    }


def run_latency_suite(model, tokenizer, prompt_lengths, max_new_tokens,
                       n_warmup, n_timed, device="cuda"):
    results = []
    for plen in prompt_lengths:
        prompt = make_prompt(tokenizer, plen)

        for _ in range(n_warmup):
            time_generation(model, tokenizer, prompt, max_new_tokens, device)

        runs = [time_generation(model, tokenizer, prompt, max_new_tokens, device)
                for _ in range(n_timed)]

        latencies = [r["elapsed_sec"] for r in runs]
        tps = [r["tokens_per_sec"] for r in runs]

        results.append({
            "prompt_len": plen,
            "mean_latency_sec": sum(latencies) / len(latencies),
            "min_latency_sec": min(latencies),
            "max_latency_sec": max(latencies),
            "mean_tokens_per_sec": sum(tps) / len(tps),
        })
    return results