"""
Central config for the quantization benchmark.
Hardware target: GTX 1650 Ti, 4GB VRAM.
"""

from dataclasses import dataclass


@dataclass
class ModelSpec:
    name: str            # HF repo id
    short_name: str      # label used in results
    approx_params_b: float


MODELS = [
    ModelSpec("Qwen/Qwen2.5-0.5B-Instruct", "qwen2.5-0.5b", 0.5),
    ModelSpec("Qwen/Qwen2.5-1.5B-Instruct", "qwen2.5-1.5b", 1.5),
    ModelSpec("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "tinyllama-1.1b", 1.1),
    # ModelSpec("microsoft/phi-2", "phi-2-2.7b", 2.7),  # try after fp16 OOM check
]

QUANT_MODES = ["fp16", "int8", "int4"]

BATCH_SIZES = [1]
MAX_NEW_TOKENS = 64
PROMPT_TOKEN_LENGTHS = [128]
N_WARMUP_RUNS = 1
N_TIMED_RUNS = 3

PPL_DATASET = "Salesforce/wikitext"
PPL_DATASET_CONFIG = "wikitext-2-raw-v1"
PPL_SPLIT = "test"
PPL_MAX_SAMPLES = 50
PPL_STRIDE = 512

MIN_FREE_VRAM_MB = 300

RESULTS_DIR = "results"
RESULTS_CSV = "results/benchmark_results.csv"
RESULTS_JSON = "results/benchmark_results.json"