# LLM Quantization Benchmark For Personal Laptop (GTX 1650 Ti, 4GB VRAM)

Compares fp16 / int8 / int4 (bitsandbytes) for small open LLMs on latency,
throughput, VRAM footprint, and perplexity - sized for a 4GB laptop GPU.

## Setup

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Usage

Smoke test (qwen2.5-0.5b fp16 only):
```bash
python run_benchmark.py --quick
```

Full sweep (all models × fp16/int8/int4):
```bash
python run_benchmark.py
```

## Structure

```
config.py           model list, quant modes, prompt lengths, eval params
src/quantization.py model loading per quant mode (fp16 / bnb int8 / bnb nf4 int4)
src/benchmark.py    latency + tokens/sec measurement
src/metrics.py      perplexity on a wikitext-2 slice
run_benchmark.py    orchestrates the sweep, one model at a time, writes results
results/            benchmark_results.csv and .json (auto-created)
```

## Results

Hardware: GTX 1650 Ti, 4GB VRAM, driver 581.95, CUDA 13.0, Windows/WDDM.

| Model | Params (B) | Quant | Load VRAM (MB) | Peak VRAM (MB) | Perplexity | Latency @128 tok (s) | Tok/s |
|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b | 0.5 | fp16 | 1010 | 957 | 10.46 | 5.39 | 11.87 |
| qwen2.5-0.5b | 0.5 | int8 | 50 | 625 | 10.35 | 16.55 | 3.87 |
| qwen2.5-0.5b | 0.5 | int4 | 30 | 457 | 11.22 | 6.47 | 9.90 |
| tinyllama-1.1b | 1.1 | fp16 | 1208 | 2120 | 6.30 | 9.04 | 7.08 |
| tinyllama-1.1b | 1.1 | int8 | 360 | 1255 | 6.32 | 12.32 | 5.20 |
| tinyllama-1.1b | 1.1 | int4 | 28 | 785 | 6.50 | 5.88 | 10.89 |
| qwen2.5-1.5b | 1.5 | fp16 | 2066 | 2964 | 7.57 | 11.81 | 5.42 |
| qwen2.5-1.5b | 1.5 | int8 | 894 | 1797 | 7.66 | 19.30 | 3.32 |
| qwen2.5-1.5b | 1.5 | int4 | 204 | 1146 | 8.03 | 8.44 | 7.59 |

*Perplexity: wikitext-2-raw-v1, ~2600–3000 tokens evaluated per model, lower is better.*

### Key findings

- Speed ranks fp16 > int4 > int8 on this GPU consistently.

- int4 cuts peak VRAM by ~52–61% vs fp16 consistently.

- Perplexity degrades monotonically (fp16 < int8 < int4) for 1.1B and 1.5B
  models.

- Inference speed does not scale cleanly with parameter count alone.

- GPU utilization during generation is low (~28%) and stays in the P5 power
  state.

## Extending

- Add models: append `ModelSpec(...)` entries in `config.py`.
- Add a quant mode: extend `get_bnb_config()` in `src/quantization.py` and
  add the mode string to `config.QUANT_MODES`.
- Add GPTQ/AWQ instead of bitsandbytes: swap the loader in
  `src/quantization.py` for `AutoGPTQForCausalLM` / `AutoAWQForCausalLM` but
  keep the same `(model, tokenizer)` return contract so `benchmark.py` and
  `metrics.py` don't need changes.