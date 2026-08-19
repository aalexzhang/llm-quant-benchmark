"""
Load a HF causal LM under a given quantization mode.
fp16 -> plain half precision
int8 -> bitsandbytes LLM.int8()
int4 -> bitsandbytes nf4, double quant on
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def get_bnb_config(mode: str):
    if mode == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    if mode == "int4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    return None


def load_model_and_tokenizer(model_name: str, mode: str, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = get_bnb_config(mode)

    load_kwargs = dict(
        pretrained_model_name_or_path=model_name,
        device_map={"": 0} if device == "cuda" else None,
        dtype=torch.float16,
    )

    if mode != "fp16":
        load_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
    model.eval()
    return model, tokenizer