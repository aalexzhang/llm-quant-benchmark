"""Perplexity on a small wikitext slice — quality signal across quant levels."""

import torch
from datasets import load_dataset


@torch.inference_mode()
def compute_perplexity(model, tokenizer, dataset_name, dataset_config, split,
                        max_samples, stride, device="cuda"):
    ds = load_dataset(dataset_name, dataset_config, split=split)
    text = "\n\n".join(ds["text"][:max_samples])
    if not text.strip():
        return {"perplexity": float("nan"), "n_tokens_eval": 0}

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    seq_len = input_ids.shape[1]

    max_len = getattr(model.config, "max_position_embeddings", 2048) or 2048
    max_len = min(max_len, 2048)

    nlls = []
    n_tokens = 0
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + max_len, seq_len)
        trg_len = end - prev_end
        ids = input_ids[:, begin:end]
        target_ids = ids.clone()
        target_ids[:, :-trg_len] = -100

        outputs = model(ids, labels=target_ids)
        nlls.append(outputs.loss * trg_len)
        n_tokens += trg_len
        prev_end = end
        if end == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / n_tokens)
    return {"perplexity": ppl.item(), "n_tokens_eval": n_tokens}