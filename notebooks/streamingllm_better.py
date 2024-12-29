#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import math
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_pg19_book(index=0):
    """
    加载 emozilla/pg19 数据集的 test split 中第 index 本书的完整文本，
    不做字符截断。
    """
    dataset = load_dataset("emozilla/pg19", split="test", trust_remote_code=True)
    text = dataset[index]["text"]
    return text

def tokenize_text(tokenizer, text, max_tokens=2048):
    """
    对整段文本做分词。若分词结果超出 max_tokens，则在 token 层面截断。
    """
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    print(f"[tokenize_text] Original token count = {len(input_ids)}")
    if len(input_ids) > max_tokens:
        input_ids = input_ids[:max_tokens]
        print(f"[tokenize_text] Truncated to max_tokens = {max_tokens}")
    else:
        print("[tokenize_text] No truncation required.")
    return input_ids

def slice_kv_cache(past_key_values, mode, window_size, anchor_size, step_idx):
    """
    在 'window'/'streaming' 模式下，截断 past_key_values 中过旧的KV。

    past_key_values 的结构:
      tuple( #layers ) of ( key, value )
    其中 key, value shape 类似:
      [batch_size, n_heads, seq_len, head_dim]

    - 'dense': 不截断
    - 'window': 只保留最近 window_size (seq_len > window_size 时截断)
    - 'streaming': 保留前 anchor_size + 最近 window_size (简化实现)
    - 'sliding': 不在此处理，因为 sliding 每步都不使用过去缓存
    """
    if past_key_values is None:
        return None

    if mode == "dense":
        return past_key_values

    new_pkv = []
    for layer_past in past_key_values:
        k, v = layer_past  # [bs, n_heads, seq_len, head_dim]
        seq_len = k.size(2)

        if mode == "window":
            # 只保留最后 window_size 个
            if seq_len > window_size:
                k = k[:, :, -window_size:, :]
                v = v[:, :, -window_size:, :]

        elif mode == "streaming":
            # 保留最后 (anchor_size + window_size)
            keep_len = anchor_size + window_size
            if seq_len > keep_len:
                k = k[:, :, -keep_len:, :]
                v = v[:, :, -keep_len:, :]

        new_pkv.append((k, v))
    return tuple(new_pkv)

def compute_ppl_with_kv(
    model,
    tokenizer,
    input_ids,
    device="cuda",
    mode="dense",
    window_size=512,
    anchor_size=128
):
    """
    使用 KV Cache 的 teacher-forcing 方式评估 PPL:
      PPL = exp( - (1/(T-1)) * ∑ log P( x_{i+1} | x_<=i ) )
    跳过第一个 token，因为它没有前文。

    mode: "dense"/"window"/"sliding"/"streaming"
      - dense:   保留全部 KV
      - window:  仅保留最近 window_size
      - sliding: 每步都不使用 past_key_values (相当于重算)
      - streaming: 保留 anchor_size + window_size
    """
    print(f"\n[compute_ppl_with_kv] mode={mode}, window_size={window_size}, anchor_size={anchor_size}")

    n_tokens = len(input_ids)
    if n_tokens < 2:
        print("[Warning] Less than 2 tokens, cannot compute PPL.")
        return float('nan')

    nll = 0.0
    count = 0
    batch_size = 1
    past_key_values = None

    # 初始输入只放第一个 token
    current_input = torch.tensor([input_ids[0]], device=device).view(batch_size, -1)

    with torch.no_grad():
        for i in range(n_tokens - 1):
            outputs = model(
                current_input,
                past_key_values=past_key_values,
                use_cache=True
            )
            next_token_logits = outputs.logits[:, -1, :]  # shape [1, vocab_size]

            gold_token_id = input_ids[i + 1]
            log_probs = torch.log_softmax(next_token_logits, dim=-1)
            token_log_prob = log_probs[0, gold_token_id].item()

            nll -= token_log_prob
            count += 1

            # sliding模式 => 每步都不用 past_key_values
            if mode == "sliding":
                past_key_values = None
            else:
                past_key_values = outputs.past_key_values

            # window / streaming => 截断
            if mode in ["window", "streaming"]:
                past_key_values = slice_kv_cache(past_key_values, mode, window_size, anchor_size, step_idx=i)

            # 下一步输入 => 刚才的 gold token
            current_input = torch.tensor([gold_token_id], device=device).view(batch_size, -1)

    ppl = math.exp(nll / count)
    print(f"[compute_ppl_with_kv] => PPL = {ppl:.2f}")
    return ppl

def main():
    """
    加载 pg19, 取 test[index], 不做字符截断，直接分词后根据 max_tokens 截断。
    然后分别用 dense/window/sliding/streaming 计算 PPL。
    """
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading tokenizer & model = {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

    # 加载 PG19 第 index=0 本书完整文本
    text = load_pg19_book(index=0)
    print(f"[main] Loaded PG19 test[0], text length={len(text)} chars")

    # 分词并截断到 max_tokens=2048
    input_ids = tokenize_text(tokenizer, text, max_tokens=2048)
    print(f"[main] Final token count={len(input_ids)}")

    # (a) dense
    ppl_dense = compute_ppl_with_kv(model, tokenizer, input_ids, device=device, mode="dense")
    # (b) window
    window_size = 256
    ppl_window = compute_ppl_with_kv(model, tokenizer, input_ids, device=device, mode="window", window_size=window_size)
    # (c) sliding
    # ppl_sliding = compute_ppl_with_kv(model, tokenizer, input_ids, device=device, mode="sliding", window_size=window_size)
    # (d) streaming
    anchor_size = 32
    ppl_streaming = compute_ppl_with_kv(model, tokenizer, input_ids, device=device,
                                        mode="streaming", window_size=window_size, anchor_size=anchor_size)

    print("\n=== Summary of results ===")
    print(f"(a) Dense PPL: {ppl_dense:.2f}")
    print(f"(b) Window (size={window_size}) PPL: {ppl_window:.2f}")
    # print(f"(c) Sliding (size={window_size}) PPL: {ppl_sliding:.2f}")
    print(f"(d) Streaming (anchor={anchor_size}, window={window_size}) PPL: {ppl_streaming:.2f}")


if __name__ == "__main__":
    main()
