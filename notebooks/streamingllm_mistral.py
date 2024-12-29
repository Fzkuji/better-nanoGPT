#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from transformers import LlamaModel


def prepare_model_and_tokenizer(model_name: str, load_in_8bit: bool = False):
    """
    加载 Llama-2-13B 模型和 tokenizer。
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # torch_dtype=torch.float16 if not load_in_8bit else None,
        # load_in_8bit=load_in_8bit,
        device_map="auto",
        # attn_implementation="eager",
    )
    return tokenizer, model


def load_pg19_index_book(index: int = 2048):
    """
    加载 deepmind/pg19 数据集的 test split 中的第一个样本（一本书），
    返回其中的前 max_chars 个字符（不是 token），仅做演示用。
    """
    dataset = load_dataset("emozilla/pg19", split="test", trust_remote_code=True)
    text = dataset[index]['text']
    return text


def tokenize_text(tokenizer, text: str, max_tokens: int = 2048):
    """
    对文本做分词，返回 input_ids（不添加特殊token）。
    """
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    print(f"Total tokens: {len(input_ids)}")
    if len(input_ids) > max_tokens:
        input_ids = input_ids[:max_tokens]
    return input_ids


import torch


def build_4d_mask(seq_len, mode="dense", window_size=512, anchor_size=128):
    """
    构造形状为 [1, 1, seq_len, seq_len] 的加性注意力掩码，加到注意力scores上：
      - mask[i,j] = 0.0   表示允许在预测第 i+1 个 token 时 attend 第 j+1 个 token
      - mask[i,j] = -∞    表示屏蔽该位置

    默认先做因果屏蔽：j > i => -∞
    再根据 mode:
      - dense:       不额外裁剪 (保留 [0..i])
      - window:      只保留 [i-window_size .. i]
      - sliding:     同 window（概念相同）
      - streaming:   保留前 anchor_size 个 token + 最近 window_size 个 token
                     (其余位置也置为 -∞)
    """
    # float16 下的负无穷
    NEG_INF = torch.finfo(torch.float16).min

    # 先创建全 0 矩阵
    mask_2d = torch.zeros((seq_len, seq_len), dtype=torch.float16)

    # 准备索引: idx_i[i,j] = i, idx_j[i,j] = j
    idx_i = torch.arange(seq_len).view(-1, 1)  # shape [seq_len, 1]
    idx_j = torch.arange(seq_len).view(1, -1)  # shape [1, seq_len]

    # 1) 因果屏蔽：禁止访问未来 (j > i => -∞)
    mask_2d[idx_j > idx_i] = NEG_INF

    # 2) 根据模式进一步裁剪“太旧的”令其 -∞
    if mode in ["window", "sliding"]:
        # j < i - window_size => -∞
        mask_2d[idx_j < (idx_i - window_size)] = NEG_INF

    elif mode == "streaming":
        # 对 streaming 来说，保留:
        #   - anchor: [0..anchor_size-1] (只要 j <= i)
        #   - window: [i - window_size .. i]
        # 其余位置 -∞

        # 先把整个 2D 矩阵都设成 -∞
        mask_2d[:] = NEG_INF

        # 允许的条件: j <= i (因果) 并且 (j < anchor_size 或 j >= i - window_size)
        # 用布尔张量一次性算出
        anchor_cond = (idx_j < anchor_size)
        window_cond = (idx_j >= (idx_i - window_size))
        causal_cond = (idx_j <= idx_i)

        allowed = causal_cond & (anchor_cond | window_cond)
        mask_2d[allowed] = 0.0

    elif mode == "dense":
        # 什么都不做, 只保留因果遮罩
        pass
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 升维到 [1, 1, seq_len, seq_len]
    return mask_2d.unsqueeze(0).unsqueeze(0)

def compute_ppl_onepass(model, tokenizer, input_ids, device, mode="dense",
                        window_size=512, anchor_size=128):
    """
    在一次 forward 中，通过定制 [1, 1, seq_len, seq_len] 的 attention_mask，
    来模拟 Dense / Window / Sliding / Streaming 四种注意力方式。

    然后从输出 logits 里依次取出第 i 个 token 的对数概率 (给定前 i-1 个token)。
    """
    seq_len = len(input_ids)
    # 构造4D mask
    # attn_mask_4d = build_4d_mask(seq_len, mode=mode, window_size=window_size, anchor_size=anchor_size).to(device)

    # 构造模型输入 shape: [batch_size=1, seq_len]
    input_ids_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    # forward
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids_tensor,
            # attention_mask=attn_mask_4d
        )
        # logits shape => [1, seq_len, vocab_size]
        logits = outputs.logits

    # 计算PPL： perplexity = exp( - (1/T) * sum_{t=1..T} log P(x_t | x_<t) )
    # 注意：logits[:, i, :] 通常对应“当已知前 i+1 个token时，对下一个token的预测”
    # 但是还有一个 off-by-one shift：transformers 通常把 "logits[i]" 视作“token i 的预测”。
    # 对于 GPT 类CausalLM，logits[:, i] 是在看到前 i 个 token 后预测第 i+1 个 token。
    # => 第 i 个 token 的概率要从 logits[:, i-1] 里取，i从1开始计数。
    nll = 0.0
    count = 0

    # 我们要计算 x_1 的概率吗？通常第一个token没有前文，就略过，从第二个token开始
    for t in range(1, seq_len):
        # t 位置的真实token是 input_ids[t]
        gold_token_id = input_ids[t]
        # logits for position t is at index t-1
        # shape [vocab_size]
        pred_logits = logits[0, t - 1, :]
        log_probs = torch.log_softmax(pred_logits, dim=-1)
        token_log_prob = log_probs[gold_token_id].item()

        nll -= token_log_prob
        count += 1

    ppl = math.exp(nll / count)
    return ppl


def main():
    """
    演示主函数：一次性前向 + 4D attention_mask 来区分 Dense/Window/Sliding/Streaming。
    """
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"  # 或本地的 checkpoint (e.g., "meta-llama/Llama-2-7b-chat-hf")
    tokenizer, model = prepare_model_and_tokenizer(model_name, load_in_8bit=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 扩展model的max_position_embeddings
    model.config.sliding_window = 4096

    # 1) 取 PG19 的一个简短文本
    raw_text = load_pg19_index_book(index=0)
    input_ids = tokenize_text(tokenizer, raw_text, max_tokens=16384)
    seq_len = len(input_ids)
    print(f"Total tokens: {seq_len}")

    # 2) 分别构造并计算四种模式
    print("=== (a) Dense Attention ===")
    ppl_dense = compute_ppl_onepass(model, tokenizer, input_ids, device, mode="dense")
    print(f"Perplexity (Dense): {ppl_dense:.2f}\n")

    # window_size = 4096
    # #
    # print("=== (b) Sliding Window ===")
    # ppl_sliding = compute_ppl_onepass(model, tokenizer, input_ids, device,
    #                                   mode="sliding", window_size=window_size)
    # print(f"Perplexity (Sliding, size={window_size}): {ppl_sliding:.2f}\n")

    # anchor_size = 10
    # print("=== (c) StreamingLLM (Anchor+Window) ===")
    # ppl_streaming = compute_ppl_onepass(model, tokenizer, input_ids, device,
    #                                     mode="streaming", window_size=window_size, anchor_size=anchor_size)
    # print(f"Perplexity (Streaming, anchor={anchor_size}, window={window_size}): {ppl_streaming:.2f}\n")


if __name__ == "__main__":
    main()
