"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers.utils import logging

from utils import RMSNorm
import warnings

from typing import List, Optional, Tuple, Union
from transformers import Qwen2Config
logger = logging.get_logger(__name__)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Qwen2
class Qwen2RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[Qwen2Config] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            logger.warning_once(
                "`Qwen2RotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in v4.46"
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed



def get_relative_positions(seq_len: int) -> torch.tensor:
    x = torch.arange(seq_len)[None, :]
    y = torch.arange(seq_len)[:, None]
    return x - y


def get_alibi_slope(num_heads):
    x = (2 ** 8) ** (1 / num_heads)
    return (
        torch.tensor([1 / x ** (i + 1) for i in range(num_heads)])
        .unsqueeze(-1)
        .unsqueeze(-1)
    )


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.block_size = config.block_size
        self.position_embedding = config.position_embedding
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        if self.position_embedding == 'rope':
            # 初始化 RoPE 位置编码
            self.rotary_emb = Qwen2RotaryEmbedding(dim=self.head_dim, max_position_embeddings=config.max_position_embeddings)
        elif self.position_embedding == 'alibi':
            self.register_buffer("m", get_alibi_slope(self.n_head))
            self.input_length = 0
        else:
            pass

    def forward(
        self,
        x,
        position_ids,
        past_key_values=None,
        use_cache=False,
        bias: Optional[torch.Tensor] = None,
    ):
        B, T, C = x.size()

        # 计算查询、键、值
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 如果有 past_key_values，需要更新 position_ids
        if past_key_values is not None:
            # past_key_values 的长度
            past_length = past_key_values[0].size(2)
            # 拼接后的总长度
            total_length = past_length + T
            # 更新 position_ids
            position_ids = torch.arange(self.global_position - total_length, self.global_position, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(0).expand(B, -1)
        else:
            total_length = T

        if self.position_embedding == 'rope':
            # 应用 RoPE 位置编码
            cos, sin = self.rotary_emb(q, position_ids=position_ids)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
        elif self.position_embedding == 'alibi':
            if self.input_length != total_length:
                position = (self.m * get_relative_positions(total_length).to(x.device)).unsqueeze(0)
            self.input_length = total_length

        # 拼接 past_key_values
        if use_cache and past_key_values is not None:
            past_keys, past_values = past_key_values
            k = torch.cat((past_keys, k), dim=2)
            v = torch.cat((past_values, v), dim=2)

        # 更新 present
        if use_cache:
            present = (k[:, :, -self.block_size:], v[:, :, -self.block_size:])
        else:
            present = None

        # 计算注意力
        # 注意，这里的 bias 尺寸应为 [1, 1, total_length, total_length]
        if self.flash and (self.position_embedding != 'alibi'):
            print("using flash attention")
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=bias,
                dropout_p=self.dropout if self.training else 0,
                is_causal=False
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if self.position_embedding == 'alibi':
                att = att + position
            att = att.masked_fill(bias == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, present


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, position_ids, past_key_values=None, use_cache=False, bias=None):
        attn_output, present = self.attn(
            self.ln_1(x),
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            bias=bias,
        )
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return x, present


@dataclass
class GPTConfig:
    block_size: int = 1024
    position_embedding: str = 'rope'  # rope or None
    max_position_embeddings: int = 32768
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        # create the sliding window mask
        # causal mask to ensure that attention is only applied to the left in the input sequence
        mask = torch.tril(torch.ones(config.max_position_embeddings, config.max_position_embeddings, dtype=torch.int8))

        # apply sliding window to the mask
        for i in range(config.max_position_embeddings):
            mask[i, :max(0, i - self.config.block_size + 1)] = 0  # Set values outside the window to 0
        mask = mask.bool() if self.flash else mask  # 将bias转换为torch.bool类型

        self.register_buffer("bias", mask.view(1, 1, config.max_position_embeddings, config.max_position_embeddings))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

        self.global_position = 0

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, past_key_values=None):

        # --------------------------------------------------------------
        # Check if the input sequence length exceeds the model's memory
        # capacity and issue a warning if necessary, indicating potential
        # information loss due to overflow.
        # --------------------------------------------------------------
        b, t = idx.size()
        warnings.warn(
            f"Input sequence length {t} exceeds the model's memory capacity of {(self.config.block_size - 1) * self.config.n_layer + 1}, which may lead to unavoidable information loss.",
            UserWarning
        ) if t > (self.config.block_size - 1) * self.config.n_layer + 1 else None


        # 计算过去的长度
        if past_key_values is not None and len(past_key_values) > 0:
            past_length = past_key_values[0][0].size(2)
        else:
            past_length = 0

        # 计算总的序列长度，包括 past_key_values
        total_length = past_length + t
        assert total_length <= self.config.max_position_embeddings, f"Cannot forward, total sequence length {total_length} exceeded max position embeddings {self.config.max_position_embeddings}"

        # 计算当前序列的 position_ids
        position_ids = torch.arange(self.global_position, self.global_position + t, dtype=torch.long, device=idx.device)
        position_ids = position_ids.unsqueeze(0).expand_as(idx)

        # 更新全局位置
        self.global_position += t

        # --------------------------------------------------------------
        # Forward pass through the GPT model.
        # 1. Embed the input tokens to obtain token embeddings of shape (b, t, n_embd).
        # 2. Process the embeddings through each transformer block in sequence.
        # 3. If caching is enabled (i.e., targets is None), store the key-value pairs
        #    for each block in `presents` to accelerate future computations.
        # --------------------------------------------------------------
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        presents = []
        for i, block in enumerate(self.transformer.h):
            past = past_key_values[i] if past_key_values is not None else None
            x, present = block(
                x,
                position_ids=position_ids,
                past_key_values=past,
                use_cache=True if targets is None else False,
                bias=self.bias[:, :, :total_length, :total_length]
            )
            if targets is None:
                presents.append(present)
        x = self.transformer.ln_f(x)

        # 初始化 segment_loss 列表
        segment_loss = []

        if targets is not None:
            # 计算逐点损失 (未聚合到总损失)
            logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)
            pointwise_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # 展平 logits 维度为 (batch_size * seq_len, vocab_size)
                targets.view(-1),  # 展平 targets 维度为 (batch_size * seq_len,)
                ignore_index=-1,
                reduction='none'  # 不进行聚合，返回逐点损失
            ).view(logits.size(0), logits.size(1))  # 恢复形状为 (batch_size, seq_len)

            # 计算整体的平均损失
            loss = pointwise_loss.mean()

        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss, segment_loss, presents if targets is None else None

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257  # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints
        config_args['bias'] = True  # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        # sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # discard this mask / buffer, not a param
        sd_keys = [k for k in sd_keys if not k.endswith('.bias')]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them

        # --------------------------------------------------------------
        # Compare keys between the custom model and HuggingFace model:
        # 1. Identify missing keys in each model.
        # 2. Warn if the number of keys differs, indicating potential mismatches.
        # --------------------------------------------------------------

        # # output differences between the two models
        # print("missing keys in HF model:")
        # for k in sd_keys:
        #     if k not in sd_keys_hf:
        #         print(k)
        # print("missing keys in our model:")
        # for k in sd_keys_hf:
        #     if k not in sd_keys:
        #         print(k)

        # assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        if len(sd_keys_hf) != len(sd_keys):
            warnings.warn(f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}, pls double check using the code above")

        # --------------------------------------------------------------
        # Transfer weights from HuggingFace model to our custom model:
        # 1. For Conv1D weights (which are stored in a 1D convolutional format in HF),
        #    we transpose them to match the expected shape for our model.
        # 2. For other parameters, we perform a direct copy to align weights.
        # --------------------------------------------------------------

        for k in sd_keys_hf:
            if k in sd_keys:  # Only copy if the key exists in both models
                if any(k.endswith(w) for w in transposed):
                    # Special treatment for Conv1D weights that need to be transposed
                    assert sd_hf[k].shape[::-1] == sd[k].shape  # Ensure shapes are compatible after transpose
                    with torch.no_grad():
                        sd[k].copy_(sd_hf[k].t())  # Transpose and copy the weights
                else:
                    # Standard copy for other parameters
                    assert sd_hf[k].shape == sd[k].shape  # Ensure shapes match for direct copy
                    with torch.no_grad():
                        sd[k].copy_(sd_hf[k])  # Copy the weights directly
            else:
                # Skip parameters in HF model that don't have a matching name in our model
                continue

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        past_key_values = None
        idx_next = idx
        for _ in range(max_new_tokens):
            # forward the model to get the logits for the index in the sequence
            logits, _, past_key_values = self(idx_next, past_key_values=past_key_values)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
