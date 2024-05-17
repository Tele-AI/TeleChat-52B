# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# This code is based on OpenAI's GPT-2 library. It has been modified from its
# original forms to accommodate architectural differences compared to GPT-2.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch TELECHAT model."""

from typing import Optional, Tuple, Union

import math
import torch
from einops import rearrange
from torch import einsum, nn
from torch.cuda.amp import autocast
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_conv1d_layer
from transformers.utils import logging
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_func # flashattn1
    print("# FLASH ATTENTION 1 DETECTED #")
except ImportError:
    try:
        from flash_attn.flash_attn_interface import flash_attn_varlen_func as flash_attn_unpadded_func # flashattn2
        print("# FLASH ATTENTION 2 DETECTED #")
    except ImportError:
        print("# NO FLASH ATTENTION DETECTED #")
        flash_attn_unpadded_func = None
from .configuration_telechat import TELECHATConfig


def debug_print_tensor(t, name, title='', show_dim=10):
    # return
    prefix = f'{title} -> '
    if isinstance(t, torch.Tensor):
        if len(t.shape) == 1:
            output = f"{name}[{t.shape}]: {t[:show_dim]}"
        elif len(t.shape) == 2:
            output = f"{name}[{t.shape}]: {t[-1, :show_dim]}"
        elif len(t.shape) == 3:
            output = f" {name}[{t.shape}]: {t[-1, -1, :show_dim]}"
        elif len(t.shape) == 4:
            output = f"{name}[{t.shape}]: {t[-1, -1, -1, :show_dim]}"
        else:
            output = f"{name}[{t.shape}]"
    elif isinstance(t, list):
        output = f"{name} [{len(t)}]: {t[:show_dim]}"
    else:
        output = f"{name} 未知类型: {type(t)}"
    print(prefix + output)



class Conv1D(nn.Module):

    def __init__(self, nf, nx, bias=True):
        super().__init__()
        self.nf = nf
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        if self.bias is not None:
            return torch.matmul(x, self.weight) + self.bias
        else:
            return torch.matmul(x, self.weight)



class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


logger = logging.get_logger(__name__)


def exists(v):
    return v is not None


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, use_xpos=False, xpos_scale_base=512, theta=10000):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.cache = dict()
        self.cache_scale = dict()
        self.use_xpos = use_xpos
        if not use_xpos:
            self.register_buffer('scale', None)
            return
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer('scale', scale)
        self.scale_base = xpos_scale_base

    def forward(self, seq, cache_key=None):

        if cache_key is not None and cache_key in self.cache:
            return self.cache[cache_key]

        inv_freq = self.inv_freq.to(device=seq.device)
        freqs = einsum('i , j -> i j', seq, inv_freq)
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        scale = torch.cat((freqs, freqs), dim=-1)
        if exists(cache_key):
            self.cache[cache_key] = scale
        return scale

    def rotate_queries_and_keys(self, q, k, seq_dim=-2):
        """
        use this only when xpos is activated.
        """
        assert self.use_xpos and q.device == k.device
        device, seq_len_k, seq_len_q = k.device, k.shape[seq_dim], q.shape[seq_dim]
        pos_seq_k = torch.arange(seq_len_k, device=device, dtype=torch.float32)
        pos_seq_q = torch.arange(seq_len_k - seq_len_q, seq_len_k, device=device, dtype=torch.float32)
        freqs_k = self.forward(pos_seq_k, cache_key=f"{0}:{seq_len_k}")
        freqs_q = self.forward(pos_seq_q, cache_key=f"{seq_len_k - seq_len_q}:{seq_len_k}")
        scale_k = self.get_scale(pos_seq_k)
        scale_q = self.get_scale(pos_seq_q, offset=seq_len_k - seq_len_q)  # 这里的offset是Q相对于K的offset
        rotated_q = apply_rotary_emb(freqs_q, q, scale=scale_q)
        rotated_k = apply_rotary_emb(freqs_k, k, scale=scale_k ** -1)
        return rotated_q, rotated_k

    def rotate_queries_or_keys(self, t, seq_dim=-2, offset=0):
        """
        use this only when xpos is NOT activated.
        """
        # t's shape e.g.  -> (batchsize, headnum, seqlen, dimofhead)
        assert not self.use_xpos, 'you must use `.rotate_queries_and_keys` method instead and pass in both queries and keys, for length extrapolatable rotary embeddings'
        device, seq_len = t.device, t.shape[seq_dim]
        pos_seq_t = torch.arange(offset, offset + seq_len, device=device, dtype=torch.float32)
        freqs = self.forward(pos_seq_t, cache_key=f"{offset}:{offset+seq_len}")
        # freqs   seqlen  x  dim
        return apply_rotary_emb(freqs, t)

    def get_scale(self, t, cache_key=None, offset=0, ):
        assert self.use_xpos, 'This function is only useful for xpos.'
        if exists(cache_key) and cache_key in self.cache_scale:
            return self.cache_scale[cache_key]
        if callable(t):
            t = t()
        length = len(t)
        min_pos = -(length + offset) // 2
        max_pos = length + offset + min_pos
        power = torch.arange(min_pos, max_pos, 1).to(device=self.scale.device) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = scale[-length:, :]
        scale = torch.cat((scale, scale), dim=-1)
        if exists(cache_key):
            self.cache_scale[cache_key] = scale
        return scale


def rotate_half(x):
    """
    change sign so the last dimension becomes [-odd, +even]
    """
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(freqs, t, start_index=0, scale=1.):
    """
    freq: seqlen  x  dim
       t: [batchsize  *  headnum  ,  seqlen  , dim (dim_of_head actually)]
    """
    dtype_t = t.dtype
    freqs = freqs.to(device=t.device)
    if isinstance(scale, torch.Tensor):
        scale = scale.to(device=t.device)
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * freqs.cos() + rotate_half(t) * freqs.sin()) * scale
    rotated = torch.cat((t_left, t, t_right), dim=-1)
    rotated = rotated.to(dtype=dtype_t)
    return rotated


class TELECHATAttention(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        # for alignment with megatron-lm in softmax scale
        self.layer_idx = max(1, layer_idx)
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        self.relative_encoding = config.relative_encoding
        self.rotary_use_xpos = config.rotary_use_xpos

        self.use_mup = config.use_mup

        self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim, bias=config.add_bias_linear)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim, bias=config.add_bias_linear)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.pruned_heads = set()

        self.use_flash_attn = False



    def set_max_positions(self, max_positions, device='cuda'):
        self.max_positions = max_positions
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((self.max_positions, self.max_positions), dtype=torch.bool)).view(
                1, 1, self.max_positions, self.max_positions
            ).to(device=device)
        )

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.num_heads, self.head_dim, self.pruned_heads)
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.num_heads) * (self.num_heads - len(heads))
        self.num_heads = self.num_heads - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # (batch, head, seq_length, head_features)
        # batch_size, head_num, k_seq_len(q_seq_len), head_features
        batch_size, head_num, k_seq_len, head_features = key.shape
        _, _, q_seq_len, _ = query.shape

        if self.use_flash_attn:
            # print("*")
            # attn_output = torch.nn.functional._scaled_dot_product_attention(query, key, value, is_causal=True)
            # attn_weights = None
            # return attn_output, attn_weights

            batch_size, seqlen_q = query.shape[0], query.shape[2]
            seqlen_k = key.shape[2]

            query, key, value = [rearrange(x, 'b h s ... -> (b s) h ...') for x in [query, key, value]]
            cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
                                        device=query.device)
            is_causal = seqlen_q == seqlen_k
            cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32,
                                        device=query.device)
            dropout_p = 0

            softmax_scale = 1/torch.full([], (value.size(-1) ** 0.5), dtype=value.dtype, device=value.device) if self.scale_attn_weights else 1
            attn_output = flash_attn_unpadded_func(
                query, key, value, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k,
                dropout_p,
                softmax_scale=softmax_scale, causal=is_causal
            )
            attn_output = rearrange(attn_output, '(b s) h ... -> b h s ...', b=batch_size)
            attn_weights = None
            return attn_output, attn_weights

        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            if self.use_mup:
                attn_weights = attn_weights / torch.full(
                    [], value.size(-1) / (value.size(-1) ** 0.5), dtype=attn_weights.dtype,
                    device=attn_weights.device
                )
            else:
                attn_weights = attn_weights / torch.full(
                    [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
                )

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None, head_mask=None):
        # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
        bsz, num_heads, q_seq_len, dk = query.size()
        _, _, k_seq_len, _ = key.size()

        # Preallocate attn_weights for `baddbmm`
        attn_weights = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=query.dtype, device=query.device)

        # Compute Scale Factor
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** 0.5

        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx)
        # Upcast (turn off autocast) and reorder (Scale K by 1 / root(dk))
        with autocast(enabled=False):
            q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
            attn_weights = torch.baddbmm(attn_weights, q, k, beta=0, alpha=scale_factor)
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

        if not self.is_cross_attention:
            attn_weights = attn_weights.float()
            if self.scale_attn_by_inverse_layer_idx:
                attn_weights *= self.layer_idx
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length]
            mask_value = -10000.0  # align with megatron-lm
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
        if attn_weights.dtype != torch.float32:
            raise RuntimeError("Error with upcasting, attn_weights does not have dtype torch.float32")
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
            self,
            hidden_states: Optional[Tuple[torch.FloatTensor]],
            layer_past: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            rotary_embedding: Optional[RotaryEmbedding] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        batch_size, head_num, k_seq_len, head_features = key.shape
        _, _, q_seq_len, _ = query.shape
        query_offset = k_seq_len - q_seq_len
        if rotary_embedding is not None:
            query = query.contiguous().view(batch_size * head_num, q_seq_len, head_features)
            key = key.contiguous().view(batch_size * head_num, k_seq_len, head_features)

            # batch_size * head_num,  k_seq_len(q_seq_len), head_features
            if self.rotary_use_xpos:
                # query: [batch_size * head_num, seqlen, hn]
                query, key = rotary_embedding.rotate_queries_and_keys(query, key)
            else:
                query = rotary_embedding.rotate_queries_or_keys(query, offset=query_offset)
                key = rotary_embedding.rotate_queries_or_keys(key)
            # batch_size * head_num, k_seq_len(q_seq_len), head_features
            query = query.view(batch_size, head_num, q_seq_len, head_features)
            key = key.view(batch_size, head_num, k_seq_len, head_features)

        if self.reorder_and_upcast_attn and not self.use_flash_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class TELECHATMLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        if config.activation_function=='silu':
            up_intermediate_size = 2 * intermediate_size
        else:
            up_intermediate_size = intermediate_size
        self.c_fc = Conv1D(up_intermediate_size, embed_dim, bias=config.add_bias_linear)
        self.c_proj = Conv1D(embed_dim, intermediate_size, bias=config.add_bias_linear)
        if config.activation_function=='silu':
            def swiglu(x):
                x = torch.chunk(x, 2, dim=-1)
                return F.silu(x[0]) * x[1]
            self.act = swiglu
        else:
            self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        # print(f'activation func: {self.act}')
        # print(f'before act: hidden_states {hidden_states.shape}')
        hidden_states = self.act(hidden_states)
        # print(f'after  act: hidden_states {hidden_states.shape}')
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class TELECHATBlock(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        LayerNorm = nn.LayerNorm if not config.use_RMSNorm else RMSNorm
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        self.layer_idx = layer_idx
        self.ln_1 = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = TELECHATAttention(config, layer_idx=layer_idx)
        self.ln_2 = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = TELECHATMLP(inner_dim, config)

    def forward(
            self,
            hidden_states: Optional[Tuple[torch.FloatTensor]],
            layer_past: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            rotary_embedding: Optional[RotaryEmbedding] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        # debug_print_tensor(hidden_states, 'after ln_1')
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            rotary_embedding=rotary_embedding,
            use_cache=use_cache,
            output_attentions=output_attentions
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states
        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]
        # debug_print_tensor(hidden_states, 'block output')

        return outputs


class TELECHATPretrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = TELECHATConfig
    load_tf_weights = None
    base_model_prefix = "transformer"
    is_parallelizable = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["TELECHATBlock"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm) or isinstance(module, RMSNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name == "c_proj.weight":
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.data.normal_(mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.n_layer)))

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, TELECHATTransformer):
            module.gradient_checkpointing = value


class TELECHATTransformer(TELECHATPretrainedModel):
    _keys_to_ignore_on_load_missing = ["attn.masked_bias"]

    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        self.relative_encoding = config.relative_encoding
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)

        self.use_mup = config.use_mup
        if self.use_mup:
            self.input_mult = config.input_mult

        if self.relative_encoding is None:
            self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        elif self.relative_encoding == 'rotary':
            pe_dim = config.n_embd // config.n_head
            self.wpe = RotaryEmbedding(pe_dim,
                                       use_xpos=config.rotary_use_xpos,
                                       xpos_scale_base=config.rotary_xpos_scale_base,
                                       theta=config.rotary_theta
                                       )

        else:
            raise RuntimeError(
                f'Unknown relative positional encoding type: `relative_encoding`={self.relative_encoding}')
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([TELECHATBlock(config, layer_idx=i + 1) for i in range(config.num_hidden_layers)])
        LayerNorm = nn.LayerNorm if not config.use_RMSNorm else RMSNorm
        self.ln_f = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    # @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.h), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.h))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        self.wte = self.wte.to(self.first_device)
        self.wpe = self.wpe.to(self.first_device)
        # Load onto devices
        for k, v in self.device_map.items():
            for block in v:
                cuda_device = "cuda:" + str(k)
                self.h[block] = self.h[block].to(cuda_device)
        # ln_f to last
        self.ln_f = self.ln_f.to(self.last_device)

    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        self.wte = self.wte.to("cpu")
        self.wpe = self.wpe.to("cpu")
        for index in range(len(self.h)):
            self.h[index] = self.h[index].to("cpu")
        self.ln_f = self.ln_f.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        # if self.config.add_cross_attention and encoder_hidden_states is not None:
        #     encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        #     encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        #     if encoder_attention_mask is None:
        #         encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
        #     encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        # else:
        #     encoder_attention_mask = None
        encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        # Mup
        if self.use_mup:
            inputs_embeds = inputs_embeds * self.input_mult
        if self.relative_encoding is None:
            position_embeds = self.wpe(position_ids)
            hidden_states = inputs_embeds + position_embeds
        elif self.relative_encoding == 'rotary':
            hidden_states = inputs_embeds
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds
        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        # debug_print_tensor(hidden_states, 'after embedding')
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    rotary_embedding=self.wpe if self.relative_encoding == 'rotary' else None,
                    use_cache=use_cache,
                    output_attentions=output_attentions
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                # if self.config.add_cross_attention:
                #     all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class TELECHAT(TELECHATPretrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = TELECHATTransformer(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.use_mup = config.use_mup
        if self.use_mup:
            self.mup_scale_factor = config.mup_scale_factor
            self.output_mult = config.output_mult / self.mup_scale_factor
        
        # 初始化时先根据config里的开关决定是否开启flashattn, 用户可以通过修改config或者model.enable_flash_attn修改flashattn的开关
        self.enable_flash_attn(config.enable_flash_attn)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()
    def enable_flash_attn(self, enabled: bool):
        for block in self.transformer.h:
            block.attn.use_flash_attn = enabled
        print(f"TELECHAT flash attention {'enabled' if enabled else 'disabled'}")
        # torch.backends.cuda.enable_flash_sdp(enabled)
    def set_max_positions(self, max_positions):
        for layer in self.transformer.h:
            device = layer.ln_1.weight.device
            layer.attn.set_max_positions(max_positions, device=device)

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.transformer.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions, SequenceClassifierOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)
        # Mup
        if self.use_mup:
            lm_logits = lm_logits * self.output_mult

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    def chat(self,tokenizer, question, history_input_list, history_output_list,generation_config):
            '''
            :param question: 当前问题
            :param history_input_list: 历史问题列表, list of strings
            :param history_output_list: 历史回答列表, list of string
            :return: response
            '''

            inputs = ""
            assert len(history_output_list) == len(history_output_list)
            for i in range(len(history_input_list)):
                inputs += "<_user>" + history_input_list[i] + "<_bot>" + history_output_list[i] + "<_end>"
            inputs += "<_user>" + question + "<_bot>"
            print("input:", inputs)
            input_ids = tokenizer.encode(inputs,
                                         return_tensors="pt"
                                         )
            if len(input_ids[0]) >= 2000:
                input_ids = input_ids[:, -2000:]
            input_ids = input_ids.to(0)
            output = self.generate(input_ids,generation_config)
            response = tokenizer.decode(output[0].cpu().numpy().tolist()).split('<_bot>')[-1].split('</s>')[0]
            return response

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )
