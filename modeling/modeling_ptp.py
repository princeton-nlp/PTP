# Modified from
# PIXEL: https://github.com/xplip/pixel/blob/main/src/pixel/models/pixel/modeling_pixel.py
# Pix2struct: https://github.com/huggingface/transformers/blob/main/src/transformers/models/pix2struct/modeling_pix2struct.py

# coding=utf-8
# Copyright 2023 The HuggingFace Inc. & Google team. All rights reserved.
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
""" Pix2Struct modeling file"""

import math
from typing import Dict, List, Optional, Tuple, Union, Any
import collections

import torch
import torch.utils.checkpoint
from torch import nn
from torch.utils.checkpoint import checkpoint

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ModelOutput,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    SequenceClassifierOutput
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_fx_proxy,
    logging,
    replace_return_docstrings,
)
from .configuration_ptp import PTPConfig
from dataclasses import dataclass
from copy import deepcopy

logger = logging.get_logger(__name__)

import sys
sys.path.append("..")
from image_utils import flattened_patches_to_vit_pixel_values
import numpy as np
from .modeling_pixel import PIXELForPreTraining, PIXELModel, PIXELEncoder, PIXELDecoder
import inspect


@dataclass
class MAEEncoderOutput(BaseModelOutput):
    masked_embedding: Optional[torch.FloatTensor] = None
    ids_restore: Optional[torch.LongTensor] = None
    mask: Optional[torch.Tensor] = None
    encoder_attention_mask: Optional[torch.Tensor] = None

@dataclass
class Seq2SeqMAEOutput(Seq2SeqLMOutput):
    mae_loss: Optional[torch.FloatTensor] = None
    text_loss: Optional[torch.FloatTensor] = None
    mask: Optional[torch.Tensor] = None


# Adapted from transformers.models.t5.modeling_t5.T5LayerNorm with T5->Pix2Struct
class Pix2StructLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=eps)

    def forward(self, hidden_states):
        return self.layer_norm(hidden_states)

ALL_LAYERNORM_LAYERS.append(Pix2StructLayerNorm)


# Copied from transformers.models.t5.modeling_t5.T5DenseGatedActDense with T5DenseGatedActDense->Pix2StructVisionMlp,T5Config->Pix2StructVisionConfig,config.d_model->config.hidden_size,dropout_rate->dropout_rate
class Pix2StructGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wi_0 = nn.Linear(config.hidden_size, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.hidden_size, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)

        # To make 8bit quantization work for google/flan-t5-xxl, self.wo is kept in float32.
        # See https://github.com/huggingface/transformers/issues/20287
        # we also make sure the weights are not in `int8` in case users will force `_keep_in_fp32_modules` to be `None``
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)

        hidden_states = self.wo(hidden_states)
        return hidden_states


class Pix2StructFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wi = nn.Linear(config.hidden_size, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class PTPPreTrainedModel(PreTrainedModel):

    def _init_weights(self, module):
        """
        Initialize the weights
        Largely copied from PIXEL and Pix2Struct 
        """
        factor = self.config.initializer_factor  # Used for testing weights initialization
        
        # ---- from PIXEL ---- #
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        # ---- end from PIXEL ---- #
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        # if isinstance(module, Pix2StructLayerNorm):
        #     module.weight.data.fill_(factor * 1.0)
        elif isinstance(module, Pix2StructGLU):
            hidden_size = (
                self.config.text_config.hidden_size
                if isinstance(self.config, PTPConfig)
                else self.config.hidden_size
            )
            d_ff = self.config.text_config.d_ff if isinstance(self.config, PTPConfig) else self.config.d_ff

            module.wi_0.weight.data.normal_(mean=0.0, std=factor * ((hidden_size) ** -0.5))
            if hasattr(module.wi_0, "bias") and module.wi_0.bias is not None:
                module.wi_0.bias.data.zero_()
            module.wi_1.weight.data.normal_(mean=0.0, std=factor * ((hidden_size) ** -0.5))
            if hasattr(module.wi_1, "bias") and module.wi_1.bias is not None:
                module.wi_1.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, Pix2StructFFN):
            hidden_size = (
                self.config.text_config.hidden_size
                if isinstance(self.config, PTPConfig)
                else self.config.hidden_size
            )
            d_ff = self.config.text_config.d_ff if isinstance(self.config, PTPConfig) else self.config.d_ff

            module.wi.weight.data.normal_(mean=0.0, std=factor * ((hidden_size) ** -0.5))
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, Pix2StructTextAttention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            hidden_size = module.hidden_size
            key_value_proj_dim = module.key_value_proj_dim
            n_heads = module.n_heads # self.config.num_heads if hasattr(self.config, "num_heads") else self.config.num_attention_heads

            module.query.weight.data.normal_(mean=0.0, std=factor * ((hidden_size * key_value_proj_dim) ** -0.5))
            module.key.weight.data.normal_(mean=0.0, std=factor * (hidden_size**-0.5))
            module.value.weight.data.normal_(mean=0.0, std=factor * (hidden_size**-0.5))
            module.output.weight.data.normal_(mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5))
            # if module.has_relative_attention_bias:
            #     module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((hidden_size) ** -0.5))
        elif isinstance(module, nn.Embedding):
            hidden_size = (
                self.config.text_config.hidden_size
                if isinstance(self.config, PTPConfig)
                else self.config.hidden_size
            )

            module.weight.data.normal_(mean=0.0, std=factor * ((hidden_size) ** -0.5))
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, PTPTextModel):
            hidden_size = (
                self.config.text_config.hidden_size
                if isinstance(self.config, PTPConfig)
                else self.config.hidden_size
            )

            module.lm_head.weight.data.normal_(mean=0.0, std=factor * ((hidden_size) ** -0.5))


    # Copied from transformers.models.t5.modeling_t5.T5PreTrainedModel._shift_right with T5->Pix2Struct
    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        if decoder_start_token_id is None:
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined. In Pix2Struct it is usually set to the pad_token_id."
                "See Pix2Struct docs for more information."
            )

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids


class Pix2StructTextLayerFF(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.DenseReluDense = Pix2StructGLU(config) if config.is_glu else Pix2StructFFN(config)

        self.layer_norm = Pix2StructLayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    # Copied from transformers.models.t5.modeling_t5.T5LayerFF.forward
    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class Pix2StructTextAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.query = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.key = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.value = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.output = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.pruned_heads = set()
        self.gradient_checkpointing = False


    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            if len(past_key_value) != 2:
                raise ValueError(
                    f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
                )
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def to_projection_shape(states):
            """projection"""
            return states.contiguous().view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = to_projection_shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = to_projection_shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    # checking that the `sequence_length` of the `past_key_value` is the same as
                    # the provided `key_value_states` to support prefix tuning
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = to_projection_shape(proj_layer(key_value_states))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        # (batch_size, n_heads, seq_length, dim_per_head)
        query_states = to_projection_shape(self.query(hidden_states))

        # get key/value states
        key_states = project(
            hidden_states, self.key, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.value, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            position_bias = torch.zeros(
                (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
            )
            if self.gradient_checkpointing and self.training:
                position_bias.requires_grad = True

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        scores += position_bias_masked
        # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)

        # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = torch.matmul(attn_weights, value_states)
        # (batch_size, seq_length, dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        attn_output = self.output(attn_output)

        present_key_value_state = (key_states, value_states) if use_cache else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


# Copied from transformers.models.t5.modeling_t5.T5LayerSelfAttention with T5LayerNorm->Pix2StructLayerNorm,T5Attention->Pix2StructTextAttention,self.SelfAttention->self.attention,config.d_model->config.hidden_size
class Pix2StructTextLayerSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Pix2StructTextAttention(config)
        self.layer_norm = Pix2StructLayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.attention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.t5.modeling_t5.T5LayerCrossAttention with T5LayerNorm->Pix2StructLayerNorm,T5Attention->Pix2StructTextAttention,self.EncDecAttention->self.attention,config.d_model->config.hidden_size
class Pix2StructTextLayerCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Pix2StructTextAttention(config)
        self.layer_norm = Pix2StructLayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.attention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


class Pix2StructTextBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.self_attention = Pix2StructTextLayerSelfAttention(config)

        self.encoder_decoder_attention = Pix2StructTextLayerCrossAttention(config)

        self.mlp = Pix2StructTextLayerFF(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
    ):
        if past_key_value is not None:
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.self_attention(
            hidden_states,
            attention_mask=attention_mask,
            # position_bias=position_bias, # Disable all position bias
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.encoder_decoder_attention(
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                # position_bias=encoder_decoder_position_bias, # Disable all position bias
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.mlp(hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs



class TextPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, attention_mask, past_key_values_length=0):
        attention_mask = attention_mask.long()

        # create positions depending on attention_mask
        positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1

        # cut positions if `past_key_values_length` is > 0
        positions = positions[:, past_key_values_length:]

        return super().forward(positions + self.offset)



class PTPTextModel(PTPPreTrainedModel):
    _no_split_modules = ["Pix2StructTextBlock"]
    _tied_weights_keys = ["lm_head.weight"]
    supports_gradient_checkpointing = True

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (Pix2StructTextAttention, PTPTextModel)):
            module.gradient_checkpointing = value

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size) # , padding_idx=self.config.pad_token_id)
        self.embed_pos = TextPositionalEmbedding(2048, config.hidden_size)

        # self.layer = nn.ModuleList(
        #     [Pix2StructTextBlock(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        # )
        # In our modification, we add absolute positional embeddings to the decoder and disable all attention positional bias
        # So that we can use flash attention
        self.layer = nn.ModuleList(
            [Pix2StructTextBlock(config) for i in range(config.num_layers)]
        )

        self.embed_layer_norm = Pix2StructLayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.final_layer_norm = Pix2StructLayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        self.gradient_checkpointing = False

    # Copied from transformers.models.t5.modeling_t5.T5PreTrainedModel._reorder_cache
    def _reorder_cache(self, past_key_values, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past_key_values is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past_key_values

        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            if reordered_layer_past_states[0].shape != layer_past_states[0].shape:
                raise ValueError(
                    f"reordered_layer_past_states[0] shape {reordered_layer_past_states[0].shape} and layer_past_states[0] shape {layer_past_states[0].shape} mismatched"
                )
            if len(reordered_layer_past_states) != len(layer_past_states):
                raise ValueError(
                    f"length of reordered_layer_past_states {len(reordered_layer_past_states)} and length of layer_past_states {len(layer_past_states)} mismatched"
                )

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
        return_dict=None,
        **kwargs,
    ):

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        mask_seq_length = past_key_values_length + seq_length 

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        if encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.layer)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions) else None
        position_bias = None
        encoder_decoder_position_bias = None

        pos_embeds = self.embed_pos(attention_mask, past_key_values_length=past_key_values_length)
        inputs_embeds = inputs_embeds + pos_embeds
        if self.config.emb_layer_norm:
            hidden_states = self.embed_layer_norm(inputs_embeds)
        hidden_states = self.dropout(hidden_states)

        for i, (layer_module, past_key_value) in enumerate(zip(self.layer, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
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
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)
                all_cross_attentions = all_cross_attentions + (layer_outputs[3],)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        logits = self.lm_head(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")

            loss = loss_fct(logits.contiguous().view(-1, logits.size(-1)), labels.contiguous().view(-1))

        if not return_dict:
            return tuple(
                v
                for v in [
                    loss,
                    logits,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class PTPForConditionalGeneration(PTPPreTrainedModel):
    config_class = PTPConfig
    main_input_name = "flattened_patches"

    def __init__(self, config):
        super().__init__(config)

        self.encoder = PIXELModel(config.vision_config) 

        if config.add_text_decoder:
            self.decoder = PTPTextModel(config.text_config)

        if config.add_mae_decoder:
            self.mae_decoder = PIXELDecoder(config.vision_config, num_patches=self.encoder.embeddings.num_patches, dtype=self.encoder.dtype)

        self.mask_ratio = 0.0 

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.decoder.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.decoder.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.decoder.set_output_embeddings(new_embeddings)

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> nn.Embedding:
        model_embeds = self.decoder.resize_token_embeddings(new_num_tokens)

        # update vocab size
        self.config.text_config.vocab_size = new_num_tokens

        return model_embeds

    def get_decoder(self):
        return self.decoder

    def get_encoder(self):
        return self.encoder

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        flattened_patches = model_kwargs.pop("flattened_patches")
        if isinstance(self.config.vision_config.patch_size, collections.abc.Iterable):
            ph, pw = self.config.vision_config.patch_size
        else:
            pw = ph = self.config.vision_config.patch_size

        # flattened_patches (bsz, npatches, ph*pw*ch) -> pixel_values (bsz, ch, H, W)

        model_kwargs["pixel_values"] = flattened_patches_to_vit_pixel_values(
            flattened_patches[:, :, 2:],  # remove the coordinates
            height=self.config.vision_config.image_size[0], 
            width=self.config.vision_config.image_size[1], 
            patch_height=ph, patch_width=pw, 
            image_mode=getattr(self.config.vision_config, 'image_mode', 'RGB')
        ) 

        # 1. get encoder
        encoder = self.get_encoder()

        # 2. Prepare encoder args and encoder kwargs from model kwargs.
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        encoder_signature = set(inspect.signature(encoder.forward).parameters)
        encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        if not encoder_accepts_wildcard:
            encoder_kwargs = {
                argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature
            }

        # 3. make sure that encoder returns `ModelOutput`
        encoder_kwargs["return_dict"] = True
        if getattr(self.config, "text_encoder", False):
            encoder_kwargs["input_ids"] = inputs_tensor
        model_kwargs["encoder_outputs"] = encoder(**encoder_kwargs)

        return model_kwargs

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W) x: (N, L, patch_size**2 *3)
        """
        ph, pw = self.encoder.embeddings.patch_embeddings.patch_size
        assert imgs.shape[2] % ph == 0 and imgs.shape[3] % pw == 0
        c = 3 if self.config.image_mode == "RGB" else 1
        h = imgs.shape[2] // ph
        w = imgs.shape[3] // pw
        x = imgs.reshape(shape=(imgs.shape[0], c, h, ph, w, pw))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, ph * pw * c))

        return x


    def mae_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W] pred: [N, L, p*p*3] mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.config.vision_config.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
    

    def forward(
        self,
        flattened_patches: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        patch_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqModelOutput]:

        use_cache = use_cache if use_cache is not None else self.config.text_config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Note that in the current implementation, 
        # For the vision attention, we pass on 0/1 attention mask and inside the vision attention we convert it to -inf/0
        # For the text attention, we convert them at the very beginning in the text_decoder

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Patch size
            if isinstance(self.config.vision_config.patch_size, collections.abc.Iterable):
                ph, pw = self.config.vision_config.patch_size
            else:
                pw = ph = self.config.vision_config.patch_size

            # flattened_patches (bsz, npatches, ph*pw*ch) -> pixel_values (bsz, ch, H, W)
            pixel_values = flattened_patches_to_vit_pixel_values(
                flattened_patches[:, :, 2:], # Remove the coordinates
                height=self.config.vision_config.image_size[0], 
                width=self.config.vision_config.image_size[1], 
                patch_height=ph, 
                patch_width=pw, 
                image_mode=getattr(self.config.vision_config, 'image_mode', 'RGB')
            ) 

            encoder_outputs = self.encoder(
                pixel_values,
                attention_mask=attention_mask,
                head_mask=head_mask,
                patch_mask=patch_mask, # patch masks generated from the data processing code
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        hidden_states = encoder_outputs[0]

        # Handle the text decoder inputs
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)
            decoder_attention_mask = (
                decoder_attention_mask
                if decoder_attention_mask is not None
                else decoder_input_ids.ne(self.config.pad_token_id).float()
            )
            # Always attend to the first token
            decoder_attention_mask[:, 0] = 1
        if labels is not None:
            labels[labels == self.config.pad_token_id] = -100

        # Text decoder
        if self.config.add_text_decoder:
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                past_key_values=past_key_values,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=encoder_outputs.encoder_attention_mask, 
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                labels=labels,
                return_dict=return_dict,
            )
        else:
            decoder_outputs = CausalLMOutputWithCrossAttentions(loss=0.0, logits=None)

        # Vision decoder (only use when there is masking)
        if self.mask_ratio > 0 and self.config.add_mae_decoder:
            # Add MAE
            latent = encoder_outputs.last_hidden_state
            ids_restore = encoder_outputs.ids_restore
            mask = encoder_outputs.mask

            mae_decoder_outputs = self.mae_decoder(latent, ids_restore, attention_mask)  # [N, L, p*p*3]
            logits = mae_decoder_outputs.logits
            merged_mask = torch.bitwise_and(mask == 1, attention_mask == 1).long()
            mae_loss = self.mae_loss(pixel_values, logits, merged_mask)
          
            return Seq2SeqMAEOutput(
                loss=decoder_outputs.loss * self.text_weight + mae_loss * self.mae_weight,
                text_loss=decoder_outputs.loss * self.text_weight,
                mae_loss=mae_loss * self.mae_weight,
                mask=mask,
                logits=mae_decoder_outputs.logits,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )
        else:
            assert hasattr(self, "decoder"), "Must have MAE decoder when there is no text decoder"

            if not return_dict:
                return decoder_outputs + encoder_outputs

            return Seq2SeqLMOutput(
                loss=decoder_outputs.loss,
                logits=decoder_outputs.logits,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        flattened_patches: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        past_key_values=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        if decoder_attention_mask is None:
            decoder_attention_mask = torch.ones_like(input_ids).to(input_ids.device)

        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "flattened_patches": flattened_patches,
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }


class PTPClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, n_classes: int = 2, avg_pooling = False):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        
        classifier_dropout = (
            config.classifier_dropout if (hasattr(config, \
                "classifier_dropout") and config.classifier_dropout is \
                not None) else config.dropout_rate
        )
        self.dropout = nn.Dropout(classifier_dropout)
        
        num_labels = n_classes
        
        self.out_proj = nn.Linear(config.hidden_size, num_labels)
        self.avg_pooling = avg_pooling

    def forward(self, features, **kwargs):
        if self.avg_pooling:
            x = features.mean(1)
        else:
            x = features[:, 0, :] 
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class PTPForSequenceClassification(PTPPreTrainedModel):
    main_input_name = "flattened_patches"

    def __init__(self, config, n_classes: int = 2, avg_pooling = False):
        super().__init__(config)

        self.n_classes = n_classes

        self.encoder = PIXELModel(config.vision_config)
        self.cls_head = PTPClassificationHead(config.vision_config, n_classes, avg_pooling)

        # Initialize weights and apply final processing
        self.post_init()


    def get_cls_head(self):
        return self.cls_head

    def get_encoder(self):
        return self.encoder

    def forward(
        self,
        flattened_patches: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], SequenceClassifierOutput]:
        use_cache = use_cache if use_cache is not None else \
            self.config.text_config.use_cache
        return_dict = return_dict if return_dict is not None else \
            self.config.use_return_dict
        
        if isinstance(self.config.vision_config.patch_size, collections.abc.Iterable):
            ph, pw = self.config.vision_config.patch_size
        else:
            pw = ph = self.config.vision_config.patch_size
        pixel_values = flattened_patches_to_vit_pixel_values(
            flattened_patches[:, :, 2:], 
            height=self.config.vision_config.image_size[0], 
            width=self.config.vision_config.image_size[1], 
            patch_height=ph, 
            patch_width=pw, 
            image_mode=getattr(self.vision_config, 'image_mode', 'RGB')
        ) 

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                pixel_values,
                attention_mask=attention_mask,
                head_mask=head_mask,
                patch_mask=None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if \
                    len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if \
                    len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]        

        logits = self.cls_head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.contiguous().view(-1), labels.contiguous().view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss(reduction="mean")
                loss = loss_fct(logits.contiguous().view(-1, logits.size(-1)), labels.contiguous().view(-1))

        if not return_dict:
            return tuple(v for v in [loss, logits] if v is \
                not None) + encoder_outputs
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions
        )


######### Flash Attention ########

# from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func

def text_forward_flash_attn(
    self,
    hidden_states,
    mask=None,
    key_value_states=None,
    position_bias=None,
    past_key_value=None,
    layer_head_mask=None,
    query_length=None,
    use_cache=False,
    output_attentions=False,
):
    """
    Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
    """
    # Input is (batch_size, seq_length, dim)
    # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
    # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
    batch_size, seq_length = hidden_states.shape[:2]

    real_seq_length = seq_length

    if past_key_value is not None:
        if len(past_key_value) != 2:
            raise ValueError(
                f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            )
        real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

    key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

    def to_projection_shape(states):
        """projection"""
        return states.contiguous().view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

    def project(hidden_states, proj_layer, key_value_states, past_key_value):
        """projects hidden states correctly to key/query states"""
        if key_value_states is None:
            # self-attn
            # (batch_size, n_heads, seq_length, dim_per_head)
            hidden_states = to_projection_shape(proj_layer(hidden_states))
        elif past_key_value is None:
            # cross-attn
            # (batch_size, n_heads, seq_length, dim_per_head)
            hidden_states = to_projection_shape(proj_layer(key_value_states))

        if past_key_value is not None:
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, key_length, dim_per_head)
                hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
            elif past_key_value.shape[2] != key_value_states.shape[1]:
                # checking that the `sequence_length` of the `past_key_value` is the same as
                # the provided `key_value_states` to support prefix tuning
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = to_projection_shape(proj_layer(key_value_states))
            else:
                # cross-attn
                hidden_states = past_key_value
        return hidden_states

    # get query states
    # (batch_size, n_heads, seq_length, dim_per_head)
    query_states = to_projection_shape(self.query(hidden_states))
    # Because torch's flash attention has a fixed scale 1/sqrt(qdim), we need to scale the query states back
    query_states = query_states * (self.key_value_proj_dim ** 0.5)

    # get key/value states
    key_states = project(
        hidden_states, self.key, key_value_states, past_key_value[0] if past_key_value is not None else None
    )
    value_states = project(
        hidden_states, self.value, key_value_states, past_key_value[1] if past_key_value is not None else None
    )

    # if key_value_states is None:
    #     # self-attention, should be causal
    #     causal = True
    # else:
    #     # cross-attention, should not be causal
    #     causal = False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states, key_states, value_states,
        attn_mask=mask, # in our text (decoder) models, attention_mask is -inf/0
        is_causal=False, # causal is set by the attention mask
        dropout_p=self.dropout if self.training else 0,
    )
    attn_output = attn_output.view(batch_size, self.n_heads, seq_length, -1)
    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(batch_size, seq_length, -1)
    attn_output = self.output(attn_output)

    present_key_value_state = (key_states, value_states) if use_cache else None
    outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

    # if output_attentions:
    #     outputs = outputs + (attn_weights,)
    return outputs


def find_module(root_module: nn.Module, key: str):
    """From OpenDelta"""
    sub_keys = key.split(".")
    parent_module = root_module
    for sub_key in sub_keys[:-1]:
        parent_module = getattr(parent_module, sub_key)
    module = getattr(parent_module, sub_keys[-1])
    return parent_module, sub_keys[-1], module


def inject_flash_attention_ptp(model):
    for key, _ in model.named_modules():
        attention_name = "attention"
        exclude_name = "_attention"

        if key[-len(attention_name):] == attention_name and key[-len(exclude_name):] != exclude_name:
            _, _, attn = find_module(model, key)
            if "decoder" in key and not "mae_decoder" in key:
                # if "encoder_decoder" not in key:
                #     print("Does not support " + key + " with positional bias, skip")
                #     continue
                print("Inject text decoder flash attn:", key)
                attn.original_forward = attn.forward
                attn.forward = text_forward_flash_attn.__get__(attn, type(attn))