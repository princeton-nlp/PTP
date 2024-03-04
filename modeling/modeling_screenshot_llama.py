import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import numpy as np
from typing import Optional, Tuple, Union, List
from dataclasses import dataclass
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast

from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaPreTrainedModel, LlamaModel

from transformers.utils import logging
from transformers.activations import ACT2FN
from transformers.models.llama.modeling_llama import _expand_mask
import copy

logger = logging.get_logger(__name__)


# def prepare_bidirectional_decoder_attn_mask(attention_mask, input_shape, inputs_embeds):
#     # create causal mask
#     # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
#     expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
#         inputs_embeds.device
#     )
#     return expanded_attn_mask


@dataclass
class ScreenshotCausalLMOutputWithPast(CausalLMOutputWithPast):

    pixel_loss: Optional[torch.FloatTensor] = None
    text_loss: Optional[torch.FloatTensor] = None
    patch_logits: Optional[torch.FloatTensor] = None


class LlamaScreenshotModel(LlamaModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config):
        super().__init__(config)

        self.patch_projection = nn.Linear(config.patch_embed_size, config.hidden_size, bias=True)
        if getattr(config, "add_input_mlp", False):
            self.input_patch_mlp_or_identity = LlamaEmbedderMLP(config)
        else:
            self.input_patch_mlp_or_identity = nn.Identity()

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        flattened_patches: Optional[torch.Tensor] = None, 
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        # Patch embedding
        if flattened_patches is not None:
            patch_embeds = self.patch_projection(flattened_patches) # (B, PL, Pemb) -> (B, PL, H)
            patch_embeds = self.input_patch_mlp_or_identity(patch_embeds) # (B, PL, H)
            patch_mask = input_ids == self.config.patch_token_id # (B, L)
            patch_mask = patch_mask.unsqueeze(-1).expand_as(inputs_embeds) # (B, L, H)
            if inputs_embeds.dtype != patch_embeds.dtype:
                inputs_embeds = inputs_embeds.to(patch_embeds.dtype)
            inputs_embeds[patch_mask] = patch_embeds.reshape(-1)

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
            padding_mask = None
        else:
            if 0 in attention_mask:
                padding_mask = attention_mask
            else:
                padding_mask = None
        
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, past_key_value, output_attentions, padding_mask=padding_mask)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer), hidden_states, attention_mask, position_ids
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    padding_mask=padding_mask,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaEmbedderMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        if hasattr(config, "intermediate_emb_size") and config.intermediate_emb_size is not None:
            self.intermediate_size = config.intermediate_emb_size
        elif hasattr(config, "multiplicative_factor") and config.multiplicative_factor is not None:
            multiplicative_factor = config.multiplicative_factor
            self.intermediate_size = multiplicative_factor * config.hidden_size
        else:
            multiplicative_factor = config.vocab_size // config.hidden_size        
            self.intermediate_size = multiplicative_factor * config.hidden_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class LlamaForScreenshot(LlamaPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.model = LlamaScreenshotModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        if getattr(config, "add_output_mlp", False):
            self.output_patch_mlp_or_identity = LlamaEmbedderMLP(config)
        else:
            self.output_patch_mlp_or_identity = nn.Identity()
        
        if getattr(config, "pixel_decoder", False):
            print("Add a transformer PIXEL decoder!")
            self.pixel_decoder = LlamaScreenshotModel(config.pixel_decoder_config)
            self.encoder_to_decoder_proj = nn.Linear(config.hidden_size, config.pixel_decoder_config.hidden_size, bias=True)
            self.patch_head = nn.Linear(config.pixel_decoder_config.hidden_size, config.patch_embed_size, bias=True)
        else:
            self.patch_head = nn.Linear(config.hidden_size, config.patch_embed_size, bias=True)
        
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def pixel_loss(self, target, pred, mask):
        # target and pred: (bsz * npatches, pH*pW*3)
        if self.config.norm_pix_loss:
            # note that this is not the same as the pix2struct normalization
            # this is normalization within patches (same as the original MAE/PIXEL loss)
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5
 
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [b * PL]
        loss = (loss * mask).sum() / mask.sum()  # mean loss on patches that have attn=1

        return loss

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        flattened_patches: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if flattened_patches is not None:
            # Remove the coordinates
            flattened_patches = flattened_patches[:, :, 2:]

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            flattened_patches=flattened_patches,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if flattened_patches is not None:
            shift_hidden_states = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous() 
            shift_attention_mask = attention_mask[..., 1:].contiguous()

            patch_mask = shift_labels == self.config.patch_token_id
            patch_hidden_states = shift_hidden_states[patch_mask] # (B*PL, H)
            patch_attention_mask = shift_attention_mask[patch_mask] # (B*PL)
            patch_logits_intermediate = self.output_patch_mlp_or_identity(patch_hidden_states) 

            if getattr(self.config, "pixel_decoder", False):
                batch_size = input_ids.shape[0]
                patch_logits_intermediate = self.encoder_to_decoder_proj(patch_logits_intermediate)
                pixel_decoder_outputs = self.pixel_decoder(
                    inputs_embeds=patch_logits_intermediate.reshape(batch_size, -1, patch_logits_intermediate.shape[-1]),
                    attention_mask=patch_attention_mask.reshape(batch_size, -1),
                )
                pixel_decoder_hidden = pixel_decoder_outputs[0].reshape(-1, pixel_decoder_outputs[0].shape[-1]) # (B*PL, H)
                patch_logits = self.patch_head(pixel_decoder_hidden)
            else:
                patch_logits = self.patch_head(patch_logits_intermediate) # (B*PL, H) -> (B*PL, Pemb)

            pixel_loss = self.pixel_loss(flattened_patches.view(-1, flattened_patches.size(-1)), patch_logits, patch_attention_mask)
            loss = pixel_loss * self.ar_pixel_weight

            # for visualization later
            patch_logits[patch_attention_mask == 0] = 0 # those parts are not used for calculating loss, but they might have huge values that mess up the visualization
            patch_logits = patch_logits.reshape(flattened_patches.shape) 
        else:
            pixel_loss = 0
            patch_logits = None

        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask = attention_mask[..., 1:].contiguous()
            # Ignore the image tokens and the unmasked tokens
            shift_labels[shift_labels == self.config.patch_token_id] = -100
            shift_labels[shift_labels == self.config.img_begin_token_id] = -100
            shift_labels[shift_labels == self.config.img_end_token_id] = -100
            # We still let the model predict \n

            shift_labels[~shift_attention_mask.bool()] = -100

            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)

            text_loss = loss_fct(shift_logits, shift_labels)
            loss = text_loss * self.ar_text_weight if loss is None else loss + text_loss * self.ar_text_weight
        else:
            text_loss = 0

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return ScreenshotCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            pixel_loss=pixel_loss,
            text_loss=text_loss,
            patch_logits=patch_logits,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past

from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

def llama_forward_flash_attn(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    # [bsz, nh, t, hd]

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states, key_states, value_states,
        attn_mask=attention_mask, # 0 or -inf
        is_causal=False,
        dropout_p=0 # llama has no dropout
    )

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def find_module(root_module: nn.Module, key: str):
    """From OpenDelta"""
    sub_keys = key.split(".")
    parent_module = root_module
    for sub_key in sub_keys[:-1]:
        parent_module = getattr(parent_module, sub_key)
    module = getattr(parent_module, sub_keys[-1])
    return parent_module, sub_keys[-1], module


def inject_flash_attention_screenshotllama(model):
    for key, _ in model.named_modules():
        attention_name = "self_attn"

        if key[-len(attention_name):] == attention_name:
            _, _, attn = find_module(model, key)
            print("Inject LLaMA flash attn:", key)
            attn.original_forward = attn.forward
            attn.forward = llama_forward_flash_attn.__get__(attn, type(attn))