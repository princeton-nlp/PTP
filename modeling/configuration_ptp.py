# Modified from https://github.com/huggingface/transformers/blob/main/src/transformers/models/pix2struct/configuration_pix2struct.py

# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
""" Pix2Struct model configuration"""

import copy
import os
from typing import Union

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from .configuration_pixel import PIXELConfig


logger = logging.get_logger(__name__)


class PTPTextConfig(PretrainedConfig):

    model_type = "ptp_text_decoder"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }

    def __init__(
        self,
        vocab_size=50244,
        hidden_size=768,
        d_kv=64,
        d_ff=2048,
        num_layers=12,
        num_heads=12,
        dropout_rate=0.1,
        attention_dropout=0.1,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        dense_act_fn="gelu_new",
        decoder_start_token_id=0,
        use_cache=False,
        pad_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=True,
        is_decoder=True,
        emb_layer_norm=True,
        is_glu=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.layer_norm_eps = self.layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.use_cache = use_cache

        self.eos_token_id = eos_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.emb_layer_norm = emb_layer_norm
        self.is_glu = is_glu

        # for backwards compatibility
        self.dense_act_fn = dense_act_fn

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            tie_word_embeddings=tie_word_embeddings,
            is_decoder=is_decoder,
            **kwargs,
        )

    @classmethod
    def from_pretrained(
        cls, pretrainehidden_size_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrainehidden_size_name_or_path, **kwargs)

        # get the text config dict if we are loading from Pix2StructConfig
        if config_dict.get("model_type") == "ptp":
            config_dict = config_dict["text_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)



class PTPVisionConfig(PretrainedConfig):

    model_type = "ptp_vision"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        image_size=(16, 8192),
        patch_size=16,
        num_channels=3,
        qkv_bias=True,
        decoder_num_attention_heads=16,
        decoder_hidden_size=512,
        decoder_num_hidden_layers=8,
        decoder_intermediate_size=2048,
        norm_pix_loss=True,
        embedding_layernorm=False,
        image_mode="RGB",
        **kwargs
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.decoder_num_attention_heads = decoder_num_attention_heads
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_num_hidden_layers = decoder_num_hidden_layers
        self.decoder_intermediate_size = decoder_intermediate_size
        self.norm_pix_loss = norm_pix_loss
        self.embedding_layernorm = embedding_layernorm
        self.image_mode = image_mode

    @classmethod
    def from_pretrained(
        cls, pretrainehidden_size_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrainehidden_size_name_or_path, **kwargs)

        # get the text config dict if we are loading from Pix2StructConfig
        if config_dict.get("model_type") == "ptp":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class PTPConfig(PretrainedConfig):

    model_type = "ptp"
    is_composition = True

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        tie_word_embeddings=True,
        is_encoder_decoder=True,
        add_mae_decoder=True,
        add_text_decoder=True,
        initializer_factor=1.0,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, is_encoder_decoder=is_encoder_decoder, **kwargs)

        if text_config is None:
            text_config = {}
            logger.info("text_config is None. Initializing the Pix2StructTextConfig with default values.")

        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. Initializing the Pix2StructVisionConfig with default values.")
        
        self.text_config = PTPTextConfig(**text_config)
        self.vision_config = PTPVisionConfig(**vision_config)

        self.decoder_start_token_id = self.text_config.decoder_start_token_id
        self.pad_token_id = self.text_config.pad_token_id
        self.eos_token_id = self.text_config.eos_token_id

        self.add_mae_decoder = add_mae_decoder
        self.add_text_decoder = add_text_decoder

        self.initializer_factor = initializer_factor
        self.initializer_range = initializer_range


    @classmethod
    def from_text_vision_configs(
        cls, text_config: PTPTextConfig, vision_config: PTPVisionConfig, **kwargs
    ):
        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output["text_config"] = self.text_config.to_dict()
        output["vision_config"] = self.vision_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
