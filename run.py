#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
import torch.distributed as dist
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets
import torch
from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from data import NumpyDataset, RenderTextCollator
from streaming_data import get_multiple_domain_dataset, MDSDataset
from modeling.modeling_ptp import inject_flash_attention_ptp
from modeling.modeling_pixel import inject_flash_attention_pixel
from modeling.modeling_screenshot_llama import inject_flash_attention_screenshotllama
from modeling.configuration_ptp import PTPConfig
from modeling.configuration_pixel import PIXELConfig
from modeling.configuration_screenshot_llama import LlamaScreenshotConfig
from modeling.modeling_screenshot_llama import LlamaForScreenshot

from trainer import trainer_addon


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    load_bf16: bool = field(
        default=False,
        metadata={
            "help": (
                "Load the model in bf16 (for llama-based models)"
            )
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    flash_attn: bool = field(
        default=False,
        metadata={
            "help": "Add flash attention (1.0)"
        }
    )
    hf_flash_attn2: bool = field(
        default=False,
        metadata={
            "help": "Add HF's flash attention 2.0 (only support BF16)"
        }
    )
    mask_ratio: float = field(
        default=0,
        metadata={
            "help": "Mask ratio for patch masking"
        }
    )
    span_masking: bool = field(
        default=False,
        metadata={
            "help": "Apply PIXEL style span masking"
        }
    )
    max_span_length: int = field(
        default=6,
        metadata={"help": "For span masking: max span length"}
    )
    mae_weight: float = field(
        default=1.0,
        metadata={
            "help": "The weight for the patch prediction (MAE) loss"
        }
    )
    text_weight: float = field(
        default=1.0,
        metadata={
            "help": "The weight for text prediction loss"
        }
    )
    add_mae_decoder: bool = field(
        default=False,
        metadata={
            "help": "Add MAE decoder"
        }
    )
    add_text_decoder: bool = field(
        default=True,
        metadata={
            "help": "Add text decoder"
        }
    )
    tie_word_embeddings: bool = field(
        default=True,
        metadata={
            "help": "Tie word embeddings"
        }
    )
    pixel: bool = field(
        default=False,
        metadata={"help": "The model is PIXEL"}
    )
    screenshot_llama: bool = field(
        default=False,
        metadata={"help": "The model is screenshot-llama"}
    )
    llama: bool = field(
        default=False,
        metadata={"help": "The model is text-only llama"}
    )
    norm_pix_loss: bool = field(
        default=True,
        metadata={"help": "Norm pix loss (standardize the target pixels before calculating the loss)"}
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Ignore mismatched sizes"}
    )
    ar_text_weight: float = field(
        default=1.0,
        metadata={
            "help": "Weight for the text loss in autoregressive models"
        }
    )
    ar_pixel_weight: float = field(
        default=1.0,
        metadata={
            "help": "Wight for the patch prediction loss in autoregressive models"
        }
    )
    embedding_layernorm: bool = field(
        default=False,
        metadata={"help": "Add a layernorm layer after the embedding"}
    )


    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    font_size: int = field(
        default=10, metadata={"help": "Font size for online rendering"}
    )
    line_space: int = field(
        default=6, metadata={"help": "(Extra) line space for online rendering. The final height of one line is font_size + line_space"}
    )
    replace_new_line: bool = field(
        default=True, metadata={"help": "Replace new line with a special token (to save rendering space)"}
    )
    new_line_token: str = field(
        default="//", 
    )
    rendered_as_target: bool = field(
        default=True, metadata={"help": "Only use the rendered text as the text target and pad the rest"}
    )
    text_mask_rate: float = field(
        default=0,
        metadata={
            "help": "Text mask rate"
        }
    )
    merge_text_masks: bool = field(
        default=True,
        metadata={
            "help": "Merge consecutive text masks"
        }
    )
    ignore_white_patches: bool = field(
        default=True,
        metadata={
            "help": "Ignore white patches: exclude them in attn masks; do not mask them"
        }
    )
    remove_unicode: bool = field(
        default=False,
        metadata={
            "help": "Remove all unicode characters"
        }
    )
    add_black_patch: bool = field(
        default=False,
        metadata={
            "help": "Add black patch to the image after the text ends"
        }
    )
    add_prefix: bool = field(
        default=False,
        metadata={
            "help": "Add a text prefix at the beginning of the image"
        }
    )
    autoregressive: bool = field(
        default=False,
        metadata={
            "help": "Autoregressive style training for screenshot-llama"
        }
    )
    ar_image_block_size: int = field(
        default=256,
        metadata={
            "help": "(For autoregressive) number of tokens rendered in the screenshot"
        }
    )
    total_block_size: int = field(
        default=None,
        metadata={
            "help": "(For autoregressive) total number of tokens"
        }
    )
    context_mask: int = field(
        default=None,
        metadata={
            "help": "(For autoregressive) do not calculate loss on the first x tokens"
        }
    )
    image_mode: str = field(
        default="RGB",
        metadata={
            "help": "Image mode"
        }
    )
    streaming_dataset: bool = field(
        default=False, metadata={"help": "Whether to use streaming dataset (mosiac) or not."}
    )
    streaming_train_root: str = field(
        default=None, metadata={"help": "The root directory of the streaming training dataset."}
    )
    streaming_val_root: str = field(
        default=None, metadata={"help": "The root directory of the streaming validation dataset."}
    )
    streaming_domains: str = field(
        default=None, metadata={"help": "The domains/proportions of the streaming dataset. Should be a JSON string."}
    )
    streaming_remote: bool = field(
        default=False, metadata={"help": "Whether to use remote streaming dataset or not."}
    )
    sample_mask_at_collator: bool = field(
        default=False, metadata={"help": "Sample masks in the data loading part instead of the model part."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None and not self.streaming_dataset:
            raise ValueError("Need either a dataset name or a training/validation file.")

@dataclass
class OurTrainingArguments(TrainingArguments):

    log_eval_image_pred: bool = field(
        default=False,
        metadata={"help": "Log eval image prediction to wandb"}
    )
    width: int = field(
        default=512, metadata={"help": "Width of the input image"}
    )
    height: int = field(
        default=256, metadata={"help": "Height of the input image"}
    )
    patch_width: int = field(
        default=16, metadata={"help": "Width of the patch"}
    )
    patch_height: int = field(
        default=16, metadata={"help": "Height of the patch"}
    )
    cosine_w_min: bool = field(
        default=False, metadata={"help": "Cosine scheduler with min lr (only activated when using cosine)"}
    )
    min_learning_rate: float = field(
        default=0, metadata={"help": "Minimum learning rate"}
    )
    log_grad_norm: bool = field(
        default=False, metadata={"help": "Log grad norm"}
    )
    log_train_input: bool = field(
        default=False, metadata={"help": "Log train input"}
    )


def get_model_and_processor(model_args, training_args, config):
    model_kwargs = {}
    if model_args.hf_flash_attn2:
        assert not model_args.flash_attn # Cannot use this with flash attention 1 together
        assert model_args.screenshot_llama or model_args.llama # Only support llama-based models
        model_kwargs["use_flash_attention_2"] = True
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs["torch_dtype"] = torch_dtype

    if model_args.screenshot_llama:
        config.model_type = "screenshot-llama"
        from modeling.processing_screenshot_llama import ScreenshotLlamaProcessor
        
        processor_kwargs = {}
        if not (training_args.patch_height == 16 and training_args.patch_width == 16):
            processor_kwargs = {"patch_size": {"height": training_args.patch_height, "width": training_args.patch_width}}
            config.patch_embed_size = training_args.patch_height * training_args.patch_width * 3

        processor = ScreenshotLlamaProcessor(model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path, **processor_kwargs)
        config.img_begin_token_id = processor.img_begin_token_id
        config.img_end_token_id = processor.img_end_token_id
        config.patch_token_id = processor.patch_token_id
        config.newline_token_id = processor.newline_token_id
        config.norm_pix_loss = model_args.norm_pix_loss
        
        if model_args.model_name_or_path:
            model = LlamaForScreenshot.from_pretrained(model_args.model_name_or_path, config=config, **model_kwargs)
            if model_args.load_bf16:
                model = model.to(dtype=torch.bfloat16)
        else:
            if model_args.hf_flash_attn2:
                config._flash_attn_2_enabled = True
            model = LlamaForScreenshot._from_config(config, torch_dtype=torch_dtype)
            if model_args.load_bf16:
                model = model.to(dtype=torch.bfloat16)
            n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
            logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

        # Add the special tokens for image boundary        
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(processor.tokenizer) > embedding_size:
            model.resize_token_embeddings(len(processor.tokenizer))

        model.ar_text_weight = model_args.ar_text_weight
        model.ar_pixel_weight = model_args.ar_pixel_weight

        return processor, model
    elif model_args.llama:
        from modeling.processing_screenshot_llama import ScreenshotLlamaProcessor
        from transformers import LlamaForCausalLM
        
        processor_kwargs = {}
        processor = ScreenshotLlamaProcessor(model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path, **processor_kwargs)

        if model_args.model_name_or_path:
            model = LlamaForCausalLM.from_pretrained(model_args.model_name_or_path, config=config, **model_kwargs)
            if model_args.load_bf16:
                model = model.to(dtype=torch.bfloat16)
        else:
            if model_args.hf_flash_attn2:
                config._flash_attn_2_enabled = True
            model = LlamaForCausalLM._from_config(config, torch_dtype=torch_dtype)
            if model_args.load_bf16:
                model = model.to(dtype=torch.bfloat16)
            n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
            logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
        
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(processor.tokenizer) > embedding_size:
            model.resize_token_embeddings(len(processor.tokenizer))
        
        return processor, model
    elif config.model_type in ['ptp']:
        # Our main model
        from modeling.processing_ptp import PTPProcessor
        from modeling.modeling_ptp import PTPForConditionalGeneration

        config.add_mae_decoder = model_args.add_mae_decoder
        if model_args.add_mae_decoder:
            config.mae_weight = model_args.mae_weight
        config.add_text_decoder = model_args.add_text_decoder
        if model_args.add_text_decoder:
            config.text_weight = model_args.text_weight
        config.tie_word_embeddings = model_args.tie_word_embeddings
        config.text_config.tie_word_embeddings = model_args.tie_word_embeddings

        config.vision_config.embedding_layernorm = model_args.embedding_layernorm
        config.vision_config.norm_pix_loss = model_args.norm_pix_loss
        config.vision_config.image_size = [
            training_args.height,
            training_args.width, 
        ]
        if not (training_args.patch_height == 16 and training_args.patch_width == 16):
            config.vision_config.patch_size = (training_args.patch_height, training_args.patch_width)
        config.vision_config.num_channels = 3 if config.image_mode == "RGB" else 1

        processor = PTPProcessor.from_pretrained(model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path)

        if not (training_args.patch_height == 16 and training_args.patch_width == 16):
            processor.image_processor.patch_size = {"height": training_args.patch_height, "width": training_args.patch_width}
        
        if model_args.model_name_or_path:
            model = PTPForConditionalGeneration.from_pretrained(model_args.model_name_or_path, config=config)
            if model_args.load_bf16:
                model = model.to(dtype=torch.bfloat16)
        else:
            model = PTPForConditionalGeneration(config)
            if model_args.load_bf16:
                model = model.to(dtype=torch.bfloat16)
            n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
            logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
        
        model.mask_ratio = model_args.mask_ratio
        model.mae_weight = model_args.mae_weight
        model.text_weight = model_args.text_weight

        model.encoder.embeddings.mask_ratio = model_args.mask_ratio
        model.encoder.embeddings.span_masking = model_args.span_masking

        return processor, model
    elif config.model_type in ['pixel']:
        from modeling.processing_ptp import PTPProcessor
        from modeling.modeling_pixel import PIXELForPreTraining

        config.embedding_layernorm = model_args.embedding_layernorm
        config.norm_pix_loss = model_args.norm_pix_loss
        config.image_size = [
            training_args.height,
            training_args.width, 
        ]
        if not (training_args.patch_height == 16 and training_args.patch_width == 16):
            config.patch_size = (training_args.patch_height, training_args.patch_width)
        config.num_channels = 3 if config.image_mode == "RGB" else 1

        processor = PTPProcessor.from_pretrained(model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path)

        if not (training_args.patch_height == 16 and training_args.patch_width == 16):
            processor.image_processor.patch_size = {"height": training_args.patch_height, "width": training_args.patch_width}
        
        if model_args.model_name_or_path:
            model = PIXELForPreTraining.from_pretrained(model_args.model_name_or_path, config=config, ignore_mismatched_sizes=model_args.ignore_mismatched_sizes)
            if model_args.load_bf16:
                model = model.to(dtype=torch.bfloat16)
        else:
            model = PIXELForPreTraining(config)
            if model_args.load_bf16:
                model = model.to(dtype=torch.bfloat16)
            n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
            logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

        
        model.vit.embeddings.mask_ratio = model_args.mask_ratio
        model.vit.embeddings.span_masking = model_args.span_masking 
        
        return processor, model 
    else: 
        raise NotImplementedError

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        model_args, data_args, training_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and (training_args.do_train or training_args.check_dataset) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    # Set seed before initializing model.
    set_seed(training_args.seed)

    if data_args.streaming_dataset:
        train_dataset = get_multiple_domain_dataset(root_dir=data_args.streaming_train_root, shuffle=True, remote=data_args.streaming_remote, block_size=data_args.block_size)
        eval_dataset = get_multiple_domain_dataset(root_dir=data_args.streaming_val_root, shuffle=False, remote=data_args.streaming_remote, block_size=data_args.block_size)
    else:
        dataset_class = NumpyDataset

        logger.info("Loading train dataset (numpy)...")
        train_dataset = dataset_class(data_args.train_file, block_size=data_args.block_size) if data_args.train_file is not None else None
        logger.info("Done")

        logger.info("Loading validation dataset (numpy)...")
        eval_dataset = dataset_class(data_args.validation_file, block_size=data_args.block_size) if data_args.validation_file is not None else None
        logger.info("Done")

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.screenshot_llama:
        config_cls = LlamaScreenshotConfig
    elif model_args.llama:
        config_cls = AutoConfig
    elif model_args.pixel:
        config_cls = PIXELConfig
    else:
        config_cls = PTPConfig
    if model_args.config_name:
        config = config_cls.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = config_cls.from_pretrained(model_args.model_name_or_path, **config_kwargs)

    config.image_mode = data_args.image_mode
    training_args.image_mode = data_args.image_mode
    processor, model = get_model_and_processor(model_args, training_args, config)

    if model_args.flash_attn:
        if model_args.screenshot_llama or model_args.llama:
            inject_flash_attention_screenshotllama(model)
        elif model_args.pixel:
            inject_flash_attention_pixel(model)
        else:
            inject_flash_attention_ptp(model)
            inject_flash_attention_pixel(model.encoder)
            if hasattr(model, "mae_decoder"):
                inject_flash_attention_pixel(model.mae_decoder)

    if training_args.do_train:
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

    collator = RenderTextCollator(
        processor=processor, 
        font_size=data_args.font_size, 
        line_space=data_args.line_space, 
        replace_new_line=data_args.replace_new_line, 
        new_line_token=data_args.new_line_token, 
        width=training_args.width, height=training_args.height, 
        block_size=data_args.block_size, 
        rendered_as_target=data_args.rendered_as_target, 
        patch_width=training_args.patch_width, 
        patch_height=training_args.patch_height, 
        text_mask_rate=data_args.text_mask_rate, 
        merge_text_masks=data_args.merge_text_masks, 
        ignore_white_patches=data_args.ignore_white_patches, 
        add_black_patch=data_args.add_black_patch, 
        add_prefix=data_args.add_prefix, 
        autoregressive=data_args.autoregressive, 
        ar_image_block_size=data_args.ar_image_block_size, 
        total_block_size=data_args.total_block_size, 
        context_mask=data_args.context_mask, 
        image_mode=data_args.image_mode, 
        sample_mask_at_collator=data_args.sample_mask_at_collator, 
        mask_ratio=model_args.mask_ratio, 
        span_masking=model_args.span_masking, 
        max_span_length=model_args.max_span_length, 
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=processor,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=collator,
        compute_metrics=None, 
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
    )

    trainer = trainer_addon(trainer, streaming_dataset=data_args.streaming_dataset)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
