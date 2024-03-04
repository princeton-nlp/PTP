from torch.utils.data import Dataset
import numpy as np
# import gi
# gi.require_version('Pango', '1.0')
# gi.require_version('PangoCairo', '1.0')
# from gi.repository import Pango, PangoCairo
# import cairo
from PIL import Image
from dataclasses import dataclass, field
import torch
from streaming import LocalDataset
from image_utils import *
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.image_utils import to_numpy_array
from modeling.span_masking import SpanMaskingGenerator
from random import sample
from image_utils import render_text

class NumpyDataset(Dataset):

    def __init__(self, path, block_size=None):
        self.tokens = np.load(path)
        self.block_size = self.tokens.shape[1] if block_size is None else block_size
        self.font_size = None

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return {"tokens": self.tokens[idx][:self.block_size], "font_size": self.font_size}


class RenderTextCollator:
    def __init__(self,
        processor: object,
        font_size: int,
        line_space: int,
        replace_new_line: bool,
        new_line_token: str,
        width: int,
        height: int,
        block_size: int = 1024,
        rendered_as_target: bool = False,
        patch_width: int = 16,
        patch_height: int = 16,
        text_mask_rate: float = 0,
        merge_text_masks: bool = False,
        ignore_white_patches: bool = False,
        add_black_patch: bool = False,
        add_prefix: bool = False,
        autoregressive: bool = False,
        ar_image_block_size: int = None,
        total_block_size: int = None,
        context_mask: int = None,
        image_mode: str = "RGB",
        sample_mask_at_collator: bool = False,
        mask_ratio: float = 0,
        span_masking: bool = False,
        max_span_length: int = 6,
    ):
        self.processor = processor
        self.font_size = font_size
        self.line_space = line_space
        self.replace_new_line = replace_new_line
        self.new_line_token = new_line_token
        self.width = width
        self.height = height
        self.block_size = block_size
        self.rendered_as_target = rendered_as_target
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.text_mask_rate = text_mask_rate
        self.merge_text_masks = merge_text_masks
        self.ignore_white_patches = ignore_white_patches
        self.add_black_patch = add_black_patch
        self.add_prefix = add_prefix
        self.autoregressive = autoregressive
        self.ar_image_block_size = ar_image_block_size
        self.total_block_size = total_block_size
        self.context_mask = context_mask
        self.image_mode = image_mode
        self.sample_mask_at_collator = sample_mask_at_collator
        self.mask_ratio = mask_ratio
        self.span_masking = span_masking
        self.max_span_length = max_span_length
    
    
    def mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Text masking
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.text_mask_rate)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.processor.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        inputs[masked_indices] = self.processor.tokenizer.mask_token_id

        return inputs, labels


    def __call__(self, batch):
        new_batch = {"flattened_patches": [], "attention_mask": [], "labels": []}
        if self.autoregressive:
            # Data for autoregressive mode
            new_batch["input_ids"] = []
            if self.ar_image_block_size == 0:
                # Text only
                new_batch = {"input_ids": [], "attention_mask": [], "labels": []}
        if self.sample_mask_at_collator:
            # Sample patch mask in data collator
            new_batch["patch_mask"] = []

        for item in batch:
            if self.autoregressive and self.ar_image_block_size == 0:
                # Autoregressive: text only
                text_tokens = torch.tensor(item["tokens"].astype(np.int64)).long()
                
                input_ids = torch.cat([torch.tensor([self.processor.tokenizer.bos_token_id]).long(), text_tokens], 0)
                attention_mask = torch.ones(input_ids.shape).long()
                if self.total_block_size is not None:
                    # Truncate
                    input_ids = input_ids[:self.total_block_size]
                    attention_mask = attention_mask[:self.total_block_size]
                new_batch["input_ids"].append(input_ids)
                new_batch["attention_mask"].append(attention_mask)
                labels = input_ids + 0
                if self.context_mask is not None:
                    # Only predict on the non-masked part (mostly for evaluation)
                    labels[:self.context_mask] = -100
                new_batch["labels"].append(labels)
            elif self.autoregressive:
                # Autoregressive with screenshot
                image_tokens = item["tokens"][:self.ar_image_block_size] # render these as screenshots

                text = self.processor.decode(image_tokens, skip_special_tokens=True) 
                if self.replace_new_line:
                    text = text.replace("\n", self.new_line_token)
                
                if self.add_prefix:
                    text = "Beginning of the sequence: " + text

                image, rendered_text = render_text(text=text, font_size=self.font_size, line_space=self.line_space, width=self.width, height=self.height)

                # In the case where not all text is rendered into the screenshot, we truncate the text
                if self.replace_new_line:
                    _ = rendered_text.replace(self.new_line_token, "\n").rstrip(" ")
                else:
                    _ = rendered_text.rstrip(" ")
                encoded_num_img_tokens = len(self.processor(text=_, add_special_tokens=False)['input_ids'])
                text_tokens = torch.tensor(item["tokens"][min(encoded_num_img_tokens,self.ar_image_block_size):].astype(np.int64)).long()
                encoding = self.processor(images=image, return_tensors="pt", add_special_tokens=True)

                new_batch["flattened_patches"].append(encoding["flattened_patches"][0])
                patch_attention_mask = encoding["attention_mask"][0]

                assert not self.add_black_patch # not supported (and not needed with </img>)

                # Mask out the attention to ending white patches
                if self.ignore_white_patches:
                    fpatches = new_batch["flattened_patches"][-1][:, 2:]
                    non_white_patches = ((fpatches - fpatches.mean(dim=-1, keepdim=True)) ** 2 < 1e-6).long().sum(-1) != fpatches.shape[-1]
                    reverse_non_white_patches = non_white_patches.flip(-1)
                    non_white_patches = reverse_non_white_patches.nonzero()
                    if len(non_white_patches) == 0:
                        first_white_patch = 0
                    else:
                        first_white_patch = len(reverse_non_white_patches) - non_white_patches[0][0]
                    
                    patch_attention_mask[first_white_patch:] = 0

                # BOS + image + text
                input_ids = torch.cat([torch.tensor([self.processor.tokenizer.bos_token_id]).long(), encoding["image_input_ids"][0], text_tokens], 0)
                attention_mask = torch.ones(input_ids.shape).long()
                patch_mask = input_ids == self.processor.patch_token_id
                attention_mask[patch_mask] = patch_attention_mask.long()
                if self.total_block_size is not None:
                    input_ids = input_ids[:self.total_block_size]
                    attention_mask = attention_mask[:self.total_block_size]
                new_batch["input_ids"].append(input_ids)
                new_batch["attention_mask"].append(attention_mask)
                new_batch["labels"].append(input_ids)

            else: 
                if self.text_mask_rate > 0:
                    input_ids = torch.tensor(item["tokens"].astype(np.int32)).long().unsqueeze(0)
                    input_ids, labels = self.mask_tokens(input_ids)
                    input_ids = input_ids.squeeze(0)
                    labels = labels.squeeze(0)
                    text = self.processor.decode(input_ids, skip_special_tokens=False) 
                else:
                    text = self.processor.decode(item["tokens"], skip_special_tokens=True) 

                if self.replace_new_line:
                    text = text.replace("\n", self.new_line_token)

                if self.merge_text_masks and self.text_mask_rate > 0:
                    while True:
                        if "<mask><mask>" not in text:
                            break
                        text = text.replace("<mask><mask>", "<mask>")

                if self.add_prefix:
                    text = "Beginning of the sequence: " + text

                image, rendered_text = render_text(text=text, font_size=self.font_size, line_space=self.line_space, width=self.width, height=self.height)
                image = image.convert(self.image_mode)
                image = to_numpy_array(image)
                if self.image_mode != "RGB":
                    image = np.expand_dims(image, -1) # h, w, 1
                if self.image_mode == "1":
                    image = image.astype(np.float32) # bool -> float for clf

                if self.rendered_as_target:
                    if self.text_mask_rate > 0:
                        # this is not very accurate as with the merge masks we can only estimate how much is rendered in the labels
                        valid_num_tokens = len(self.processor.tokenizer.tokenize(rendered_text))
                        # consider the merged masks
                        valid_num_tokens = int(valid_num_tokens / (len(self.processor.tokenizer.tokenize(text)) / len(labels)))
                        labels[valid_num_tokens:] = self.processor.tokenizer.pad_token_id
                    else:
                        labels = self.processor.tokenizer(rendered_text, return_tensors="pt", add_special_tokens=False, max_length=self.block_size, padding="max_length", truncation=True)["input_ids"].squeeze() 
                   
                encoding = self.processor(images=image, return_tensors="pt", add_special_tokens=True)
                new_batch["flattened_patches"].append(encoding["flattened_patches"][0])
                new_batch["attention_mask"].append(encoding["attention_mask"][0])
                new_batch["labels"].append(labels)

                if self.add_black_patch:
                    self.ignore_white_patches

                if self.ignore_white_patches:
                    fpatches = new_batch["flattened_patches"][-1][:, 2:]
                    # White patches should have all pixels = 1 (normalized)
                    non_white_patches = (fpatches > 1 - 1e-6).long().sum(-1) != fpatches.shape[-1]
                    reverse_non_white_patches = non_white_patches.flip(-1)
                    non_white_patches = reverse_non_white_patches.nonzero()
                    if len(non_white_patches) == 0:
                        first_white_patch = 0
                    else:
                        first_white_patch = len(reverse_non_white_patches) - non_white_patches[0][0]
                    
                    new_batch["attention_mask"][-1][first_white_patch:] = 0

                    if self.add_black_patch:
                        if first_white_patch == len(reverse_non_white_patches):
                            first_white_patch -= 1 # if there is no white patch, force changing the last one to black
                        
                        black = 0 
                        new_batch["flattened_patches"][-1][first_white_patch, 2:] = black
                        new_batch["attention_mask"][-1][first_white_patch] = 1

            if self.sample_mask_at_collator:
                assert self.span_masking is True # we are only doing this for span masking
                seq_length = new_batch["flattened_patches"][-1].shape[0]
                len_keep = int(seq_length * (1 - self.mask_ratio))
                span_masking_generator = SpanMaskingGenerator(
                    num_patches=seq_length,
                    num_masking_patches=seq_length-len_keep,
                    max_span_length=self.max_span_length,
                    spacing="span",
                    cumulative_span_weights=[0.2,0.4,0.6,0.8,0.9,1]
                )
                patch_mask = torch.tensor(span_masking_generator())
                new_batch["patch_mask"].append(patch_mask)

        for key in new_batch:
            new_batch[key] = torch.stack(new_batch[key])
        
        return new_batch
