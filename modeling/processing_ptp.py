# Modified based on 
# Pix2Struct image processor: https://github.com/huggingface/transformers/blob/main/src/transformers/models/pix2struct/image_processing_pix2struct.py
# Pix2Struct processor: https://github.com/huggingface/transformers/blob/main/src/transformers/models/pix2struct/processing_pix2struct.py

from typing import List, Optional, Union, Dict

from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from transformers.utils import TensorType
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
import torch
import numpy as np
from transformers.image_utils import (
    ChannelDimension,
    ImageInput,
    get_image_size,
    infer_channel_dimension_format,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
from transformers.image_transforms import convert_to_rgb, normalize, to_channel_dimension_format, to_pil_image
from transformers.utils import TensorType, is_torch_available, is_vision_available, logging
from transformers.utils.import_utils import requires_backends
from transformers import AutoTokenizer
import math
import json
import os

class PTPImageProcessor(BaseImageProcessor):

    model_input_names = ["flattened_patches"]

    def __init__(
        self,
        do_convert_rgb: bool = True, 
        do_normalize: bool = False, # if do_normalize, image = (image - mean) / std; otherwise image = image / 255
        patch_size: Dict[str, int] = None,
        concat_coord: bool = True, # prepend the coordinates of the patches to the patch features
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.patch_size = patch_size if patch_size is not None else {"height": 16, "width": 16}
        self.do_normalize = do_normalize 
        self.do_convert_rgb = do_convert_rgb
        self.concat_coord = concat_coord

    def extract_flattened_patches(self, image: np.ndarray, patch_size: dict, concat_coord = True, **kwargs) -> np.ndarray:
        
        requires_backends(self.extract_flattened_patches, "torch")

        # convert to torch
        image = to_channel_dimension_format(image, ChannelDimension.FIRST)
        image = torch.from_numpy(image)

        patch_height, patch_width = patch_size["height"], patch_size["width"]
        image_height, image_width = get_image_size(image)
        
        patches = torch_extract_patches(image, patch_height, patch_width)

        patches_shape = patches.shape
        rows = patches_shape[1]
        columns = patches_shape[2]
        depth = patches_shape[3]

        # [rows * columns, patch_height * patch_width * image_channels]
        patches = patches.reshape([rows * columns, depth])

        # [rows * columns, 1]
        if concat_coord:
            row_ids = torch.arange(rows).reshape([rows, 1]).repeat(1, columns).reshape([rows * columns, 1])
            col_ids = torch.arange(columns).reshape([1, columns]).repeat(rows, 1).reshape([rows * columns, 1])

            # Offset by 1 so the ids do not contain zeros, which represent padding.
            row_ids += 1
            col_ids += 1

            # Prepare additional patch features.
            # [rows * columns, 1]
            row_ids = row_ids.to(torch.float32)
            col_ids = col_ids.to(torch.float32)

            # [rows * columns, 2 + patch_height * patch_width * image_channels]
            result = torch.cat([row_ids, col_ids, patches], -1)

            # [max_patches, 2 + patch_height * patch_width * image_channels]
            # result = torch.nn.functional.pad(result, [0, 0, 0, max_patches - (rows * columns)]).float()
        else:
            result = patches

        result = to_numpy_array(result)

        return result

    def normalize(
        self, image: np.ndarray, data_format: Optional[Union[str, ChannelDimension]] = None, **kwargs
    ) -> np.ndarray:
        """
        Normalize an image. image = (image - image_mean) / image_std.

        The image std is to mimic the tensorflow implementation of the `per_image_standardization`:
        https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization

        Args:
            image (`np.ndarray`):
                Image to normalize.
        """
        if image.dtype == np.uint8:
            image = image.astype(np.float32)

        # take mean across the whole `image`
        mean = np.mean(image)
        std = np.std(image)
        adjusted_stddev = max(std, 1.0 / math.sqrt(np.prod(image.shape)))

        return normalize(image, mean=mean, std=adjusted_stddev, **kwargs)

    def preprocess(
        self,
        images: ImageInput,
        do_convert_rgb: bool = None,
        do_normalize: Optional[bool] = None,
        concat_coord: bool = None,
        patch_size: Optional[Dict[str, int]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        **kwargs,
    ) -> ImageInput:
        """
        Preprocess an image or batch of images. The processor first computes the maximum possible number of
        aspect-ratio preserving patches of size `patch_size` that can be extracted from the image. It then pads the
        image with zeros to make the image respect the constraint of `max_patches`. Before extracting the patches the
        images are standardized following the tensorflow implementation of `per_image_standardization`
        (https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization).


        Args:
            images (`ImageInput`):
                Image to preprocess.
            header_text (`Union[List[str], str]`, *optional*):
                Text to render as a header. Only has an effect if `image_processor.is_vqa` is `True`.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            max_patches (`int`, *optional*, defaults to `self.max_patches`):
                Maximum number of patches to extract.
            patch_size (`dict`, *optional*, defaults to `self.patch_size`):
                Dictionary containing the patch height and width.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                    - Unset: Return a list of `np.ndarray`.
                    - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                    - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                    - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                    - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
        """
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        patch_size = patch_size if patch_size is not None else self.patch_size
        concat_coord = concat_coord if concat_coord is not None else self.concat_coord

        if kwargs.get("data_format", None) is not None:
            raise ValueError("data_format is not an accepted input as the outputs are ")

        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        # PIL RGBA images are converted to RGB
        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if do_normalize:
            images = [self.normalize(image=image) for image in images]
        else:
            images = [image / 255.0 for image in images]

        # convert to torch tensor and permute
        images = [
            self.extract_flattened_patches(image=image, patch_size=patch_size, concat_coord=concat_coord).astype(np.float32)
            for image in images
        ]

        # create attention mask in numpy
        attention_masks = [(image.sum(axis=-1) != 0).astype(np.float32) for image in images]

        encoded_outputs = BatchFeature(
            data={"flattened_patches": images, "attention_mask": attention_masks}, tensor_type=return_tensors
        )

        return encoded_outputs



class PTPProcessor:

    def __init__(self, config_path, **image_processor_kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(config_path)
        self.tokenizer.return_token_type_ids = False

        self.image_processor_config = json.load(open(os.path.join(config_path, "preprocessor_config.json")))
        self.image_processor_config.update(image_processor_kwargs)
        self.image_processor = PTPImageProcessor(**self.image_processor_config)

    @staticmethod
    def from_pretrained(config_path):
        return PTPProcessor(config_path)

    def save_pretrained(self, save_directory):
        self.tokenizer.save_pretrained(save_directory)
        json.dump(self.image_processor_config, open(os.path.join(save_directory, "preprocessor_config.json"), "w"), indent=4)

    def __call__(
        self,
        images=None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_token_type_ids: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchEncoding:

        if images is None and text is None:
            raise ValueError("You have to specify either images or text.")

        # Get only text
        if images is None:
            self.current_processor = self.tokenizer
            text_encoding = self.tokenizer(
                text=text,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_token_type_ids=return_token_type_ids,
                return_length=return_length,
                verbose=verbose,
                return_tensors=return_tensors,
                **kwargs,
            )
            return text_encoding

        # add pixel_values
        encoding_image_processor = self.image_processor(
            images, return_tensors=return_tensors, **kwargs
        )

        if text is not None:
            text_encoding = self.tokenizer(
                text=text,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_token_type_ids=return_token_type_ids,
                return_length=return_length,
                verbose=verbose,
                return_tensors=return_tensors,
                **kwargs,
            )

            if "attention_mask" in text_encoding:
                text_encoding["decoder_attention_mask"] = text_encoding.pop("attention_mask")
            if "input_ids" in text_encoding:
                text_encoding["decoder_input_ids"] = text_encoding.pop("input_ids")
        else:
            text_encoding = None

        if text_encoding is not None:
            encoding_image_processor.update(text_encoding)
        
        return encoding_image_processor

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Pix2StructTokenizerFast's [`~PreTrainedTokenizer.batch_decode`].
        Please refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Pix2StructTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))




# adapted from: https://discuss.pytorch.org/t/tf-image-extract-patches-in-pytorch/171409/2
def torch_extract_patches(image_tensor, patch_height, patch_width):
    """
    Utiliy function to extract patches from a given image tensor. Returns a tensor of shape (1, `patch_height`,
    `patch_width`, `num_channels`x `patch_height` x `patch_width`)

    Args:
        image_tensor (torch.Tensor):
            The image tensor to extract patches from.
        patch_height (int):
            The height of the patches to extract.
        patch_width (int):
            The width of the patches to extract.
    """
    requires_backends(torch_extract_patches, ["torch"])

    image_tensor = image_tensor.unsqueeze(0)
    patches = torch.nn.functional.unfold(image_tensor, (patch_height, patch_width), stride=(patch_height, patch_width))
    patches = patches.reshape(image_tensor.size(0), image_tensor.size(1), patch_height, patch_width, -1)
    patches = patches.permute(0, 4, 2, 3, 1).reshape(
        image_tensor.size(2) // patch_height,
        image_tensor.size(3) // patch_width,
        image_tensor.size(1) * patch_height * patch_width,
    )
    return patches.unsqueeze(0)


