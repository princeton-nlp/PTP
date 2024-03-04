# Most part of the code was adopted from https://gist.github.com/pojda/8bf989a0556845aaf4662cd34f21d269

from PIL import Image, ImageDraw, ImageFont
import PIL
import numpy as np

from io import BytesIO
import base64

import sys
sys.path.append("rendering/src")

try:
    import renderer
except ImportError as e:
    print("Fail to import simple renderer")
    print(e)


def render_text(
    text, 
    width=512, 
    height=256, 
    font_size=10, 
    line_space=6, # the final height of one line is font_size + line_space
    white_bg=True,
    no_full_rendering_warning=False, # print a message if the text is not fully rendered
):
    array = np.zeros(width*height, dtype=np.int8)
    # extra_line_space = int(font_size * line_space - font_size)
    
    rendered, rendered_text = renderer.render_unicode(
        array, text, height, width, font_size, line_space, True, True, True, True, True
    )
    if no_full_rendering_warning and len(rendered_text) != len(text):
        print("Warning: text got cut off and was not fully rendered!!")
    rendered = rendered.reshape(height, width)
    rendered = (255 - rendered) if white_bg else rendered
    return Image.fromarray(rendered, "L").convert("RGB"), rendered_text


def renormalize_pred(image_tensor):
    min = image_tensor.min(-1).values.min(-1).values
    max = image_tensor.max(-1).values.max(-1).values
    std = 255 / (max - min)
    mean = -min * std
    image_tensor = image_tensor * std.unsqueeze(-1).unsqueeze(-1) + mean.unsqueeze(-1).unsqueeze(-1)
    return image_tensor


def renormalize(image_tensor):
    return image_tensor * 255


def flattened_patches_to_image(flattened_patches, height=256, width=512, patch_height=16, patch_width=16, mask=None, original_patches=None, image_mode="RGB"):
    # Convert flattened_patches back to PIL image
    # flattend_patches: (num_patches, 768)
    h = height // patch_height
    w = width // patch_width
    c = 3 if image_mode == "RGB" else 1
    if image_mode == "1":
        flattened_patches = flattened_patches * 255
        image_mode = "L" # convert to grayscale for further processing
    if original_patches is not None and mask is not None:
        original_patches = renormalize(original_patches)
        flattened_patches = renormalize_pred(flattened_patches)
        flattened_patches = flattened_patches * mask.unsqueeze(-1) + original_patches * (1 - mask.unsqueeze(-1))
    else: 
        flattened_patches = renormalize(flattened_patches)

    flattened_patches = flattened_patches.reshape(h * w, patch_height, patch_width, c) # (h * w, ph, pw, 3)
    if mask is not None:
        flattened_patches[mask.bool(), :, :, 0] = flattened_patches[mask.bool(), :, :, 0] * 0.7 + 255 * 0.3
    flattened_patches = flattened_patches.reshape(h, w, patch_height, patch_width, c) # (h, w, ph, pw, 3)
    flattened_patches = flattened_patches.permute(0, 2, 1, 3, 4) # (h, ph, w, pw, 3)
    flattened_patches = flattened_patches.reshape(h * patch_height, w * patch_width, c) # (h * ph, w * pw, 3)
    if c == 1:
        flattened_patches = flattened_patches.squeeze(-1)
    image = flattened_patches.numpy()

    image = Image.fromarray(image.astype(np.uint8), mode=image_mode)

    return image


def flattened_patches_to_vit_pixel_values(flattened_patches, height=256, width=512, patch_height=16, patch_width=16, image_mode="RGB"):
    # Convert flattened_patches to pixel values
    # pixel_values: batch_size, num_channels, height, width 
    # flattend_patches: (num_patches, 768)
    h = height // patch_height
    w = width // patch_width
    num_channels = 3 if image_mode == "RGB" else 1
    bsz, num_patch, patch_emb = flattened_patches.shape

    flattened_patches = flattened_patches.reshape(bsz, h * w, patch_height, patch_width, num_channels) # (bsz, h * w, ph, pw, c)
    flattened_patches = flattened_patches.reshape(bsz, h, w, patch_height, patch_width, num_channels) # (bsz, h, w, ph, pw, c)
    flattened_patches = flattened_patches.permute(0, 5, 1, 3, 2, 4) # (bsz, c, h, ph, w, pw)
    flattened_patches = flattened_patches.reshape(bsz, num_channels, h * patch_height, w * patch_width) # (bsz, c, h * ph, w * pw)

    return flattened_patches

