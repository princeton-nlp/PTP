from transformers import LlamaConfig
import copy

class LlamaScreenshotConfig(LlamaConfig):

    model_type = "screenshot-llama"

    def __init__(
        self,
        patch_embed_size=768,
        img_begin_token_id=None,
        img_end_token_id=None,
        patch_token_id=None,
        newline_token_id=None,
        norm_pix_loss=False,
        pixel_decoder_config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # This is the size of the patch embedding (ph x pw x 3)
        self.patch_embed_size = patch_embed_size

        # Newly added special token
        self.img_begin_token_id = img_begin_token_id
        self.img_end_token_id = img_end_token_id
        self.patch_token_id = patch_token_id
        self.newline_token_id = newline_token_id

        # Loss for pixel-level supervision
        self.norm_pix_loss = norm_pix_loss

        if isinstance(pixel_decoder_config, dict):
            self.pixel_decoder_config = LlamaScreenshotConfig(pixel_decoder_config)
        else:
            self.pixel_decoder_config = pixel_decoder_config
    
    def to_dict(self):

        output = copy.deepcopy(self.__dict__)
        if self.pixel_decoder_config is not None:
            output['pixel_decoder_config'] = self.pixel_decoder_config.to_dict()
        output["model_type"] = self.__class__.model_type

        return output
