import math

import torch

import comfy.conds
import comfy.latent_formats
import comfy.model_base
import comfy.supported_models_base

from .model import SenseNovaU15
from .conditioning import block_causal_mask, condition_input_ids, conditioned_input_length, preprocess_reference, smart_resize, thw_indexes
from .sampling import SenseNovaModelSampling, time_snr_shift
from .text_encoder import SenseNovaTextEncoder, SenseNovaTokenizer


class SenseNovaBaseModel(comfy.model_base.BaseModel):
    def __init__(self, model_config, device=None):
        super().__init__(
            model_config,
            comfy.model_base.ModelType.FLOW,
            device=device,
            unet_model=SenseNovaU15,
        )
        self.model_sampling = SenseNovaModelSampling(model_config)
        self.memory_usage_factor_conds = ("reference_image",)

    def process_timestep(self, timestep, **kwargs):
        base_timestep = timestep / self.model_sampling.multiplier
        return 1.0 - time_snr_shift(self.model_sampling.shift, 1.0 - base_timestep)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        text_input_ids = kwargs.get("text_input_ids")
        if text_input_ids is not None:
            reference_image = kwargs.get("sensenova_reference_image")
            if reference_image is not None:
                reference_image = preprocess_reference(reference_image)
                token_height = reference_image.shape[-2] // 32
                token_width = reference_image.shape[-1] // 32
                text_input_ids = condition_input_ids(
                    text_input_ids,
                    token_height,
                    token_width,
                    image_only=kwargs.get("sensenova_reference_mode") == "image_only",
                )
                indexes = thw_indexes(text_input_ids, token_height, token_width)
                out["prefix_indexes"] = comfy.conds.CONDRegular(indexes)
                out["prefix_mask"] = comfy.conds.CONDRegular(block_causal_mask(indexes))
                out["reference_image"] = comfy.conds.CONDRegular(reference_image)
            out["text_input_ids"] = comfy.conds.CONDRegular(text_input_ids)
        return out

    def extra_conds_shapes(self, **kwargs):
        image = kwargs.get("sensenova_reference_image")
        if image is None:
            return {}
        height, width = smart_resize(*image.shape[1:3])
        out = {"reference_image": [image.shape[0], 3, height, width]}
        text_input_ids = kwargs.get("text_input_ids")
        if text_input_ids is not None:
            length = conditioned_input_length(
                text_input_ids.shape[1],
                height // 32,
                width // 32,
                image_only=kwargs.get("sensenova_reference_mode") == "image_only",
            )
            out["prefix_mask"] = [1, 1, length, length]
        return out

    def memory_required(self, input_shape, cond_shapes={}):
        memory = super().memory_required(input_shape, cond_shapes)
        mask_shapes = cond_shapes.get("prefix_mask", ())
        return memory + sum(math.prod(shape) * 4 for shape in mask_shapes)


class SenseNovaModelConfig(comfy.supported_models_base.BASE):
    unet_config = {"image_model": "sensenova_u15"}
    sampling_settings = {"shift": 3.0, "noise_scale": 1.0}
    latent_format = comfy.latent_formats.HiDreamO1Pixel
    memory_usage_factor = 0.033
    supported_inference_dtypes = [torch.bfloat16, torch.float32]
    optimizations = {"fp8": False}

    def get_model(self, state_dict, prefix="", device=None):
        return SenseNovaBaseModel(self, device=device)

    def process_unet_state_dict(self, state_dict):
        state_dict.pop("language_model.lm_head.weight", None)
        return state_dict

    def process_vae_state_dict(self, state_dict):
        return {"pixel_space_vae": torch.tensor(1.0)}

    def clip_target(self, state_dict={}):
        return comfy.supported_models_base.ClipTarget(SenseNovaTokenizer, SenseNovaTextEncoder)
