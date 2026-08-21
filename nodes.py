import math

from typing_extensions import override

import torch

import comfy.model_management
import comfy.model_patcher
import comfy.patcher_extension
import comfy.samplers
import folder_paths
import node_helpers
from comfy_api.latest import ComfyExtension, io

from .sensenova_u15.loader import load_pixel_vae, load_sensenova_clip, load_sensenova_model
from .sensenova_u15.guidance import edit_guidance
from .sensenova_u15.sampling import SenseNovaModelSampling


class SenseNovaU15Loader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SenseNovaU15Loader",
            display_name="SenseNova U1.5 Loader",
            category="loaders/SenseNova",
            description="Load the verified single-file SenseNova-U1.5 checkpoint. No network requests are made.",
            inputs=[
                io.Combo.Input(
                    id="model_name",
                    options=folder_paths.get_filename_list("diffusion_models"),
                ),
            ],
            outputs=[io.Model.Output(), io.Clip.Output(), io.Vae.Output()],
        )

    @classmethod
    def execute(cls, *, model_name):
        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        clip = load_sensenova_clip()
        model = load_sensenova_model(model_path, torch.bfloat16)
        return io.NodeOutput(model, clip, load_pixel_vae())


class EmptySenseNovaLatentImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EmptySenseNovaLatentImage",
            display_name="Empty SenseNova Pixel Latent",
            category="latent/SenseNova",
            inputs=[
                io.Int.Input(id="width", default=2048, min=64, max=4096, step=32),
                io.Int.Input(id="height", default=2048, min=64, max=4096, step=32),
                io.Int.Input(id="batch_size", default=1, min=1, max=1),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, *, width, height, batch_size=1):
        if batch_size != 1:
            raise ValueError("SenseNova-U1.5 currently supports batch_size=1 only")
        samples = torch.zeros(
            (batch_size, 3, height, width),
            device=comfy.model_management.intermediate_device(),
        )
        return io.NodeOutput({"samples": samples})


class SenseNovaSamplingOptions(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SenseNovaSamplingOptions",
            display_name="SenseNova Sampling Options",
            category="model/patch/SenseNova",
            description="Set the official flow timestep shift while preserving the upstream sigma trajectory.",
            inputs=[
                io.Model.Input(id="model"),
                io.Float.Input(id="shift", default=3.0, min=0.01, max=100.0, step=0.01),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, *, model, shift):
        patched = model.clone()
        model_sampling = SenseNovaModelSampling(patched.model.model_config)
        model_sampling.set_parameters(shift=shift)
        patched.add_object_patch("model_sampling", model_sampling)
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            "sensenova_prefix_cache",
            _prefix_cache_sample_wrapper,
        )
        return io.NodeOutput(patched)


def _prefix_cache_sample_wrapper(executor, *args, **kwargs):
    guider = executor.class_obj
    original_model_options = guider.model_options
    guider.model_options = comfy.model_patcher.create_model_options_clone(original_model_options)
    cache = {}
    guider.model_options.setdefault("transformer_options", {})["sensenova_prefix_cache"] = cache
    try:
        return executor(*args, **kwargs)
    finally:
        cache.clear()
        guider.model_options = original_model_options


class SenseNovaReferenceImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SenseNovaReferenceImage",
            display_name="SenseNova Reference Image",
            category="conditioning/SenseNova",
            description="Attach 1-10 source images for instruction editing. The negative input should encode an empty prompt.",
            inputs=[
                io.Conditioning.Input(id="positive"),
                io.Conditioning.Input(id="negative"),
                io.Autogrow.Input(
                    "images",
                    display_name="reference images",
                    template=io.Autogrow.TemplateNames(
                        io.Image.Input("image", display_name="reference image"),
                        names=["image"] + [f"image_{index}" for index in range(2, 11)],
                        min=1,
                    ),
                    tooltip="Reference images. Use one image for normal editing or up to ten images for multi-reference editing.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="image_condition"),
            ],
        )

    @classmethod
    def execute(cls, *, positive, negative, images):
        references = [images[name] for name in ["image"] + [f"image_{index}" for index in range(2, 11)] if name in images]
        for image in references:
            if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] < 3:
                raise ValueError("Each SenseNova reference input requires one IMAGE with at least three channels")
        positive = node_helpers.conditioning_set_values(
            positive,
            {
                "sensenova_reference_images": references,
                "sensenova_reference_mode": "condition",
            },
            append=True,
        )
        negative = node_helpers.conditioning_set_values(
            negative,
            {
                "sensenova_reference_images": references,
                "sensenova_reference_mode": "image_only",
            },
            append=True,
        )
        return io.NodeOutput(positive, negative)


class SenseNovaEditGuiderImpl(comfy.samplers.CFGGuider):
    def set_conds(self, positive, image_condition, negative):
        image_condition = node_helpers.conditioning_set_values(image_condition, {"prompt_type": "negative"})
        negative = node_helpers.conditioning_set_values(negative, {"prompt_type": "negative"})
        self.inner_set_conds(
            {
                "positive": positive,
                "image_condition": image_condition,
                "negative": negative,
            }
        )

    def set_cfg(self, cfg, img_cfg):
        self.cfg = cfg
        self.img_cfg = img_cfg

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        positive = self.conds.get("positive")
        image_condition = self.conds.get("image_condition")
        negative = self.conds.get("negative")

        if math.isclose(self.cfg, 1.0) and math.isclose(self.img_cfg, 1.0):
            return comfy.samplers.calc_cond_batch(self.inner_model, [positive], x, timestep, model_options)[0]
        if math.isclose(self.img_cfg, 1.0):
            image_out, positive_out = comfy.samplers.calc_cond_batch(
                self.inner_model, [image_condition, positive], x, timestep, model_options
            )
            return image_out + self.cfg * (positive_out - image_out)
        if math.isclose(self.cfg, self.img_cfg):
            negative_out, positive_out = comfy.samplers.calc_cond_batch(
                self.inner_model, [negative, positive], x, timestep, model_options
            )
            return negative_out + self.cfg * (positive_out - negative_out)

        negative_out, image_out, positive_out = comfy.samplers.calc_cond_batch(
            self.inner_model, [negative, image_condition, positive], x, timestep, model_options
        )
        return edit_guidance(positive_out, image_out, negative_out, self.cfg, self.img_cfg)


class SenseNovaEditGuider(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SenseNovaEditGuider",
            display_name="SenseNova Edit Guider",
            category="sampling/custom_sampling/guiders/SenseNova",
            description="Three-branch SenseNova editing guidance for SamplerCustomAdvanced.",
            inputs=[
                io.Model.Input(id="model"),
                io.Conditioning.Input(id="positive"),
                io.Conditioning.Input(id="image_condition"),
                io.Conditioning.Input(id="negative"),
                io.Float.Input(id="cfg", default=4.0, min=0.0, max=100.0, step=0.1),
                io.Float.Input(id="img_cfg", default=1.0, min=0.0, max=100.0, step=0.1),
            ],
            outputs=[io.Guider.Output()],
        )

    @classmethod
    def execute(cls, *, model, positive, image_condition, negative, cfg, img_cfg):
        guider = SenseNovaEditGuiderImpl(model)
        guider.set_conds(positive, image_condition, negative)
        guider.set_cfg(cfg, img_cfg)
        return io.NodeOutput(guider)


class SenseNovaExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [
            SenseNovaU15Loader,
            EmptySenseNovaLatentImage,
            SenseNovaSamplingOptions,
            SenseNovaReferenceImage,
            SenseNovaEditGuider,
        ]


async def comfy_entrypoint():
    return SenseNovaExtension()
