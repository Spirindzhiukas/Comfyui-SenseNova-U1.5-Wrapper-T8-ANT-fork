# SenseNova U1.5 for ComfyUI — ANT fork

English | [简体中文](README_CN.md)

[![CI](https://github.com/Spirindzhiukas/Comfyui-SenseNova-U1.5-Wrapper-T8-ANT-fork/actions/workflows/ci.yml/badge.svg)](https://github.com/Spirindzhiukas/Comfyui-SenseNova-U1.5-Wrapper-T8-ANT-fork/actions/workflows/ci.yml)

[Changelog](CHANGELOG.md) · [GitHub Releases](https://github.com/Spirindzhiukas/Comfyui-SenseNova-U1.5-Wrapper-T8-ANT-fork/releases)

> **This fork is not published on the ComfyUI Registry** and is not listed in
> ComfyUI-Manager by name. Install it from Git — see
> [Installation](#installation). The Registry package `sensenova-u15-t8` belongs
> to the upstream author,
> [T8mars](https://github.com/T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8).

## About this fork

This repository is a maintenance fork of
[`T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8`](https://github.com/T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8)
with upstream features and fixes synchronized through **1.5.2**, with the ConvRot quantization stack of
[`Milor123/ComfyUI-SenseNova-U1.5-ConvRot`](https://github.com/Milor123/ComfyUI-SenseNova-U1.5-ConvRot)
ported on top of it, plus this fork's own fixes. Upstream's fixes, checkpoint
contract and Blackwell-safe RoPE are all kept.

What this fork adds or changes on top of those two codebases:

- **English UI everywhere** — node labels, slot names, structured-prompt defaults,
  the frontend extension and the shipped example workflows. The upstream Chinese
  wording is kept as comments so upstream merges stay easy to review.
- **Windows/CRLF-proof loading** — the packaged tokenizer assets are validated
  against both the raw and the LF-normalised digest, so a `core.autocrlf=true`
  clone no longer aborts with `tokenizer asset digest mismatch`.
- **ConvRot quantized checkpoints (ported from Milor123)** — INT8 (about 17.6 GB),
  ConvRot W4A4 and asymmetric W4A8 (about 13.8 GB), including per-layer mixes, as
  an *optional* path that never changes bf16 behaviour.
- **Transitional RoPE hook readiness** — the three per-axis RoPE bases can be
  overridden through `transformer_options` and are cache-safe across normal,
  thinking and interleave paths. The intended loader-agnostic RoPE Lab MODEL
  patch replacement is specified in
  [`docs/ROPE_LAB_MODELPATCH_ARCHITECTURE.md`](docs/ROPE_LAB_MODELPATCH_ARCHITECTURE.md).
- **Conversion tooling** in `tools/` and a maintenance contract in
  [`memory.md`](memory.md).

Who wrote what — the full per-file breakdown of the code taken from T8mars, the
code ported from Milor123 and the changes made here — is documented in
[Credits and provenance](#credits-and-provenance).

Native ComfyUI nodes for SenseNova U1.5. The model, sampler, scheduler, VRAM offloading, LoRA loading, and workflows all use ComfyUI's native pipeline.

Supported features:

- Text-to-image generation
- Thinking image generation, with model reasoning before diffusion sampling
- Interleaved text/image generation with generated-image feedback into later turns
- Single-image editing
- Multi-reference editing with 1 to 10 images
- Generate 1 to 16 different results from the same prompt and references
- Standard ComfyUI `KSampler`
- Official U1.5 Final and U1.5 SFT checkpoints
- Q2_K, Q3_K_M, Q5_K_M, Q6_K and Q8_0 GGUF quantizations of U1.5 Final
- Official U1.5 8-step LoRA through ComfyUI's native LoRA and `ModelPatcher` pipeline
- Three-branch guidance with a separate `img_cfg`
- CFG Norm and configurable CFG intervals
- A structured prompt helper for complex image-editing tasks
- Execution-local text and reference-image prefix KV cache

The nodes only read local model files. They never download models while ComfyUI is running.

## Installation

Install from Git. This fork is not on the ComfyUI Registry and cannot be
installed with `comfy node install` or found by name in ComfyUI-Manager:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Spirindzhiukas/Comfyui-SenseNova-U1.5-Wrapper-T8-ANT-fork.git
```

Then restart ComfyUI. To update later:

```bash
cd ComfyUI/custom_nodes/Comfyui-SenseNova-U1.5-Wrapper-T8-ANT-fork
git pull
```

In ComfyUI-Manager you can also use **Install via Git URL** with the same
address. Do not install the upstream `SenseNova U1.5 (T8)` Registry package next
to this one — two copies of the same node ids in `custom_nodes/` is the most
common cause of `checkpoint key mismatch`.

Dependencies:

- This custom node has no extra Python dependencies for the bf16 path.
- GGUF support requires `gguf >= 0.13.0`; installing this repository normally
  installs it from `pyproject.toml`.
- The optional quantized (ConvRot) path uses `comfy-kitchen` from your ComfyUI
  environment; `comfy-kitchen >= 0.2.31` is recommended because its INT8 kernel
  is the first one that actually applies the ConvRot rotation. Older releases
  still work — this pack measures the kernel at load time and falls back to its
  own ConvRot forwards.

If you would rather run the Registry-published upstream version, it is
[T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8](https://github.com/T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8);
it does not load the quantized ConvRot checkpoints.

## Download the models

### Which repository has which models

Two community repositories host the checkpoints this node pack loads, and they
hold different things. The model weights themselves were released by
OpenSenseNova; both community repos publish conversions of those official files.

| Repository | Published by | What it hosts |
|---|---|---|
| [`t8star/SenseNova-U1.5-Comfy`](https://huggingface.co/t8star/SenseNova-U1.5-Comfy/) | **T8mars** — author of the upstream ComfyUI wrapper this fork is based on | **Full-precision (bf16) checkpoints**: all-BF16 Final (~35 GB), legacy mixed-precision Final (~50 GB), SFT (~35 GB), and the ComfyUI-native 8-step LoRA (~815 MB). Each checkpoint also has a `*.manifest.json` tensor listing. Mirror: [Quark](https://pan.quark.cn/s/6b756fdae32d) |
| [`Milor123/ComfyUI-ConvRot-SenseNova-U1.5-8B-MoT-T8`](https://huggingface.co/Milor123/ComfyUI-ConvRot-SenseNova-U1.5-8B-MoT-T8) | **Milor123** — author of the ConvRot quantization fork the quant code here is ported from | **Quantized ConvRot checkpoints**: INT8-ConvRot (~17.6 GiB, recommended quant) and hybrid INT8+W4A8 `L18-41` (~13.8 GiB), plus a copy of the same 8-step LoRA under `Loras/` |
| [`realrebelai/SenseNova-U1.5-8B_GGUFs`](https://huggingface.co/realrebelai/SenseNova-U1.5-8B_GGUFs) | **realrebelai** | Verified Final GGUF files: Q2_K, Q3_K_M, Q5_K_M, Q6_K and Q8_0; loaded by this pack's dedicated GGUF loader |
| [`sensenova/SenseNova-U1.5-8B-MoT`](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT) · [`-SFT`](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT-SFT) · [`-LoRAs`](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT-LoRAs) | **OpenSenseNova** — the model authors | The **official source weights** both community repos were converted from. Use these only if you want to convert files yourself: the raw LoRA needs `tools/convert_lora_to_comfy.py` and the sharded checkpoints need `tools/merge_safetensors.py` |

Simple rule: **bf16 comes from the `t8star` repo, quantized comes from the
`Milor123` repo.**

Three details that matter:

- Milor123's quants were converted from the **legacy mixed-precision Final**
  (revision `1f6ec604`), so this pack validates them as `final_legacy`. The
  metadata injector keeps that tag automatically (`--variant auto`).
- The 8-step LoRA is the **same file** in both repos — byte-identical
  (SHA256 `dd5320f0…`) — so download it from whichever repo you are already
  using.
- No repository publishes a ConvRot **W4A4** file (~11 GB). Convert it yourself
  with `tools/convert_sensenova_int4_convrot.py`.

### bf16 and official-precision files — `t8star/SenseNova-U1.5-Comfy`

Download only the files you need:

| File | Place it in | Purpose |
|---|---|---|
| `SenseNova-U1.5-8B-MoT-BF16-T8.safetensors` | `ComfyUI/models/diffusion_models/` | Current all-BF16 U1.5 Final single file, about 35 GB; recommended |
| `SenseNova-U1.5-8B-MoT-T8.safetensors` | `ComfyUI/models/diffusion_models/` | Legacy mixed-precision U1.5 Final single file, about 50 GB; still supported |
| `SenseNova-U1.5-8B-MoT-SFT-T8.safetensors` | `ComfyUI/models/diffusion_models/` | U1.5 SFT single-file checkpoint, about 35 GB |
| `SenseNova-U1.5-8B-MoT-LoRA-8step-ComfyUI.safetensors` | `ComfyUI/models/loras/` | ComfyUI-native conversion of the official 8-step LoRA, about 815 MB |

### Quantized ConvRot files — `Milor123/ComfyUI-ConvRot-SenseNova-U1.5-8B-MoT-T8`

| File | Place it in | Purpose |
|---|---|---|
| `SenseNova-U1.5-8B-MoT-T8-int8-convrot-tagged.safetensors` | `ComfyUI/models/diffusion_models/` | INT8 + ConvRot, about 17.6 GiB; recommended quantized file |
| `SenseNova-U1.5-8B-MoT-T8-hybw4a8-L18-41.safetensors` | `ComfyUI/models/diffusion_models/` | Hybrid: INT8 for layers 0–17, asymmetric W4A8 for layers 18–41, about 13.8 GiB |
| `Loras/SenseNova-U1.5-8B-MoT-LoRA-8step-ComfyUI.safetensors` | `ComfyUI/models/loras/` | The same 8-step LoRA as in the `t8star` repo |

These files already carry the SenseNova provenance metadata this loader
requires, so they load without running the injector. See
[ConvRot quantized checkpoints](#convrot-quantized-checkpoints) for how to
confirm the quantized path is active.

### GGUF files — `realrebelai/SenseNova-U1.5-8B_GGUFs`

Use `SenseNova U1.5 GGUF Loader (Final)` for the verified Q2_K, Q3_K_M,
Q5_K_M, Q6_K and Q8_0 files. Place them in `ComfyUI/models/gguf/`. The loader
strictly checks the selected profile's size, SHA256, tensor names, shapes and
quantization types before construction. Full validation details are in
[`docs/gguf-validation.md`](docs/gguf-validation.md).

Base-model directory:

```text
ComfyUI/models/diffusion_models/
```

LoRA directory:

```text
ComfyUI/models/loras/
```

GGUF directory:

```text
ComfyUI/models/gguf/
```

A subdirectory such as `ComfyUI/models/diffusion_models/SenseNovaU1.5/` also
works — ComfyUI lists it in the loader. Nothing in this node pack downloads
model files: installing the nodes and downloading the weights are separate
steps, whichever way you install them.

Final and SFT are both SenseNova U1.5 checkpoints. The current BF16 Final is the official all-BF16 conversion and re-shard of the same Final model; this node strictly supports both the current 35 GB Final and the legacy 50 GB Final. SFT is a separate training-stage checkpoint.

The official 8-step LoRA must be used with Final. Do not apply it to SFT or Preview. The dedicated `SenseNova U1.5 8-Step LoRA` node checks the base model and gives a clear error if the combination is invalid.

| File or combination | Supported | Notes |
|---|---:|---|
| U1.5 Final BF16, 50-step generation/editing | ✅ | Current recommended checkpoint, about 35 GB |
| Legacy mixed-precision U1.5 Final, 50-step generation/editing | ✅ | Existing downloads remain supported, about 50 GB |
| U1.5 Final + `-ComfyUI` 8-step LoRA | ✅ | 8-step text-to-image only |
| U1.5 SFT, 50-step generation/editing | ✅ | Standalone checkpoint; do not add the 8-step LoRA |
| Quantized Final (INT8 ConvRot / hybrid W4A8), 50-step generation/editing | ✅ | Fork compatibility path, from the `Milor123` repo; the 8-step LoRA also works on top |
| Final GGUF, 50-step generation/editing | ✅ | Five strictly verified profiles; Q3_K_M is the suggested low-VRAM starting point |
| Final GGUF + 8-step LoRA | ✅ | Uses the native ModelPatcher LoRA path |
| U1.5 Preview | ❌ | Older preview checkpoint |
| Unconverted official raw LoRA | ❌ | Use the `-ComfyUI` file or convert it with the included tool |

## Ready-to-use workflows

These are normal ComfyUI canvas workflows. Download a JSON file and drag it onto the ComfyUI canvas. There are no API-format workflows in this repository. For editing workflows, select your own image in each `Load Image` node after importing.

- [Text-to-image](examples/t2i_workflow.json)
- [Thinking text-to-image](examples/thinking_t2i_workflow.json)
- [Interleaved text/image generation](examples/interleave_workflow.json)
- [GGUF text-to-image](examples/gguf_t2i_workflow.json)
- [GGUF image editing](examples/gguf_edit_workflow.json)
- [Batch text-to-image, two results by default](examples/batch_t2i_workflow.json)
- [8-step LoRA text-to-image](examples/t2i_8step_workflow.json)
- [Standard image editing, img_cfg=1](examples/edit_workflow.json)
- [Stable multi-reference editing, virtual try-on example](examples/multi_reference_edit_workflow.json)
- [SFT text-to-image](examples/sft_t2i_workflow.json)
- [SFT image editing](examples/sft_edit_workflow.json)

### Native ComfyUI core workflows

These two workflows target ComfyUI builds that include native SenseNova U1.5 core support and do not depend on this repository's custom loader:

- [Native core text-to-image](examples/core_t2i_workflow.json)
- [Native core image editing](examples/core_edit_workflow.json)

The core workflows use ComfyUI's built-in `CheckpointLoaderSimple`, so place the base checkpoint in `ComfyUI/models/checkpoints/`. The 8-step LoRA can use the built-in `LoraLoaderModelOnly`; keep the LoRA file in `ComfyUI/models/loras/`. The merged core implementation from [ComfyUI PR #15922](https://github.com/Comfy-Org/ComfyUI/pull/15922) reuses `EmptyHiDreamO1LatentImage` and `HiDreamO1ReferenceImages`.

This pack's thinking and interleave implementation follows [ComfyUI PR #16032](https://github.com/Comfy-Org/ComfyUI/pull/16032) while retaining the custom Final/SFT, GGUF and ConvRot-compatible loader paths. If that core PR is merged, prefer the native implementation unless you need this pack's specialized loaders.

### Thinking and interleaved generation

For thinking generation, use `SenseNova 1.x Text Encode` in `image` mode with
thinking enabled, then connect `SenseNova Thinking Preview` to the sampled
latent. For interleaved output, encode both branches in `interleave` mode and
use `SenseNova 1.x Interleave`; generated images are fed back into later turns.
`SenseNova Interleave Preview` preserves model text/image order and can include
or hide thinking text.

Start with these settings:

```text
steps: 50
CFG: 4
img_cfg: 1
shift: 3
sampler: euler
scheduler: normal
denoise: 1
```

`Empty SenseNova Pixel Latent` also exposes the official suggested resolution presets. Select `Custom` to keep using the width and height fields:

| Aspect ratio | Resolution |
|---|---:|
| 1:1 | 2048 × 2048 |
| 16:9 | 2720 × 1536 |
| 9:16 | 1536 × 2720 |
| 2:3 | 1664 × 2496 |
| 3:2 | 2496 × 1664 |

For more complex editing, begin with these values in `SenseNova Edit Guider`:

```text
CFG: 4
img_cfg: 1
cfg_norm: global
cfg_interval: 0 → 1
```

`global` CFG Norm pulls excessive guidance back toward the magnitude of the positive condition. It often reduces oversaturation, over-sharpening, and subject drift. Switch back to `none` if the edit becomes too conservative. `channel` normalizes each 32×32 generation token independently and can help with localized over-guidance.

`cfg_interval` uses ComfyUI's normalized denoising progress: `0` is the first step and `1` is the last. Both boundaries are inclusive. Keep `0 → 1` for full-time CFG, which matches the official default behavior.

Use the official settings for the 8-step LoRA:

```text
LoRA strength: 1
steps: 8
CFG: 1
cfg_norm: none
shift: 3
sampler: euler
scheduler: normal
denoise: 1
```

### Text-to-image

![SenseNova text-to-image workflow](docs/images/t2i-workflow.jpg)

The basic connection is:

```text
Loader → Sampling Options → KSampler → VAE Decode → Save Image
```

Always pass `MODEL` through `SenseNova Sampling Options` before sampling. Keep `shift` at `3` unless you intentionally want to experiment.

### Batch generation

Set `batch_size` in `Empty SenseNova Pixel Latent` to a value from `2` to `16`. Each result uses different noise while sharing the same prompt and reference set. `Save Image` saves every result separately.

VRAM use increases with batch size. The batch example uses `768×768, batch_size=2`. Do not begin with `2048×2048, batch_size=16`. On a 24 GB GPU, start with 512 or 768 pixels and a batch size of 2.

A full dual-reference editing test at `512×512, batch_size=2, 50 steps` took about 495 seconds. Both results followed the clothing-transfer instruction, and total GPU memory usage peaked at 22,986 MiB.

### 8-step LoRA text-to-image

The protected 8-step node still uses ComfyUI's native LoRA mapping and `ModelPatcher` internally:

```text
SenseNova Loader (Final) → SenseNova U1.5 8-Step LoRA → Sampling Options → KSampler
```

Keep LoRA strength at `1`. The official LoRA is intended for fast text-to-image generation. Use the regular 50-step workflow without the LoRA for image editing.

### Standard image editing

![SenseNova standard editing workflow](docs/images/edit-workflow.jpg)

Connect the reference image to `SenseNova Reference Image`; do not use it as the latent input. When `img_cfg=1`, the standard `KSampler` works. Connect the node's `image_condition` output to KSampler's negative input.

### Multiple references and custom guidance

The standard `SenseNova Reference Image` node exposes `Image-1` and an optional `Image-2`. Use `SenseNova Reference Images (1-10)` when you need 3 to 10 inputs.

Image order matches the labels used in the prompt. For virtual try-on or clothing transfer:

- Put the person or main scene in `Image-1`.
- Put the garment reference in `Image-2`.

Older workflows using legacy `images.image` socket names are migrated automatically. Older workflows with more than two references are also migrated to the 1-to-10-image node.

For complex edits, avoid vague prompts such as “make her wear this.” Use `SenseNova Structured Edit Prompt`, or write the same structure directly in `CLIP Text Encode`:

```text
[Main change] Make the person in Image-1 wear the garment from Image-2.
[Reference roles] Image-1 provides only the person; Image-2 provides only the garment. Do not copy the mannequin or background from Image-2.
[Must preserve] Keep the face, pose, lighting, background, and framing from Image-1 unchanged.
[Must avoid] Do not add another person and do not change unspecified regions.
```

When `img_cfg` is not 1, use `SenseNova Edit Guider` with ComfyUI's built-in `SamplerCustomAdvanced`. The same `MODEL` output from `Sampling Options` must be connected to both `Edit Guider` and `BasicScheduler`:

```text
SenseNova Sampling Options (MODEL)
├── SenseNova Edit Guider ───────────┐
└── BasicScheduler ──────────────────┤
RandomNoise + KSamplerSelect + Latent├──→ SamplerCustomAdvanced
                                     ┘
```

The node inserts official `Image-1`, `Image-2`, and later labels, and processes every reference at its own supported size. It does not concatenate the images into one strip.

## Real results

All images below were generated by this custom node. They were not color-graded or retouched afterward.

### 2048×2048 dual-reference clothing transfer

[Open the original 2048×2048 PNG](docs/images/result-garment-edit-2048.png)

![SenseNova U1.5 dual-reference clothing transfer](docs/images/result-garment-edit-2048.png)

Final checkpoint, 2048×2048, 50 steps, CFG 4, img_cfg 1, global CFG Norm, shift 3, Euler/normal, seed 31082026. Image-1 supplied the person, face, pose, and background; Image-2 supplied only the black-and-white garment. The result preserved the hand-on-chin pose and indoor composition while transferring the apron, ruffles, bow, and cuffs. It completed in about 506 seconds on an RTX 5090 Laptop GPU with 24 GB VRAM.

### U1.5 SFT: 2048×2048, 50-step text-heavy generation

[Open the original 2048×2048 PNG](docs/images/result-sft-t2i-2048.png)

![SenseNova U1.5 SFT Chinese fried-chicken infographic](docs/images/result-sft-t2i-2048.png)

SFT checkpoint, 2048×2048, 50 steps, CFG 4, shift 3, Euler/normal, seed 42. The title, ingredient amounts, three steps, and 170°C note were generated directly by the model without text correction. The run took about 297 seconds on an RTX 5090 Laptop GPU with 24 GB VRAM.

### 2048×2048, 8-step text-heavy generation

[Open the original 2048×2048 PNG](docs/images/result-t2i-8step-2048.png)

![SenseNova U1.5 8-step Chinese fried-chicken infographic](docs/images/result-t2i-8step-2048.png)

2048×2048, 8 steps, CFG 1, shift 3, LoRA strength 1, Euler/normal, seed 42. The title, subtitle, five ingredients, three steps, and temperature note were all generated by the model. The run took about 86 seconds on an RTX 5090 Laptop GPU with 24 GB VRAM.

### 2048×2048 text-to-image

[Open the original 2048×2048 PNG](docs/images/result-t2i-2048.png)

![SenseNova 2048 text-to-image result](docs/images/result-t2i-2048.png)

2048×2048, 50 steps, CFG 4, shift 3, Euler/normal, seed 42. Successfully tested on 24 GB VRAM.

### 2048×2048 dual-reference editing

[Open the original 2048×2048 PNG](docs/images/result-multi-reference-2048.png)

![SenseNova 2048 dual-reference result](docs/images/result-multi-reference-2048.png)

2048×2048, 50 steps, CFG 4, img_cfg 1, shift 3, Euler/normal, seed 42. Image-1 supplied the notebook layout and text density; Image-2 supplied the fried-chicken subject. The prompt requested a title, ingredients, three steps, and a tip. The large title and main sections are readable, while some small text still contains spelling errors and overlaps. The image was not corrected afterward.

## What the KV cache does

SenseNova uses the same text and reference-image prefix at every denoising step. `SenseNova Sampling Options` caches those prefix keys and values for the current execution, so later steps do not encode the same references again.

For batch generation, the text and reference prefix is computed once per guidance branch, and only the smaller per-layer KV data is expanded across generated variants. The complete reference-image encoder is not repeated `batch_size` times.

The cache exists only during the current job. It is cleared when the job finishes, fails, or is cancelled, so it does not keep VRAM allocated between jobs. Cached and uncached three-branch editing paths were verified to be element-wise identical.

## What if the colors are too saturated?

First check the prompt for words such as `bright`, `vivid`, `neon`, or `highly saturated`. They can strongly increase saturation.

Try the following:

- Start with `CFG 4`; try 3 to 3.5 if the image still looks over-guided.
- Keep `img_cfg` at 1 initially.
- Use `global` CFG Norm for complex edits or overcooked-looking images.
- Add `natural colors` or `restrained color grading` to the prompt.

## ConvRot quantized checkpoints

This is a fork-only, fully optional feature, ported from
[`Milor123/ComfyUI-SenseNova-U1.5-ConvRot`](https://github.com/Milor123/ComfyUI-SenseNova-U1.5-ConvRot)
(commit `7e1e320`); the upstream T8mars wrapper has no quantization support. It
activates only when a checkpoint carries the per-layer `comfy_quant` sidecars
that ComfyUI's quantized formats use; official bf16 files keep the exact upstream
validation path (including the strict file-size check), and nothing is installed
eagerly.

| Checkpoint | Approx. size | Formats | Published by Milor123 |
| --- | --- | --- | --- |
| INT8 + ConvRot | 17.6 GB | `int8_tensorwise` with `convrot: true` | ✅ `SenseNova-U1.5-8B-MoT-T8-int8-convrot-tagged.safetensors` |
| Hybrid W4A8 (L18-41) | 13.8 GB | `asym_w4a8_int8` on later layers, INT8 elsewhere | ✅ `SenseNova-U1.5-8B-MoT-T8-hybw4a8-L18-41.safetensors` |
| ConvRot W4A4 | 11 GB | `convrot_w4a4` | ❌ not published — convert it yourself |

The two published files are in
[`Milor123/ComfyUI-ConvRot-SenseNova-U1.5-8B-MoT-T8`](https://huggingface.co/Milor123/ComfyUI-ConvRot-SenseNova-U1.5-8B-MoT-T8)
(see [Download the models](#download-the-models)). They were converted from the
legacy mixed-precision Final, so this pack validates them as `final_legacy`.

Convert your own copy of an official file (needs the ComfyUI environment with
`comfy-kitchen`):

```bash
cd ComfyUI/custom_nodes/Comfyui-SenseNova-U1.5-Wrapper-T8-ANT-fork

# 1. quantize; --mode mixed is the quality-first recipe (o_proj/down_proj and the
#    conditioning MLPs stay INT8, everything else becomes ConvRot W4A4)
python tools/convert_sensenova_int4_convrot.py \
    -i ../../models/diffusion_models/SenseNova-U1.5-8B-MoT-BF16-T8.safetensors \
    -o ../../models/diffusion_models/SenseNova-U1.5-8B-MoT-int8.safetensors \
    --mode w4a8

# 2. tag the header with the SenseNova provenance metadata the loader requires;
#    --variant auto (the default) keeps the tags of the file you converted from,
#    which matters: a conversion of the legacy mixed-precision Final has to stay
#    final_legacy, a conversion of the all-BF16 Final has to stay final.
python tools/inject_sensenova_metadata.py \
    -i ../../models/diffusion_models/SenseNova-U1.5-8B-MoT-int8.safetensors\
    -o ../../models/diffusion_models/SenseNova-U1.5-8B-MoT-int8-tagged.safetensors\
    --variant final_legacy        # or 'auto' when the source file is already tagged

# 3. optional: build hybrid rungs from an INT8 and a W4A8 file
python tools/make_hybrid_ladder.py \
    --int8 ../../models/diffusion_models/...-int8-tagged.safetensors \
    --w4a8 ../../models/diffusion_models/...-w4a8-tagged.safetensors \
    --rungs hybw4a8-L18-41
```

Then point `SenseNova U1.5 Loader (Final / SFT)` at the tagged file — the same
loader node as for bf16. There is no separate "ConvRot loader": the per-layer
`comfy_quant` sidecars are what select the quantized path, so a quantized and a
bf16 file can never be loaded through the wrong code path.

### Confirming the quantized path

ComfyUI's console must show these lines while the model loads:

```text
Found quantization metadata version 1
Using mixed precision operations
[SenseNova-U1.5] quantized checkpoint detected (int8_tensorwise); loading with mixed-precision quantization ops.
[sensenova-u15] comfy-kitchen INT8 convrot probe on cuda:0: relative error 0.0131 rotated / 1.4142 unrotated -> honours convrot
[sensenova-u15] quantized weights: ComfyUI native mixed-precision operations (SENSENOVA_FORCE_BRIDGE=1 / SENSENOVA_NO_BRIDGE=1 to override)
```

The probe line is the interesting one: this node does not trust the
`comfy-kitchen` version or its function signature, it *measures* whether the INT8
kernel rotates activations, and only then lets ComfyUI's own ops handle the file.
If the kernel ignores the flag, the message says `ignores convrot` and this node's
own ConvRot forwards are used instead, so the weights are always evaluated in the
right basis. `SENSENOVA_NO_CONVROT_PROBE=1` skips the measurement and always takes
the bridge route.

`Found quantization metadata version 1` and `Using mixed precision operations` come
from ComfyUI core and are the important ones: without them the packed weights are
never un-packed, and the symptom is a regular square-pattern / checkerboard image
instead of an error, together with a long `unet unexpected: [... weight_scale,
... comfy_quant]` warning. This node refuses to load in that state and raises. If
your ComfyUI's `comfy-kitchen` is older than 0.2.31 (its INT8 kernel accepts but
ignores the convrot flags), set `SENSENOVA_FORCE_BRIDGE=1` to use this pack's own
ConvRot forwards.
 The loader
validates the derived contract (packed shapes, sidecar dtypes, per-layer formats)
before a single weight is read, and reports the active formats if something does
not line up. The 8-step LoRA keeps working on quantized Final checkpoints.

Environment switches — all of them only affect quantized loads:

| Variable | Effect |
| --- | --- |
| `SENSENOVA_NO_QUANT=1` | refuse quantized checkpoints; only the upstream bf16 contract runs |
| `SENSENOVA_NO_BRIDGE=1` | never install the ConvRot Linear ops, use stock ComfyUI ops |
| `SENSENOVA_FORCE_BRIDGE=1` | always install them (reproduces the original ConvRot fork numerics) |
| `SENSENOVA_NO_QT_GUARDS=1` | skip the `QuantizedTensor` cast guards |
| `SENSENOVA_NO_CONVROT_PROBE=1` | skip the load-time kernel measurement and always use this pack's ConvRot forwards |

By default the ConvRot bridge installs itself only when the running
ComfyUI/comfy-kitchen cannot rotate activations itself — and that decision is
measured on your GPU at load time, not read off a version number. ConvRot W4A4
and W4A8 weights always go through ComfyUI's own kernels, because the kitchen
layouts apply the rotation for those formats themselves.

If a quantized checkpoint is rejected with `quantized checkpoint key mismatch`,
re-run both steps of the conversion: the sidecar set differs between formats and
the metadata tag is required by this node pack.

## System requirements

Local and CI validation coverage:

- Local ComfyUI 0.33.x
- CI: minimum supported ComfyUI 0.31 and current stable ComfyUI v0.34.0
- Python 3.10, 3.12, 3.13, and 3.14
- NVIDIA CUDA with BF16 support
- RTX 5090 Laptop GPU, 24 GB VRAM
- 64 GB system RAM

2048×2048 50-step text-to-image generation, dual-reference editing, and `512×512, batch_size=2` full-model batch execution all completed on 24 GB VRAM. Loading and offloading the model also uses substantial system memory. 64 GB RAM and enough virtual memory are recommended.

## Current limitations

- Only NVIDIA CUDA with BF16 has been fully validated.
- Models are not downloaded automatically at runtime.
- Quantized checkpoints can be loaded (see below); quantizing a checkpoint from
  inside ComfyUI, bbox/marker controls, and think mode are still not exposed.
- Complex subject replacement, multi-region edits, and heavily constrained edits can drift.
- FP16, ROCm, MPS, DirectML, XPU, and NPU have not been validated.

## Model verification

Current BF16 Final checkpoint (recommended):

```text
File: SenseNova-U1.5-8B-MoT-BF16-T8.safetensors
Size: 35,065,860,328 bytes
SHA256: a32b117f40ad4575c6709b3ad6efb1c6b743ef1c1c3d75360f14090b997f1d29
Official revision: 19bc874ef6ffc97fda9837b40fc1d1301806158a
Tensors: 1116, all stored as BF16
```

Legacy mixed-precision Final checkpoint (still supported):

```text
File: SenseNova-U1.5-8B-MoT-T8.safetensors
Size: 50,222,155,152 bytes
SHA256: 2e5c4451969a8af9d7bcbf9d00a0fe463b15ed44149d8d79f31409e671587615
Tensors: 1116
Source revision: 1f6ec60423d29939dde4202fd82ae340b144e280
```

SFT checkpoint:

```text
Size: 35,065,860,320 bytes
SHA256: 9c105bb4baaf244bbd99f814c36f190228c5878f8889295e3dba285441442f2f
Tensors: 1116, all stored as BF16
Source revision: 661834c5b5aee0f89958353511d6ac0ccaacb646
```

Quantized checkpoints published by Milor123 — both were converted from the
legacy mixed-precision Final, so they validate as `final_legacy`:

```text
File: SenseNova-U1.5-8B-MoT-T8-int8-convrot-tagged.safetensors
Size: 18,872,613,872 bytes (17.58 GiB)
SHA256: f49eb10fd8c51f172a69788a3fe0e68534f69b3e04f7a45b0f042f46aa7da855
Format: int8_tensorwise, convrot
```

```text
File: SenseNova-U1.5-8B-MoT-T8-hybw4a8-L18-41.safetensors
Size: 14,821,019,712 bytes (13.80 GiB)
SHA256: 72551898faee2aeada901e407e76987d1de68a323e4d16cccdb02ded471a73db
Formats: int8_tensorwise for layers 0-17, asym_w4a8_int8 for layers 18-41
```

The loader distinguishes current Final, legacy Final, and SFT files and checks metadata, exact file size, all tensor names, shapes, and each profile's storage dtype. Invalid, incomplete, and unsupported checkpoints fail with a clear error instead of loading silently.

Quantized files skip the exact-file-size rule (their size depends on the
conversion recipe); they are validated through the derived contract — tensor
names, shapes, per-layer formats and sidecar dtypes — and by the post-load
`QuantizedTensor` invariant. The hashes above are for verifying your download.

### If you see `tokenizer asset digest mismatch`

On Windows this was usually a checkout artifact: `core.autocrlf=true` rewrites the
packaged text files to CRLF, and the loader used to hash those bytes directly. The
fork now compares the LF-normalised digest too and prints a note instead of
refusing to load. To get byte-identical files back:

```powershell
git config --global core.autocrlf false
git rm --cached -r . ; git reset --hard
```

A mismatch that survives normalisation is real: compare the printed
`got=`/`lf_normalized=` values with `sha256sum` (or `Get-FileHash`) for the file
named in the message, and re-clone if they differ.

### If you see `checkpoint key mismatch`

Update this custom node to version `1.3.5` or newer, completely close ComfyUI, and start it again. Do not modify the loader, remove reported keys, or disable dynamic model loading to bypass verification. Those workarounds may allow the model to run with incorrect weights, causing blurred output, unusual colors, or poor prompt following.

If the error remains:

- Check `ComfyUI/custom_nodes/` for duplicate copies of this custom node.
- Compare your model's exact size and SHA256 with the values above.
- Keep the full error message. Newer versions include the actual `model=` and `loader=` paths, which make stale installations easy to identify.

8-step LoRA verification:

```text
Official source: sensenova/SenseNova-U1.5-8B-MoT-LoRAs
Source revision: e909f4636d119d65fe4cba8770c19daff2ac102e
Official file SHA256: 3ef32180cdf1e30a870a83f4f136e897ea50b7ee467f863d75633464ebb25708
ComfyUI file SHA256: dd5320f06986688dd41b0a4a2cb6ebd0036308f8a8a2d0c349ca22875a805aa1
Modules: 294
Tensors: 882
```

The conversion only adds the `diffusion_model.` prefix required by ComfyUI. All LoRA tensor data remains byte-for-byte identical. Most users should download the converted file. Advanced users who cloned the source repository can also run [`tools/convert_lora_to_comfy.py`](tools/convert_lora_to_comfy.py).

Manual hash checks on Windows:

```powershell
Get-FileHash .\SenseNova-U1.5-8B-MoT-BF16-T8.safetensors -Algorithm SHA256
Get-FileHash .\SenseNova-U1.5-8B-MoT-T8.safetensors -Algorithm SHA256
Get-FileHash .\SenseNova-U1.5-8B-MoT-SFT-T8.safetensors -Algorithm SHA256
Get-FileHash .\SenseNova-U1.5-8B-MoT-LoRA-8step-ComfyUI.safetensors -Algorithm SHA256
```

## Credits and provenance

Three codebases make up this repository (the model itself comes from
OpenSenseNova): the node pack is **T8mars'**, the quantization stack is
**Milor123's**, and the items listed under “What this fork changed” are our own
work on top of both. Both upstream projects are Apache-2.0 — please star and
support them, this fork would not exist without either.

### Lineage

```text
OpenSenseNova/SenseNova-U1                     model + reference implementation (Apache-2.0)
└─ T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8   ComfyUI wrapper, Apache-2.0 — features synced through 1.5.2
   ├─ Milor123/ComfyUI-SenseNova-U1.5-ConvRot        ConvRot quantization fork, Apache-2.0 (7e1e320)
   └─ Spirindzhiukas/...-Wrapper-T8-ANT-fork         this repository
```

This repository is a direct GitHub fork of **T8mars'** wrapper, not of Milor123's
fork: it started from upstream 1.3.6, Milor123's quantization code was ported
into it file by file, and subsequent T8mars fixes are reviewed and integrated.
The synchronization history is documented in
[`docs/UPSTREAM_SYNC_1.3.7.md`](docs/UPSTREAM_SYNC_1.3.7.md) and
[`docs/UPSTREAM_SYNC_1.5.2.md`](docs/UPSTREAM_SYNC_1.5.2.md).

### What comes from T8mars

[`T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8`](https://github.com/T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8)
(Apache-2.0) provides the entire ComfyUI integration. It is kept as upstream
wrote it unless a fix required a change:

| Area | Files |
| --- | --- |
| Model, conditioning, guidance, sampling, text encoder, LoRA | `sensenova_u15/model.py`, `conditioning.py`, `guidance.py`, `sampling.py`, `text_encoder.py`, `lora.py`, `model_config.py` |
| bf16 loading and the 1116-tensor checkpoint contract | `sensenova_u15/loader.py` (bf16 path), `sensenova_u15/checkpoint_contract.json` |
| Packaged tokenizer | `sensenova_u15/tokenizer/*` |
| The nodes, their V3 schemas and the Manager node list | `nodes.py`, `node_list.json` |
| Frontend label extension | `web/sensenova_reference_labels_v131e.js` |
| Example workflows | `examples/*.json` |
| Checkpoint and LoRA tooling | `tools/build_checkpoint_contract.py`, `tools/convert_lora_to_comfy.py`, `tools/merge_safetensors.py`, `tools/validate_lifecycle.py`, `tools/validate_native.py`, `tools/run_upstream_oracle.py`, `tools/run_upstream_edit_oracle.py`, `tools/analyze_trace.py`, `tools/analyze_module_trace.py` |
| GGUF loading and dequantization | `sensenova_u15/gguf_support.py`, `gguf_dequant.py`; adapted upstream from City96/ComfyUI-GGUF with attribution in `NOTICE` |
| Thinking and interleaved generation | `sensenova_u15/interleave.py`, LM-head/decode additions in `model.py`, new nodes/workflows and `web/sensenova_interleave_preview.js`; adapted upstream from ComfyUI PR #16032 |
| Test suite | `tests/*`, except the three `test_fork_*` files we added |
| Documentation images | `docs/images/*` |
| Upstream fixes this fork deliberately keeps | the **1.3.4 pure-PyTorch split-half RoPE** (Blackwell / CUDA 13 safe), the **1.3.5 checkpoint contract** with file-size, storage-dtype and legacy-revision validation, and the **1.3.7 prefix-mask dtype fix** for BF16 PyTorch SDPA |

### What was ported from Milor123

[`Milor123/ComfyUI-SenseNova-U1.5-ConvRot`](https://github.com/Milor123/ComfyUI-SenseNova-U1.5-ConvRot)
at commit `7e1e320` (Apache-2.0) was the only implementation that could run
SenseNova U1.5 quantized, and every quantized code path here comes from it:

| Area | Files in this fork | What it does |
| --- | --- | --- |
| ConvRot-aware Linear forwards | `sensenova_u15/quant_bridge.py` | rotates activations and routes convrot weights through the comfy-kitchen kernels, so checkpoints with the rotation folded offline are evaluated in the right basis |
| Quantized-tensor cast guards | `sensenova_u15/qt_guards.py` | keeps `QuantizedTensor`s pristine when a LoRA patches a layer or a manual cast is requested; relocated from the repo root into the package |
| Quant detection and per-format contract | `sensenova_u15/loader.py`: `QUANT_*`, `_is_quant_candidate`, `_read_quant_formats`, `_quant_checkpoint_contract`, `_expected_storage_dtype` | Milor123's logic, rebuilt on top of T8's JSON contract (see below) |
| Ops hook for quantized loads | `sensenova_u15/model_config.py::get_model` | installs `custom_operations` when `comfy_quant` sidecars are present |
| Conversion tooling | `tools/convert_sensenova_int4_convrot.py`, `tools/make_hybrid_ladder.py`, `tools/inject_sensenova_metadata.py` | hard-coded Windows paths removed |
| Published quantized weights | [`Milor123/ComfyUI-ConvRot-SenseNova-U1.5-8B-MoT-T8`](https://huggingface.co/Milor123/ComfyUI-ConvRot-SenseNova-U1.5-8B-MoT-T8) | INT8-ConvRot and hybrid W4A8 checkpoints, including the empirically found L18-41 layer boundary |

We did **not** take Milor123's `sensenova_u15/checkpoint_contract.py` (a Python
`BASE_CONTRACT` dict that replaced upstream's JSON contract): it knows only one
variant and drops the file-size, `final_legacy` and SFT-revision checks upstream
added. Quantized validation here is derived from T8's `checkpoint_contract.json`
instead, and was cross-checked to produce the same storage-dtype rules as
Milor123's on all 1116 tensors for the `final_legacy` profile.

### What this fork changed

| Change | Where | Why |
| --- | --- | --- |
| CRLF-tolerant tokenizer digest (raw **and** LF-normalised bytes; warn instead of abort) plus an LF pin in `.gitattributes` | `sensenova_u15/loader.py::_tokenizer_digest_kind`, `_validate_tokenizer_assets` | a Windows clone with `core.autocrlf=true` used to abort with `tokenizer asset digest mismatch` |
| English UI: node labels, slot names, structured-prompt defaults, frontend label, example workflows | `nodes.py`, `sensenova_u15/guidance.py`, `web/*.js`, `examples/*.json` | upstream is Chinese; the original wording is kept as comments and docstrings so upstream merges stay reviewable |
| English-first documentation layout (`README.md` = English, `README_CN.md` = Chinese) | this file, `README_CN.md` | upstream keeps the Chinese document as `README.md` and English as `README_EN.md` |
| Quant support gated behind `comfy_quant` detection; file-size check kept for bf16 only | `sensenova_u15/loader.py` | keeps the bf16 path byte-identical to upstream, which is what makes future upstream merges cheap |
| Quantization wiring: `detect_quant_config` → `model_config.quant_config`, `comfy.utils.convert_old_quants`, and a post-load `QuantizedTensor` invariant | `sensenova_u15/loader.py::detect_quant_config`, `_validate_quantized_weights_loaded` | without it ComfyUI picks plain `Linear` ops, treats packed int8 as the weight and renders a checkerboard while logging nothing fatal |
| Measured INT8 ConvRot capability probe instead of a version or signature check | `sensenova_u15/quant_bridge.py::kitchen_honours_int8_convrot` | comfy-kitchen 0.2.28 and 0.2.31 both *accept* `convrot`, only 0.2.31 applies it; an ignored flag is again a silent wrong-basis image |
| Capability-aware bridge and lazy guard installation | `quant_bridge_needed` / `core_supports_convrot`; `qt_guards` installed from `get_model` instead of at package import | modern ComfyUI keeps its own kernels, and bf16 sessions never pay for quant hooks |
| `--variant auto` in the metadata injector | `tools/inject_sensenova_metadata.py::detect_variant` | Milor123's published quants come from the legacy Final and must stay `final_legacy`, or the non-quantized tensors fail their dtype check |
| Per-axis RoPE bases readable through `transformer_options` and included in the prefix-cache key | `sensenova_u15/model.py::resolve_rope_thetas` | preparation for ANT RoPE Lab context-scaling experiments ([`docs/rope_lab_integration.md`](docs/rope_lab_integration.md)) |
| Windows path cleanups in the ported tools | `tools/convert_sensenova_int4_convrot.py`, `tools/make_hybrid_ladder.py` | the originals carried hard-coded paths |
| Fork tests, maintenance contract and research docs | `tests/test_fork_quant_checkpoint.py`, `tests/test_fork_tokenizer_assets.py`, `tests/test_fork_rope_theta.py`, `memory.md`, `docs/*` | regression tripwires for the silent-failure modes above |

### Also credited

- **ConvRot** — Hanzuliang et al.; technical notes in [Comfy-Org/ComfyUI#14735](https://github.com/Comfy-Org/ComfyUI/issues/14735).
- **[Comfy-Org/comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen)** — the quantized tensor runtimes both Milor123's fork and this one build on.
- **[silveroxides/convert_to_quant](https://github.com/silveroxides/convert_to_quant)** — converter lineage behind the quantization scripts (via Milor123).
- **ComfyUI (Comfy-Org)** — used through its public model, node, sampling, attention, operations and model-management interfaces.

## Upstream author links

**T8mars / T8star** — author of the upstream wrapper this fork is based on:

- [Wrapper repository](https://github.com/T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8)
- [Model repository: `t8star/SenseNova-U1.5-Comfy`](https://huggingface.co/t8star/SenseNova-U1.5-Comfy/) — bf16 and official-precision checkpoints
- [Hugging Face profile](https://huggingface.co/t8star)
- [Bilibili](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/)
- [Model package](https://pan.quark.cn/s/264edb7e36bd) · [Model mirror](https://pan.quark.cn/s/6b756fdae32d)
- [AI API](https://api.seedance.nz/sign-up?aff=5f4w) · [Online AI applications](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121)

**Milor123** — author of the ConvRot quantization fork and of the quantized checkpoints:

- [ConvRot fork repository](https://github.com/Milor123/ComfyUI-SenseNova-U1.5-ConvRot)
- [Model repository: `Milor123/ComfyUI-ConvRot-SenseNova-U1.5-8B-MoT-T8`](https://huggingface.co/Milor123/ComfyUI-ConvRot-SenseNova-U1.5-8B-MoT-T8) — INT8-ConvRot and hybrid W4A8 checkpoints

## Source and license

SenseNova U1.5 and its reference implementation come from [OpenSenseNova/SenseNova-U1](https://github.com/OpenSenseNova/SenseNova-U1), licensed under Apache License 2.0.

- [Official U1.5 Final](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT)
- [Official U1.5 SFT](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT-SFT)
- [Official U1.5 LoRAs](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT-LoRAs)

This repository is a fork of [T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8](https://github.com/T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8) (Apache License 2.0) and contains code ported from [Milor123/ComfyUI-SenseNova-U1.5-ConvRot](https://github.com/Milor123/ComfyUI-SenseNova-U1.5-ConvRot) (Apache License 2.0). Both are credited in [Credits and provenance](#credits-and-provenance) and in [NOTICE](NOTICE), and this fork is released under the same [Apache License 2.0](LICENSE).

This repository provides the local ComfyUI integration only. It does not contain model weights, and it is not published on the ComfyUI Registry. Weights are distributed by their own repositories under their own licenses — see [Download the models](#download-the-models).
