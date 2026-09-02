# Upstream synchronization report — T8mars 1.5.2

**Reviewed/integrated:** 2026-09-03
**Upstream:** [`T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8`](https://github.com/T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8)
**Upstream range:** 1.3.7 (`5836acbc`) → 1.5.2 (`9ac8182c`)
**Fork release:** 1.6.0

## Changes reviewed

| Upstream PR | Release | Result in this fork |
| --- | --- | --- |
| [#7](https://github.com/T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8/pull/7) | 1.3.8 | Integrated native workflows updated for merged ComfyUI core PR #15922 (`EmptyHiDreamO1LatentImage` and `HiDreamO1ReferenceImages`). |
| [#8](https://github.com/T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8/pull/8) | 1.4.0 | Integrated the dedicated, strictly validated GGUF loader, dequantizers, workflows, tests, documentation and City96 attribution. |
| [#9](https://github.com/T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8/pull/9) | 1.4.1 | Integrated the guidance to use a different seed when editing a SenseNova-generated image. |
| [#10](https://github.com/T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8/pull/10) | 1.5.0 | Integrated LM-head thinking, token preview, interleaved text/image generation, generated-image KV feedback, workflows and tests. |
| [#11](https://github.com/T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8/pull/11) | 1.5.1 | Integrated frontend preview compatibility for ComfyUI 0.33/frontend 1.49. |
| [#12](https://github.com/T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8/pull/12) | 1.5.2 | Integrated numbered-image placement and duplicate-reference handling in interleave previews. |

Upstream PR #10 adapts work proposed in [ComfyUI PR #16032](https://github.com/Comfy-Org/ComfyUI/pull/16032). At review time PR #16032 is still open; [ComfyUI PR #15922](https://github.com/Comfy-Org/ComfyUI/pull/15922), which supplies native basic SenseNova generation/editing, is merged.

## Integration strategy

Upstream's 1.5.2 implementation is the base for model, conditioning, text encoding, GGUF and interleave behavior. Fork functionality is applied as narrow overlays rather than retaining the older 1.3.7 implementation:

1. `nodes.py`, `model.py`, `model_config.py`, `conditioning.py` and `text_encoder.py` start from upstream 1.5.2.
2. ConvRot operation selection is added only inside `SenseNovaModelConfig.get_model` and only when quantization sidecars are detected.
3. Upstream's `language_model.lm_head.weight` is retained. The old fork behavior that removed it was deliberately not carried forward because thinking and interleave require it.
4. English labels/defaults are reapplied after taking upstream node changes.
5. The upstream model paths are retained for prefix encoding, autoregressive decode, image feedback, and diffusion generation. A short-lived experimental theta compatibility bridge was removed in fork release 1.6.1 because it had no repository consumer.

## New upstream functionality

### GGUF

The following Final profiles are supported by `SenseNova U1.5 GGUF Loader (Final)`:

- Q2_K
- Q3_K_M
- Q5_K_M
- Q6_K
- Q8_0

The implementation validates exact profile metadata, tensor names, shapes and quantization types before model construction. It uses standard ComfyUI MODEL/CLIP/VAE, ModelPatcher LoRA, sampling and offloading interfaces. `gguf>=0.13.0` is now a package dependency.

### Thinking and interleave

The model now retains and loads the LM head. New nodes support:

- image prompting with optional autoregressive thinking;
- decoding generated thinking tokens after sampling;
- interleaved text and image events;
- feeding each generated image into positive and negative live KV prefixes;
- ordered text/thinking/image preview and Markdown output;
- cancellation and bounded token/image generation.

## Fork invariants preserved

### ConvRot

Still present and gated by `*.comfy_quant` detection:

- strict derived quantized checkpoint contract;
- `quant_config` wiring and legacy quant conversion;
- post-load `QuantizedTensor` invariant;
- measured `comfy-kitchen` INT8 ConvRot capability probe;
- fallback ConvRot operations;
- quantized-tensor cast guards;
- INT8, W4A4, W4A8 and mixed-layer conversion tools.

GGUF does not contain ComfyUI quant sidecars, so it does not activate the ConvRot bridge. Plain BF16 behavior likewise remains on upstream operations.

### Correctness and compatibility

- PyTorch split-half RoPE remains Blackwell/CUDA-13 safe.
- Prefix attention masks are cast to query dtype.
- Tokenizer validation remains CRLF tolerant.
- Checkpoint contract and digest pins are unchanged.
- User-visible custom-node strings and shipped custom workflows remain English.
- The two native workflows now match merged ComfyUI core node names.

### RoPE and prefix cache

The upstream pure-PyTorch RoPE implementation covers normal prefix encoding,
thinking-token decode, interleave image feedback, reference vision embedding, and
diffusion image embedding. The independent prefix cache remains enabled for KV
reuse. Fork release 1.6.1 removed the unused experimental theta compatibility
bridge and its extra cache-identity fields without changing either behavior.

## Files adopted from upstream

- `sensenova_u15/gguf_dequant.py`
- `sensenova_u15/gguf_support.py`
- `sensenova_u15/interleave.py`
- `web/sensenova_interleave_preview.js`
- `examples/gguf_*`, `examples/thinking_t2i_workflow.json`, `examples/interleave_workflow.json`
- `tests/test_gguf.py`, `tests/test_interleave.py`
- `docs/gguf-validation.md` and its two validation images

The existing upstream-derived files listed above were integrated with fork overlays where required.

## Version decision

Upstream ends at 1.5.2 while this fork was already on its own 1.4.x line. This combined feature synchronization is released as **1.6.0**, avoiding collision with upstream tags and accurately signaling a feature-bearing fork release.

## Validation

Validated against ComfyUI v0.34.0 with its CPU dependencies and `gguf>=0.13.0`:

```text
160 tests passed, 1 skipped
Ruff passed
compileall passed
git diff --check passed
```

The fork regression tests were also corrected to avoid allocating the official
~2.5 GB F32 LM head in a tiny safetensors header test, to construct current
ComfyUI `QuantizedTensor` objects with their required parameter metadata, and
to mock quant-header inspection explicitly. These changes test the same
contracts without changing production behavior.

In addition to no-weight tests, release validation should exercise BF16, a published ConvRot INT8 or hybrid checkpoint, one GGUF profile, thinking generation and one interleave image-feedback cycle in a real ComfyUI installation.
