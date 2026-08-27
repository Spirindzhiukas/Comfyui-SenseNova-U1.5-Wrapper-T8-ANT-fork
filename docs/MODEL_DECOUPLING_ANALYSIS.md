# SenseNova-U1.5-8B-MoT Checkpoint Decoupling Analysis

## TL;DR

- **VAE**: Already decoupled — it's a **dummy pixel-space VAE** (`HiDreamO1Pixel`, 3-channel RGB). No weights, just `{"pixel_space_vae": 1.0}`. Splitting is trivial.
- **Text Encoder**: **Not a separate model** in the classic SD/Flux sense. It's a **tokenizer-only** wrapper (`Qwen2Tokenizer` + system prompt builder). All LLM weights live *inside* the diffusion checkpoint.
- **Diffusion Model**: Contains **everything**: 
  - Qwen3-8B backbone (embed_tokens + 42 layers)
  - Dual weights per layer: `*_mot_gen` for image generation, normal for text/reference prefix
  - Vision encoders for reference images
  - Flow head (ConvDecoder)

**Can it be split into 3 files?** Yes, but not in the usual `clip_l + t5xxl + dit` way. You need to split the **MoT** (Mixture-of-Transformers) into:
1. `sensenova_prefix_encoder` (LLM part: embed_tokens + normal path)
2. `sensenova_dit` (generation part: vision_model_mot_gen + timestep_embedders + fm_head + mot_gen path)
3. `sensenova_vae` (dummy, already separate)

This is **possible** and useful for VRAM cleanup and rope/embedding manipulation, but requires custom loader and KV-cache handoff.

---

## 1. What the checkpoint actually contains

From `checkpoint_contract.py` (official Final revision):

```
fm_modules.fm_head.conv1/conv2          -> Flow decoder (PixelShuffle + Conv)
fm_modules.noise_scale_embedder         -> Resolution-aware noise scale
fm_modules.timestep_embedder            -> Time embedding
fm_modules.vision_model_mot_gen         -> Patch embed for noisy image (x)
language_model.lm_head.weight           -> Unused, popped in process_unet_state_dict
language_model.model.embed_tokens       -> Qwen3 vocab 151936 x 4096 (1.2B params, ~2.4GB bf16)
language_model.model.layers.0-41:
  - input_layernorm                     -> Prefix path norm
  - input_layernorm_mot_gen             -> Generation path norm
  - self_attn.q/k/v/o_proj              -> Prefix attention
  - self_attn.q/k/v/o_proj_mot_gen      -> Generation attention
  - self_attn.q_norm, k_norm, q_norm_hw, k_norm_hw (and _mot_gen variants) -> QK RMSNorm
  - post_attention_layernorm / _mot_gen
  - mlp.gate/up/down_proj / _mot_gen
language_model.model.norm / norm_mot_gen
vision_model.embeddings                 -> Reference image encoder (patch 16 + dense 2)
```

**No VAE weights.** `process_vae_state_dict` returns dummy tensor.

### Model class (`model.py`)

```python
class SenseNovaU15:
  vision_model = VisionModel (reference images)
  language_model = LanguageModel (embed_tokens + 42 DecoderLayers)
  fm_modules = {
    vision_model_mot_gen,
    timestep_embedder,
    fm_head: ConvDecoder,
    noise_scale_embedder
  }
```

Forward:
1. `prefix = embed_tokens(text_ids)` + `vision_model(ref_images)` inserted at `IMAGE_CONTEXT_ID`
2. For each layer: `forward_prefix` (normal weights) -> produces `prefix_key, prefix_value` (KV cache)
3. `forward_generation` (mot_gen weights) processes noisy image `x` + `prefix KV` + `time embedding` -> velocity

### Text Encoder (`text_encoder.py`)

```python
class SenseNovaQwenTokenizer:  # Wraps Qwen2Tokenizer from tokenizer/ folder
class SenseNovaTokenizer:      # Builds chat template with <think> etc.

class SenseNovaTextEncoder:
  def encode_token_weights:
    input_ids = tensor([[ids]])
    return input_ids, None, {"text_input_ids": input_ids}
  def load_sd: return []
```

**Zero weights.** It just tokenizes and passes `text_input_ids` as extra cond.

---

## 2. Why classic 3-file split doesn't directly apply

Standard SDXL/Flux split:
- `clip_l.safetensors` -> text embeddings
- `t5xxl.safetensors` -> text embeddings  
- `flux1-dev.safetensors` -> DiT

SenseNova is **single-transformer MoT**:
- Text understanding and image generation share the *same* 42 layers, but with *different* weight sets per layer (normal vs mot_gen).
- KV cache from prefix pass is fed into generation pass *inside same forward*.
- You cannot run text encoder alone to get embeddings that DiT consumes, because DiT needs the *KV cache* from text encoder, not just final hidden state.

So decoupling requires:
- Splitting each `DecoderLayer` into two halves
- Keeping KV cache communication

---

## 3. Proposed split design (useful for VRAM cleanup & manipulation)

### File 1: `sensenova_prefix_encoder.safetensors` (LLM)

Contains:
```
language_model.model.embed_tokens.weight
language_model.model.layers.*.input_layernorm.weight
language_model.model.layers.*.self_attn.q_proj.weight
language_model.model.layers.*.self_attn.k_proj.weight
language_model.model.layers.*.self_attn.v_proj.weight
language_model.model.layers.*.self_attn.o_proj.weight
language_model.model.layers.*.self_attn.q_norm.weight
language_model.model.layers.*.self_attn.k_norm.weight
language_model.model.layers.*.self_attn.q_norm_hw.weight
language_model.model.layers.*.self_attn.k_norm_hw.weight
language_model.model.layers.*.post_attention_layernorm.weight
language_model.model.layers.*.mlp.gate_proj.weight
language_model.model.layers.*.mlp.up_proj.weight
language_model.model.layers.*.mlp.down_proj.weight
language_model.model.norm.weight
vision_model.embeddings.*  # reference image encoder belongs to prefix side
```

Size: ~50% of model (~25GB bf16, ~12.5GB int8). This is the part that can be **offloaded after KV cache is computed**.

### File 2: `sensenova_dit.safetensors` (Generation Transformer)

Contains:
```
fm_modules.vision_model_mot_gen.*
fm_modules.timestep_embedder.*
fm_modules.noise_scale_embedder.*
fm_modules.fm_head.*
language_model.model.layers.*.input_layernorm_mot_gen.weight
language_model.model.layers.*.self_attn.q_proj_mot_gen.weight
language_model.model.layers.*.self_attn.k_proj_mot_gen.weight
language_model.model.layers.*.self_attn.v_proj_mot_gen.weight
language_model.model.layers.*.self_attn.o_proj_mot_gen.weight
language_model.model.layers.*.self_attn.q_norm_mot_gen.weight
... (all _mot_gen)
language_model.model.layers.*.post_attention_layernorm_mot_gen.weight
language_model.model.layers.*.mlp_mot_gen.*
language_model.model.norm_mot_gen.weight
```

Size: ~50% + small fm_modules. This stays on GPU during sampling.

### File 3: `sensenova_vae.safetensors` (Dummy)

Already dummy. Could be empty or `pixel_space_vae: 1.0`. No VRAM.

### Benefits for workflows

1. **VRAM/RAM cleanup between nodes**: 
   - Compute prefix KV once, save to disk or RAM, unload prefix encoder (free ~12GB int8)
   - Run sampling with only DiT (~13GB int8) — fits 12GB VRAM with ComfyUI streaming even better
   - Re-load prefix only if prompt changes

2. **RoPE manipulation nodes**:
   - RoPE is in `model.py:_apply_llm_rope` with `theta=5_000_000` for time and `10_000` for HW
   - If prefix encoder is separate, you can expose `rope_theta`, `rope_theta_hw` as inputs
   - Could add `SenseNovaRoPEPatch` node that patches `q_norm` etc.

3. **Embeddings manipulation**:
   - Currently embeddings are `embed_tokens(text_ids)` — you could intercept after this
   - Add node `SenseNovaEmbeddingManipulation` that takes `text_input_ids` cond and returns modified embeddings (e.g., style transfer, prompt weighting, etc.)
   - Since text encoder is tokenizer-only, you need access to embed_tokens weight — which would be in prefix encoder file

---

## 4. Is it worth it? Challenges

**Pros:**
- Better VRAM control than ComfyUI's generic offloading
- Enables new research: RoPE scaling, KV cache editing, prefix embedding surgery
- Matches how user thinks: "LLM + DiT + VAE"

**Cons:**
- **Quantized checkpoints** (`int8-convrot-tagged`, `hybw4a8`) have sidecar keys (`weight_scale`, `comfy_quant` JSON). Splitting must preserve those.
- **KV cache format**: `prefix_key, prefix_value` are per-layer, shape `[B, KV_HEADS, L, HEAD_DIM]`. Need to define serialization.
- **ComfyUI integration**: Need custom `ModelPatcher` that holds two models and orchestrates prefix -> DiT handoff. Current `ComfyExtension` loads single MODEL, CLIP, VAE.
- **Reference images**: `vision_model` is tied to prefix encoder, but `vision_model_mot_gen` is tied to DiT — split already natural.

**Alternative (simpler) approach:**
Don't split files, but add nodes that **patch** existing model:
- `SenseNovaRoPEPatch`: patches `model.py` Attention to use custom theta
- `SenseNovaEmbeddingPatch`: hooks into `embed_tokens` forward to allow manipulation
- `SenseNovaKVCacheSaver`: saves KV cache to disk, allows unloading prefix part via `model_management` tricks

This gives 80% of benefits without file split.

---

## 5. Prototype splitting tool

I drafted `tools/split_sensenova.py` (see workspace) that:
- Reads official or quantized checkpoint via `safe_open`
- Splits keys by suffix `_mot_gen` vs normal
- Writes two safetensors + dummy VAE
- Preserves quantization sidecars (`weight_scale`, `comfy_quant`, `weight_s_rel`, etc.)
- Keeps metadata (`format`, `config_sha256`, etc.)

Usage:
```bash
python tools/split_sensenova.py \
  --input ComfyUI/models/diffusion_models/SenseNovaU1.5/SenseNova-U1.5-8B-MoT-T8-int8-convrot-tagged.safetensors \
  --output-dir split/
# Produces:
# split/sensenova_prefix_encoder.safetensors
# split/sensenova_dit.safetensors
# split/sensenova_vae.safetensors
```

Then you'd need a new loader node:
```python
class SenseNovaSplitLoader:
  def load(prefix_path, dit_path):
    prefix_sd = load_torch_file(prefix_path)
    dit_sd = load_torch_file(dit_path)
    merged = {**prefix_sd, **dit_sd}
    # ... same as load_sensenova_model but from merged dict
```

For true separate VRAM management, you'd need to keep them as two ModelPatchers and implement custom `forward_prefix` that runs on prefix model, caches KV, then `forward_generation` runs on DiT model.

---

## 6. Recommendation

For your use case (VRAM cleanup, RoPE, embeddings):

**Phase 1 (no split, minimal code):**
- Add 3 new nodes to existing wrapper:
  1. `SenseNovaPrefixCache` - computes KV once, caches, allows unloading LLM part via `comfy.model_management` soft empty cache
  2. `SenseNovaRoPEConfig` - exposes `rope_theta=5M`, `rope_theta_hw=10k`, `rope_theta_vision=10k` as patchable
  3. `SenseNovaEmbeddingEdit` - takes `text_input_ids` cond, allows adding/subtracting embedding vectors

**Phase 2 (split files, advanced):**
- Use `split_sensenova.py` to produce 2 files
- Create `SenseNovaSplitLoader` that loads them as two separate models but merges for compatibility
- Implement `SenseNovaUnloader` node that calls `comfy.model_management.unload_all_models()` or `model_patcher.unload()` between workflow sections

**VAE**: No action needed — it's already pixel-space dummy. If you want a real VAE for latent manipulation, you'd need to train one, but not in original model.

---

## 7. References

- Original model: https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT (50GB bf16)
- Quantized: https://huggingface.co/Milor123/ComfyUI-ConvRot-SenseNova-U1.5-8B-MoT-T8
- Architecture: Qwen3-8B (42 layers, 4096 hidden, 32 heads) + MoT dual weights
- Upstream wrapper: https://github.com/T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8
- This fork: ConvRot quantization + QT guards

---

*Prepared after fixing tokenizer digest bug and removing Chinese UI labels.*
