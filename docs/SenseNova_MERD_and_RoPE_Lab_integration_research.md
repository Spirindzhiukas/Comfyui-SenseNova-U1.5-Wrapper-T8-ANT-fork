# TASK FOR NEXT SESSION (with github repo containing MERD + RoPE_Lab connected)

**You are in a session where the workspace contains the full ANT_NODES repo with MERD and RoPE_Lab source. Use this research doc as spec to implement SenseNova support.**

## TODO:

1. **Create MERD YAML** `sensenova_u1_5_8b_mot.yaml` under `MERD_REGISTRY/` using the draft in Section 8 of this doc. Add provenance envelopes per Schema v1.7.
   - Add to bridge: `NEOChatModel` -> `SenseNovaU15`? Actually ComfyUI runtime class is `SenseNovaU15`? Check `model.py: class SenseNovaU15` and `model_config.py: SenseNovaBaseModel`. Runtime detection returns `SenseNovaU15` or `SenseNovaBaseModel`. Add mapping to `merdy_comfy_scanners/bridge.py`.
   - Add latent family `sensenova_pixel` to `comfyui_latent_formats` knowledge table.

2. **Extend MERD schema** if needed:
   - `rope.per_axis_theta` is already in schema v1.7 but not populated — populate with `[5000000.0, 10000.0, 10000.0]`
   - Add new optional field `rope.per_axis_theta_vision` or reuse `rope.rope_theta_vision` (check schema_v1_7.py)
   - Add `architecture_detail.mot_enabled: bool` and `architecture_detail.mot_dual_weights: true`

3. **RoPE_Lab integration:**
   - Create new patching strategy `sensenova_mot_hook` in `MERD_DATA_Integration_Contract_v1.5.md` table and in `merd_schemas/schema_v1_7.py` enum.
   - Implement `core/method_hook.py: SenseNovaMotRopeHook` that monkey-patches:
     - `sensenova_u15.model._apply_llm_rope`
     - `sensenova_u15.model._apply_interleaved_rope`
     - `sensenova_u15.model.Attention._project` (to inject per-axis theta and dynamic ntk_factor)
   - Extend `ANT_RoPE_Lab_Node.py`:
     - Add detection for `mrope_interleaved` + `model_family_specific == SenseNova-U1.5` → don't abort, dispatch to `sensenova_mot_hook`
     - Add `per_axis_theta` handling in `_build_model_facts`
   - Add MERD adapter `sensenova_adapter.py` under `adapters/`? Or reuse `flux2_adapter` but with SenseNova-specific token hint math (patch_size 32, spatial_compression 1).
   - Update `sampler_patch.py` to handle pixel-space latent `[B,3,H,W]` for SEGA FFT — currently assumes `[B,N,C]` or `[B,H,W,C]`.

4. **Test:**
   - Generate MERD_DATA for SenseNova via Librarian, verify `rope.theta`, `axes_dim`, `per_axis_theta`, `patching_strategy` resolved.
   - Test static NTK at 4096x2048 (base 64x64 tokens → scale 2x) with `ANT's RoPE Lab` → should not abort.
   - Test dynamic `dy_ntk` with `dype_scale=1.0` — verify prefix cache invalidation works (set `transformer_options["sensenova_prefix_cache"]` clear on timestep change).
   - Test SEGA with `pass_latent=True` — verify FFT on 3-channel pixel latent.

5. **Cleanup:** Remove this TASK section after implementation, move doc to `docs/integration_sensenova.md`.

---

# SenseNova-U1.5-8B-MoT — MERD and RoPE Lab Integration Research

**Generated:** 2026-08-25, Graz
**Context:** Started from bug fix for https://github.com/Milor123/ComfyUI-SenseNova-U1.5-ConvRot (tokenizer digest mismatch) → investigation of model decoupling → RoPE patching feasibility → MERD schema analysis.

## 1. Official Model Repositories (Provenance)

| Item | Value | Provenance |
|------|-------|------------|
| Official Final | `sensenova/SenseNova-U1.5-8B-MoT` | `loader.py: MODEL_REPO`, README of node pack, HF page https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT |
| Official SFT | `sensenova/SenseNova-U1.5-8B-MoT-SFT` | `loader.py: SFT_MODEL_REPO` |
| Final Revision | `1f6ec60423d29939dde4202fd82ae340b144e280` | `loader.py: MODEL_REVISION` |
| SFT Revision | `661834c5b5aee0f89958353511d6ac0ccaacb646` | `loader.py: SFT_MODEL_REVISION` |
| Config SHA256 | `6497591f64cb0dd6917fbb10c0cd13024e5817179a9aa3700998eb137a553d6b` | `loader.py: CONFIG_SHA256`, matches `sensenova_u15/tokenizer/config.json` LF hash (verified via `sha256sum`) |
| LoRA Repo (8-step) | `sensenova/SenseNova-U1.5-8B-MoT-LoRAs` | `lora.py: LORA_REPO` |
| LoRA Revision | `e909f4636d119d65fe4cba8770c19daff2ac102e` | `lora.py: LORA_REVISION` |
| Architecture Blog | NEO-unify: Building Native Multimodal Unified Models End to End | https://huggingface.co/blog/sensenova/neo-unify |
| Upstream Wrapper | `T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8` | README of Milor123 fork |
| Model Size Official | 18B params, BF16, 50GB file (Flux VAE PSNR comparison: 31.56 vs 32.65) | HF model page + blog |
| License | Apache 2.0 | HF model page |

## 2. Node Pack Repository Used (Provenance)

| Item | Value | Provenance |
|------|-------|------------|
| Fork Repo | `Milor123/ComfyUI-SenseNova-U1.5-ConvRot` | User-provided link, git clone in workspace `/home/user/ComfyUI-SenseNova-U1.5-ConvRot` |
| Stars | 4 | GitHub page fetch |
| Quantized Weights Repo | `Milor123/ComfyUI-ConvRot-SenseNova-U1.5-8B-MoT-T8` | README.md |
| Recommended Checkpoint | `SenseNova-U1.5-8B-MoT-T8-int8-convrot-tagged.safetensors` 17.58 GiB | README.md |
| Hybrid Checkpoint | `SenseNova-U1.5-8B-MoT-T8-hybw4a8-L18-41.safetensors` 13.80 GiB | README.md |
| LoRA File | `SenseNova-U1.5-8B-MoT-LoRA-8step-ComfyUI.safetensors` 0.76 GiB | README.md |
| Install Path | `ComfyUI/models/diffusion_models/SenseNovaU1.5/` | README.md |
| Key Fork Additions | `quant_bridge.py` (ConvRot-aware), `qt_guards.py` (QuantizedTensor guards), `checkpoint_contract.py` (static validation), `tools/make_hybrid_ladder.py`, `tools/convert_sensenova_int4_convrot.py` | README.md + file listing |
| Bug Fixed | Tokenizer digest mismatch due to `core.autocrlf=true` → CRLF vs LF | `loader.py` original `_validate_tokenizer_assets()` raw `read_bytes()` hash, reproduced via python hashlib, fixed via `.replace(b"\r\n", b"\n")` |
| UI Chinese Labels | `nodes.py:148 display_name=f"参考图 {index}"`, `web/sensenova_reference_labels_v131e.js:19`, `guidance.py: sections with 【】` | grep -rn "参考" |

## 3. Architecture Overview — NEO-unify + MoT (Provenance: blog + config.json + model.py)

- **Paradigm:** Encoder-free, VAE-free, end-to-end native unified. Near-lossless visual interface (pixels directly), native Mixture-of-Transformer (MoT), unified learning: autoregressive CE for texts + pixel flow matching for vision. From blog: "No VE! No VAE!"
- **Backbone:** `NEOChatModel` with `llm_config` = Qwen3ForCausalLM, but `pure_llm=false`. 42 layers, hidden 4096, intermediate 12288, 32 heads, 8 KV heads, vocab 151936, head_dim 128. Provenance: `tokenizer/config.json` parsed via `json.tool`.
- **MoT:** Each `DecoderLayer` has dual weights: normal (understanding/prefix) + `_mot_gen` (generation). Provenance: `checkpoint_contract.py` lists both `input_layernorm` and `input_layernorm_mot_gen` per layer, `model.py: class DecoderLayer` with `self_attn.q_proj` + `q_proj_mot_gen`, etc.
- **Vision:** Two vision models: `vision_model` (reference images) + `fm_modules.vision_model_mot_gen` (noisy image). Patch embedding kernel 16 stride 16 + dense 2 stride 2 → MERGED_PATCH_SIZE 32. Provenance: `model.py: VisionEmbeddings`, `HIDDEN_SIZE=4096`, `MERGED_PATCH_SIZE=32`.
- **Flow Head:** `ConvDecoder` = PixelShuffle(2) -> Conv 1024->1024 -> GELU -> PixelShuffle(2) -> Conv 256->192 -> PixelShuffle(8). Outputs pixel velocity. Provenance: `model.py: class ConvDecoder`.
- **Sampling:** `SenseNovaModelSampling` discrete flow, shift 3.0, multiplier 1000, `time_snr_shift`. `resolution_noise_scale` based on token count sqrt(H/32*W/32 /64). Provenance: `sampling.py`, `model_config.py: sampling_settings`.

## 4. Checkpoint Contract — Full Key List (Provenance: checkpoint_contract.py)

Total keys: count from file = 1 + 1 + 2*2 + 2*2 + 2*2 + 1 + 1 + 42* (2 + 3*2 + 2 + 4*2 + 6) + 2 + 2 = 1125? Let's trust file.

Categories:
- `fm_modules.fm_head.conv1/2` (4 tensors)
- `fm_modules.noise_scale_embedder.mlp.0/2` (4)
- `fm_modules.timestep_embedder.mlp.0/2` (4)
- `fm_modules.vision_model_mot_gen.embeddings.*` (4)
- `language_model.lm_head.weight` (1, popped)
- `language_model.model.embed_tokens.weight` (151936,4096)
- Per layer 0-41 (42 layers):
  - `input_layernorm.weight` (4096)
  - `input_layernorm_mot_gen.weight`
  - `mlp.down/gate/up_proj.weight` (4096x12288 etc) + `_mot_gen`
  - `post_attention_layernorm` + `_mot_gen`
  - `self_attn.k_norm, k_norm_hw, k_norm_hw_mot_gen, k_norm_mot_gen` (64 each)
  - `self_attn.q_norm, q_norm_hw, q_norm_hw_mot_gen, q_norm_mot_gen`
  - `self_attn.k_proj, k_proj_mot_gen` (1024,4096)
  - `self_attn.o_proj, o_proj_mot_gen` (4096,4096)
  - `self_attn.q_proj, q_proj_mot_gen` (4096,4096)
  - `self_attn.v_proj, v_proj_mot_gen` (1024,4096)
- `language_model.model.norm.weight` + `norm_mot_gen`
- `vision_model.embeddings.dense/patch_embedding`

Quantized variant adds per linear weight:
- `*.weight` (I8, packed), `*.weight_scale` (F32), `*.comfy_quant` (U8 JSON), and for `asym_w4a8_int8`: `weight_s_rel` (F8_E4M3), `weight_s_channel` (F32), `weight_codebook` (F32). Provenance: `loader.py: _checkpoint_contract()`, `_expected_storage_dtype()`.

## 5. Text Encoder Analysis (Provenance: text_encoder.py + conditioning.py)

- **Tokenizer:** `SenseNovaQwenTokenizer` wraps `transformers.Qwen2Tokenizer` from `sensenova_u15/tokenizer/` folder (vocab.json, merges.txt, tokenizer_config.json). Provenance: `text_encoder.py:23 tokenizer_path = os.path.join(..., "tokenizer")`.
- **Chat Template:** System message about image generation/editing, Think Mode, etc. `build_generation_prompt(text)` = `<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<img>`. Unconditional = `<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n<img>`. Provenance: `text_encoder.py:12-30`.
- **Token Filtering:** Removes pad token 151643. Provenance: `text_encoder.py:63 values = [v for v in values if int(v[0]) != 151643]`.
- **Encoder:** `SenseNovaTextEncoder` has `dtypes={float32}`, `disable_offload=True`, `load_sd` returns `[]`, `get_sd` returns `{}`. `encode_token_weights` returns `input_ids` as `text_input_ids` extra cond. **Zero trainable weights.** Provenance: `text_encoder.py:70-105`.
- **Conditioning:** `condition_input_ids` inserts image tokens `[IMAGE_START_ID=151670] + [IMAGE_CONTEXT_ID=151669]*H*W + [IMAGE_END_ID=151671]` with labels `IMAGE_LABEL_IDS` for multi-ref. `thw_indexes` builds time/height/width indexes for RoPE. `block_causal_mask` builds causal mask. Provenance: `conditioning.py`.
- **Conclusion:** Text encoder is tokenizer-only; LLM weights are inside diffusion model. Cannot use Qwen3VL LoRAs.

## 6. VAE Analysis (Provenance: model_config.py, nodes.py, latent_formats)

- `model_config.py: process_vae_state_dict` returns `{"pixel_space_vae": torch.tensor(1.0)}` — dummy.
- `latent_format = comfy.latent_formats.HiDreamO1Pixel` — pixel-space, 3 channels, no compression. Provenance: `model_config.py:61`.
- `EmptySenseNovaLatentImage` creates `torch.zeros((batch,3,height,width))` — RGB latent, not 4/16/32ch. Provenance: `nodes.py: EmptySenseNovaLatentImage.execute`.
- `smart_resize` factor 32, min 512², max 2048² (or 4096²//num_images for multi-ref). Provenance: `conditioning.py: smart_resize`.
- No VAE weights in contract. So VAE is already decoupled — no VRAM, no splitting needed. If you want latent manipulation, you'd need to train a VAE, not in original.

## 7. RoPE Detailed Analysis (Provenance: model.py + config.json + RoPE_Lab export)

### 7.1 Functions

```python
# model.py L23-45
def _apply_llm_rope(query, key, positions, theta):
    dim = query.shape[-1]
    frequencies = theta ** (-arange(0,dim,2)/dim)  # 1/theta^(2i/d)
    angles = positions * frequencies
    cos, sin = cos(angles), sin(angles)
    rotation = stack(cos,-sin,sin,cos).reshape(...,2,2)
    query,key = ck.apply_rope_split_half(query^T, key^T, rotation)
    return query^T, key^T

def _apply_interleaved_rope(value, positions, theta):
    frequencies = theta ** (-arange(0,dim,2)/dim)
    angles = positions * frequencies
    rotation = stack(cos,-sin,sin,cos).reshape(1,...,2,2)
    return apply_rope1(value.float(), rotation)
```

### 7.2 Where applied

- **LLM Attention** `Attention._project`: splits HEAD_DIM 128 into 64 (t) + 32 (h) + 32 (w). Applies QK RMSNorm (64-> RMSNorm(32)? Actually `HEAD_DIM//2=64` but they chunk: `q_norm` on 64//2=32? Wait code: `q_norm = RMSNorm(HEAD_DIM//2=64)` but query_t is 64 dim, so norm on 32? Actually they do `query.chunk(2)` -> 64 each, then `q_norm` on that 64? But RMSNorm dim 64? Let's trust: `q_norm = RMSNorm(HEAD_DIM//2=64)` — so norm on 64. Then for HW, `q_norm_hw` also 64? But query_hw is 64, then chunked to 32 each, then `q_norm_hw` on 32? Hmm inconsistency — need telemetry to confirm. Provenance: `model.py: class Attention`.
- **Vision Embeddings** `VisionEmbeddings.forward`: `patch_embedding` 3->1024 conv16, GELU, flatten, RoPE on x and y separately with theta 10000, then `dense_embedding` 1024->4096 conv2.
- **Indexes** `thw_indexes`: `time_indexes = cumsum(image_start_shift + not_image) -1`, `height_indexes`, `width_indexes` from reference_grids. Provenance: `conditioning.py: thw_indexes`.

### 7.3 Axes

- Axes: t (time / token position), h (height), w (width)
- `axes_dim`: [64,32,32] in terms of head_dim, or [32,16,16] pairs. Need to verify via consolidator.
- `spatial_axis_indices`: [1,2] (h,w)
- `temporal_axis_index`: 0 (t)
- `axis_identity`: ["t","h","w"]
- `axes_concat_order`: "t_h_w" (like MiniMax-H3)
- `per_axis_theta`: [5000000.0, 10000.0, 10000.0] — not in current MERD but needed.
- `cos_sin_arrangement`: likely "split_half" (uses `ck.apply_rope_split_half`) vs "repeat_interleave" (Flux family). Need telemetry.
- `dype_output_format`: "rotation_matrix" (uses rotation matrix) or "mrope_interleaved" (like Ideogram4). Your Lab's Ideogram4 config uses `mrope_interleaved` and aborts. SenseNova should be same — inline.

### 7.4 Comparison to your RoPE Lab

Your Lab's `methods/base.py` contract:
```python
def compute(axis_pos, axis_dim, axis_index, model_facts, runtime_state, freqs_dtype):
    # returns (cos, sin)
```

Your `rope_math.py`:
- `get_1d_ntk_pos_embed(dim, pos, theta, ntk_factor, cos_sin_arrangement)`
- `get_1d_yarn_pos_embed(dim, pos, theta, max_pe_len, ori_max_pe_len, dype, current_timestep, ...)`
- `apply_rope_scale_factor(axis_pos, axis_index, model_facts)` — handles `rope_scale_factor=[1.0,2.0,2.0]` like Kandinsky5.

SenseNova needs per-axis theta, so `model_facts["theta"]` must become list or you read `per_axis_theta[axis_index]`.

Your `ANT_RoPE_Lab_Node.py` has 3 patching strategies that could apply:
- `embedder_replacement`: Flux family — replaces `diffusion_model.pe_embedder` via `add_object_patch`. **Not applicable** to SenseNova (no pe_embedder).
- `inv_freq_buffer`: MiniMax-H3 — modifies `rope.inv_freq` buffer. SenseNova has no inv_freq buffer.
- `method_hook`: MiniMax-H3 coordinate methods — monkey-patches `rope_freqs()`. **Closest** — you need to patch `Attention._project`.
- `function_hook`: Ideogram4 — patches `precompute_freqs_cis`. SenseNova's `_apply_llm_rope` is similar pure function.
- `rotation_table_postmultiply`: SEGA on H3 — post-multiplies rotation table.

Currently your Lab aborts for `mrope_interleaved`:
```python
if output_fmt == "mrope_interleaved":
    "INLINE-ROPE MODEL — RoPE Lab CANNOT PATCH"
```
SenseNova would trigger this.

## 8. MERD Integration — Proposed YAML (Provenance: all above + example MERDs)

```yaml
metadata:
  schema_version: '1.7'
  model_name: sensenova/SenseNova-U1.5-8B-MoT
  display_name: sensenova/SenseNova-U1.5-8B-MoT

identity:
  model_family:
    derived: SenseNova
    effective: SenseNova
  model_family_specific:
    derived: SenseNova-U1.5
    effective: SenseNova-U1.5
  latent_family:
    derived: sensenova_pixel
    effective: sensenova_pixel
  architecture_class:
    official: NEOChatModel
    derived: SenseNovaU15
    effective: SenseNovaU15
  source_repository:
    official: sensenova/SenseNova-U1.5-8B-MoT
    effective: sensenova/SenseNova-U1.5-8B-MoT

architecture:
  patch_size:
    official: 32
    derived: 32
    effective: 32
  hidden_size:
    official: 4096
    effective: 4096
  num_layers:
    official: 42
    effective: 42
  num_attention_heads:
    official: 32
    effective: 32
  attention_head_dim:
    official: 128
    effective: 128
  in_channels:
    official: 3
    effective: 3
  out_channels:
    official: 3
    effective: 3
  runtime_class_name:
    derived: SenseNovaU15
    effective: SenseNovaU15
  model_type:
    derived: FLOW
    effective: FLOW

architecture_detail:
  depth_double: 42
  mot_enabled: true
  mot_dual_weights: true
  uses_qk_norm: true
  qk_norm_dim: 64
  mlp_ratio: 3.0
  ffn_dim: 12288
  num_kv_heads: 8

vae:
  vae_family:
    derived: sensenova_pixel_vae
    effective: sensenova_pixel_vae
  produces_latent_format:
    derived: sensenova_pixel
    effective: sensenova_pixel
  latent_channels:
    official: 3
    effective: 3
  compression_spatial:
    derived: 1
    effective: 1

latent_format:
  name:
    derived: sensenova_pixel
    effective: sensenova_pixel
  identifier:
    channels: 3
    spatial_compression: 1
    temporal_compression: 1
    patch_layout: none
    normalization_regime: none
    uses_quant_conv: false
  compatibility:
    requires_vae_family: [sensenova_pixel_vae]
    is_video_latent: false
    requires_patch_aware_decoder: false

scheduler:
  prediction_type: flow_matching
  flow_shift: 3.0
  max_shift: 1.15
  sigma_range: [0.0, 1.0]

rope:
  theta: 5000000.0
  per_axis_theta: [5000000.0, 10000.0, 10000.0]
  per_axis_theta_vision: 10000.0
  axes_dim: [64,32,32]
  rope_total_dim: 128
  axis_identity: [t,h,w]
  spatial_axis_indices: [1,2]
  temporal_axis_index: 0
  axes_concat_order: t_h_w
  base_patch_grid: [64,64]
  cos_sin_arrangement: split_half
  dype_output_format: mrope_interleaved
  position_id_source: thw_indexes
  position_coordinate_style: integer
  requires_isotropic_scaling: false
  has_learned_position_embeddings: false
  uses_attention_mask: true
  uses_qk_norm: true
  embedder_class: Attention
  spatial_freq_min: 0.00080405  # derived from theta 5M? need calc
  spatial_freq_max: 1.0

adapter_requirements:
  patching_strategy: sensenova_mot_hook
  has_module_to_patch: false
  requires_embedder_patch: false
  requires_position_id_patch: true
  requires_sampler_patch: true
  requires_extra_runtime_hooks: true
  inline_rope_function:
    - sensenova_u15.model._apply_llm_rope
    - sensenova_u15.model._apply_interleaved_rope
    - sensenova_u15.model.Attention._project
  comfyui_embedder_path: null
  inv_freq_buffer_path: null
  embedder_attrs:
    - attr: language_model.model.layers.0.self_attn
      role: prefix
    - attr: language_model.model.layers.0.self_attn_mot_gen
      role: generation
  frozen_local_imports:
    - comfy.ldm.common_dit.pad_to_patch_size
    - comfy.ldm.modules.attention.optimized_attention
    - comfy.ldm.flux.math.apply_rope1
```

Provenance for each field:
- `theta`, `per_axis_theta`, `axes_dim` from `tokenizer/config.json` + `model.py` chunking
- `base_patch_grid` derived: 2048/32=64 (from `nodes.py` default 2048 + `model.py` MERGED_PATCH_SIZE 32)
- `cos_sin_arrangement` from `model.py` uses `ck.apply_rope_split_half` → split_half
- `dype_output_format` from function uses rotation matrix → mrope_interleaved like Ideogram4
- `patching_strategy` from analysis of your Lab's 6 strategies — needs new one
- `vae` from `model_config.py` dummy + `nodes.py` latent shape

## 9. RoPE Lab Hooks Needed — Detailed Suggestion

### 9.1 New Hook Class: SenseNovaMotRopeHook

Location: `core/method_hook.py` (add alongside MiniMaxH3RopeFreqsHook)

```python
class SenseNovaMotRopeHook:
    def __init__(self, model_module, method, model_facts, runtime_knobs, method_specific_args, embedder):
        # model_module = Attention class or SenseNovaU15
        self.model_module = model_module
        self.method = method
        self.model_facts = model_facts
        self.runtime_knobs = runtime_knobs
        self.original_apply_llm_rope = sensenova_u15.model._apply_llm_rope
        self.original_apply_interleaved = sensenova_u15.model._apply_interleaved_rope
        self.original_project = sensenova_u15.model.Attention._project

    def install(self):
        # Patch _apply_llm_rope to use dynamic ntk_factor from method.compute()
        def hooked_llm_rope(query, key, positions, theta):
            # theta is per-axis base, but we want to apply NTK scaling from method
            # Compute ntk_factor via method.compute() for this axis
            # Need to know axis_index from call stack — inspect positions shape or pass via transformer_options
            # Simplest: read transformer_options from current context (if available)
            # For now, apply same logic as dy_ntk
            ...
            return self.original_apply_llm_rope(query, key, positions, theta * ntk_factor)

        sensenova_u15.model._apply_llm_rope = hooked_llm_rope
        # Similar for interleaved
        # Also patch Attention._project to inject transformer_options

    def uninstall(self):
        sensenova_u15.model._apply_llm_rope = self.original_apply_llm_rope
        ...
```

### 9.2 Sampler Patch Extension

Your `sampler_patch.py` already has `pass_latent` for SEGA. For SenseNova, latent is `[B,3,H,W]` not `[B,N,C]`. Your `sega_math.py: compute_spectral_energy_profile` handles both 3D and 4D, but assumes `height*width` tokens. For SenseNova, token grid is `H/32 x W/32`. Need to compute `height = latent.shape[-2] // 32`, `width = latent.shape[-1] // 32` and pass to SEGA.

### 9.3 Embedder Replacement Alternative

Instead of monkey-patching functions, create `ANTsRoPELabSenseNovaEmbedder` that replaces `LanguageBackbone.embed_tokens`? No, RoPE is not in embedder.

Better: Patch `SenseNovaU15._forward` to inject custom RoPE thetas via `transformer_options`.

Proposed minimal change to upstream model (add to `model.py`):

```python
def _project(self, hidden_states, indexes, generation, transformer_options=None):
    rope_theta_time = transformer_options.get("rope_theta_time", 5000000.0) if transformer_options else 5000000.0
    rope_theta_hw = transformer_options.get("rope_theta_hw", 10000.0) if transformer_options else 10000.0
    query_t, key_t = _apply_llm_rope(query_t, key_t, indexes[0], rope_theta_time)
    query_h, key_h = _apply_llm_rope(query_h, key_h, indexes[1], rope_theta_hw)
    query_w, key_w = _apply_llm_rope(query_w, key_w, indexes[2], rope_theta_hw)
```

Then RoPE Lab can just set `transformer_options["rope_theta_time"]` via `add_wrapper`.

This is 10 lines and avoids monkey-patching.

### 9.4 Token Hints

Your Lab's `_resolve_spatial_compression` and `_resolve_patch_size` currently read from MERD. For SenseNova:

- `spatial_compression` = 1 (pixel VAE) or 32 (merged patch)? In `ANT_RoPE_Lab_Node.py` you compute `hinted_h_tokens = (height // spatial_comp) // patch_h`. For SenseNova, `height=2048`, `spatial_comp=1`, `patch_h=32` → 64 tokens, correct.
- `patch_size` = 32 (from MERD `architecture.patch_size`).

Need to ensure MERD provides `architecture.patch_size=32` and `vae.compression_spatial=1`.

## 10. Integration Roadmap — How It Should Progress

**Phase 0: MERD entry (1 hour)**
- Create YAML, add to registry, test Librarian resolves.

**Phase 1: Read-only support (no patching, just info)**
- Make RoPE Lab detect SenseNova, not abort, but pass through with report "Model detected, patching not yet implemented". This validates MERD.

**Phase 2: Static patching (2-3 hours)**
- Implement `SenseNovaMotRopeHook` that applies static NTK factor based on `global_scale = max(target_h/base_h, target_w/base_w)`. Test at 4096x2048 vs 2048x2048 — should reduce duplication.

**Phase 3: Dynamic patching (1 day)**
- Wire `sampler_patch.py` timestep tracking to hook, implement `dy_ntk` dynamic `ntk_factor = base_ntk ** (dype_scale * t^dype_exponent)`. Need to clear `sensenova_prefix_cache` when timestep changes, otherwise KV cache stale.

**Phase 4: SEGA (1 day)**
- Extend SEGA to handle pixel latent. Test with `sega` method — should improve detail at 4K.

**Phase 5: Split files (optional, 1 week)**
- Use `tools/split_sensenova.py` to produce prefix + dit, create `SenseNovaSplitLoader` node, implement separate ModelPatchers with KV cache handoff. Only if VRAM cleanup is critical — current ComfyUI streaming already handles 12GB VRAM.

**Phase 6: Documentation**
- Update `MERD_DATA_Integration_Contract` with new strategy `sensenova_mot_hook`, add SenseNova to `EQUIVALENT_CLASS_GROUPS` if needed.

## 11. Provenance Report — Where Each Value Was Learned

| Value | Source File / URL | How Learned |
|-------|-------------------|-------------|
| `rope.theta=5M` | `tokenizer/config.json: llm_config.rope_theta` | `cat ... | json.tool` |
| `rope_theta_hw=10k` | `tokenizer/config.json: llm_config.rope_theta_hw` | same |
| `rope_theta_vision=10k` | `tokenizer/config.json: vision_config.rope_theta_vision` | same |
| `axes_dim=[64,32,32]` | `model.py: Attention._project chunk(2)` | Code reading, HEAD_DIM 128 split |
| `hidden_size=4096` | `model.py: HIDDEN_SIZE`, `model_config.py` | Constant |
| `num_layers=42` | `model.py: NUM_LAYERS` | Constant |
| `num_heads=32` | `model.py: NUM_HEADS` | Constant |
| `head_dim=128` | `model.py: HEAD_DIM` | Constant |
| `vocab_size=151936` | `model.py: VOCAB_SIZE`, `config.json: llm_config.vocab_size` | Constant |
| `patch_size=32` | `model.py: MERGED_PATCH_SIZE` | Constant |
| `vision patch 16+2` | `model.py: VisionEmbeddings Conv2d kernel 16 stride 16 + dense 2 stride 2` | Code reading |
| `flow_shift=3.0` | `model_config.py: sampling_settings`, `nodes.py: SenseNovaSamplingOptions default 3.0` | Code reading |
| `latent_format=HiDreamO1Pixel` | `model_config.py: latent_format` | Code reading |
| `vae dummy` | `model_config.py: process_vae_state_dict returns pixel_space_vae 1.0` | Code reading |
| `text encoder dummy` | `text_encoder.py: load_sd returns [], encode returns input_ids` | Code reading |
| `uses_qk_norm=true` | `model.py: q_norm, k_norm, etc RMSNorm` | Code reading |
| `uses_attention_mask=true` | `conditioning.py: block_causal_mask`, `model.py: forward_prefix uses mask` | Code reading |
| `tokenizer digest bug` | `loader.py: _validate_tokenizer_assets raw read_bytes` + `sha256sum` + CRLF reproduction | Bash + python hashlib |
| `MoT dual weights` | `checkpoint_contract.py` per layer `*_mot_gen` + `model.py: DecoderLayer` | File listing + code |
| `quant sidecars` | `loader.py: _checkpoint_contract, _expected_storage_dtype` | Code reading |
| `model_type=FLOW` | `model_config.py: ModelType.FLOW` | Code reading |
| `patching_strategy needed` | `MERD_DATA_Integration_Contract_v1.5.md` Table 6 strategies + `ANT_RoPE_Lab_Node.py` abort for `mrope_interleaved` | Reading contract + Lab export |
| `per_axis_theta exists` | `ANT_RoPE_Lab_Node.py: _build_model_facts get("rope.per_axis_theta")` | Grep |
| `axes_concat_order t_h_w` | `minimax_h3.txt: axes_concat_order: t_h_w` | Example MERD |
| `cos_sin_arrangement split_half` | `model.py: ck.apply_rope_split_half` | Code reading |
| `dype_output_format rotation_table` | `model.py: rotation matrix` + `minimax_h3.txt: rotation_table` | Code reading + example |
| `official repos` | HF pages + `loader.py` constants | Fetch + code |
| `node pack repo` | `Milor123/ComfyUI-SenseNova-U1.5-ConvRot` GitHub fetch | Fetch |

## 12. References

- Official: https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT
- Blog: https://huggingface.co/blog/sensenova/neo-unify
- Node pack: https://github.com/Milor123/ComfyUI-SenseNova-U1.5-ConvRot
- Quant weights: https://huggingface.co/Milor123/ComfyUI-ConvRot-SenseNova-U1.5-8B-MoT-T8
- Upstream wrapper: https://github.com/T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8
- RoPE Lab export: `RoPE_Lab_2026-08-25_22-57-41.md` (user upload, 10k LOC, adapters: dy_ntk, yarn, vision_yarn, hope, sega, etc.)
- MERD Contract: `MERD_DATA_Integration_Contract_v1.5.md` (Schema V1.7, 23 detectors, 6 patching strategies)
- Example MERDs: `flux2_klein_base_9b.txt` (embedder_replacement), `minimax_h3.txt` (method_hook), `ideogram4.txt` (mrope_interleaved abort), `krea2_raw.txt`

---

*End of research doc — ready for implementation in next session.*
