# RoPE Lab / MERD integration — SenseNova U1.5 (fork notes)

> **Migration note (2026-09-03):** RoPE policy should move out of this wrapper
> into RoPE Lab as a loader-agnostic MODEL patch node. The adapter design,
> scoped hook strategy, cache requirements and custom/core test matrix are in
> [`ROPE_LAB_MODELPATCH_ARCHITECTURE.md`](ROPE_LAB_MODELPATCH_ARCHITECTURE.md).
> The direct `transformer_options` keys below remain a transitional bridge.

Distilled from `docs/SenseNova_MERD_and_RoPE_Lab_integration_research.md` (full
provenance and the complete MERD YAML live there) and narrowed to what this
node pack has to keep stable for `ANT_NODES/RoPE_Lab` + MERD v1.7.

Status of the node-pack side: **ready**. The two blockers the research doc
listed are removed in this fork:

1. the RoPE maths no longer depends on a comfy-kitchen kernel (upstream 1.3.4
   fix, preserved here), so a patcher can wrap the reference formulas;
2. the three per-axis bases are read from `transformer_options`
   (`sensenova_u15.model.resolve_rope_thetas`), so dynamic methods do not need
   to monkey-patch `model.py` at all.

Everything else (the hook class, the MERD entry, the sampler patch) lives in
the Lab repository.

---

## 1. What SenseNova's RoPE actually is

| Property | Value | Source |
| --- | --- | --- |
| Position layout | 3D M-RoPE, axes `t, h, w`, concat order `t_h_w` | `sensenova_u15/conditioning.py::thw_indexes` |
| Head dim / axis split | 128 total → 64 (t) + 32 (h) + 32 (w); pairs 32/16/16 | `sensenova_u15/model.py::Attention._project` |
| Bases | `t = 5 000 000.0`, `h = w = 10 000.0` | `ROPE_THETA_TIME`, `ROPE_THETA_SPATIAL` |
| Cos/sin arrangement (LLM) | split-half, plain PyTorch `rotate_half` | `_apply_llm_rope` |
| Vision RoPE | interleaved, `apply_rope1` with 4D value + 6D rotation, base `10 000.0` | `_apply_interleaved_rope`, `ROPE_THETA_VISION` |
| QK norm | per-axis RMSNorm on the 64-dim halves (`q_norm`, `q_norm_hw`, …) | `Attention.__init__` |
| Attention mask | block-causal prefix mask + GQA (8 KV heads) | `conditioning.py::block_causal_mask` |
| Token grid | `H/32 × W/32` (pixel latent, `MERGED_PATCH_SIZE = 32`), 64×64 at 2048 | `nodes.py`, `model.py` |
| `dype_output_format` | `mrope_interleaved` → RoPE Lab aborts unless a new strategy is used | research doc §7.3/§7.4 |
| Dual weights | MoT: every attention/MLP/norm has a `*_mot_gen` twin used for generation | `model.py` |

Two structural facts matter for any patcher:

- the **prefix** (text + reference images) is computed **once per sampler step**
  and cached per layer (`transformer_options["sensenova_prefix_cache"]`), then
  reused by the generation branch;
- the cache key includes the active RoPE bases, so changing a theta invalidates
  it automatically. Never "optimise" that key away — with a dynamic (per
  timestep) schedule the cached KV would silently go stale.

## 2. Hook surface the fork exposes

```python
# sensenova_u15/model.py
ROPE_THETA_TIME = 5000000.0
ROPE_THETA_SPATIAL = 10000.0
ROPE_THETA_VISION = 10000.0
ROPE_THETA_OPTIONS = ("sensenova_rope_theta_t", "sensenova_rope_theta_hw", "sensenova_rope_theta_vision")

def resolve_rope_thetas(transformer_options=None) -> tuple[float, float, float]: ...
```

Every value is validated (must be a positive finite number), so a broken scale
fails loudly at the first step instead of painting a corrupted image.

A Lab wrapper therefore only needs an `OUTER_SAMPLE`/`DIFFUSION_MODEL` wrapper
that sets the three options:

```python
def rope_lab_options(executor, *args, **kwargs):
    guider = executor.class_obj
    opts = guider.model_options.setdefault("transformer_options", {})
    opts["sensenova_rope_theta_t"] = 5_000_000.0 * ntk_factor_t
    opts["sensenova_rope_theta_hw"] = 10_000.0 * ntk_factor_hw
    opts["sensenova_rope_theta_vision"] = 10_000.0 * ntk_factor_hw
    return executor(*args, **kwargs)
```

Remaining hook needs (in the Lab repo):

- `SenseNovaMotRopeHook` in `core/method_hook.py` for methods that need more
  than a base rescale (per-axis YaRN/Hope/SEGa mixing, rotation post-multiply),
  patching `_apply_llm_rope`, `_apply_interleaved_rope` and
  `Attention._project`;
- sampler patch: pixel latent is `[B, 3, H, W]`, so SEGA's FFT needs
  `pass_latent=True` with the token grid derived as `H/32 × W/32`;
- frozen-import handling: `optimized_attention`, `pad_to_patch_size` and
  `apply_rope1` are imported into `sensenova_u15.model`, so patch the module
  attributes (or walk `sys.modules`), not just the origin modules;
- a `sensenova_mot_hook` patching strategy — `embedder_replacement` (Flux) and
  `inv_freq_buffer` (MiniMax-H3) do not apply, there is no `pe_embedder` and no
  `inv_freq` buffer.

## 3. Dype math (per axis)

```
base_ntk = scale ** (dim / (dim - 2))          # dim = axis dim (64 for t, 32 for h/w)
kappa    = dype_scale * t ** dype_exponent     # t = current denoising progress
ntk_factor = base_ntk ** kappa                 # multiply the axis base by it
```

`scale = target_grid / base_patch_grid` per axis, with `base_patch_grid = [64, 64]`
at 2048 px. Because `resolve_rope_thetas` takes the three axes separately, the
temporal axis can stay unscaled while only `h`/`w` are stretched — that is the
correct behaviour for pure resolution scaling and the reason the fork did not
collapse the bases into one value.

## 4. MERD fields still to add (database v1.7)

New/renamed fields the SenseNova entry needs — full YAML in the research doc
§8, including `architecture.patch_size: 32`, `vae.compression_spatial: 1`,
`mot_enabled`, `mot_dual_weights`, `uses_qk_norm`, `uses_attention_mask`:

```yaml
rope:
  per_axis_theta: [5000000.0, 10000.0, 10000.0]
  per_axis_theta_vision: 10000.0
  axes_dim: [64, 32, 32]
  cos_sin_arrangement: split_half
  dype_output_format: mrope_interleaved
  base_patch_grid: [64, 64]
  position_id_source: thw_indexes
  uses_attention_mask: true
adapter_requirements:
  patching_strategy: sensenova_mot_hook
  inline_rope_function:
    - sensenova_u15.model._apply_llm_rope
    - sensenova_u15.model._apply_interleaved_rope
    - sensenova_u15.model.Attention._project
  frozen_local_imports:
    - comfy.ldm.common_dit.pad_to_patch_size
    - comfy.ldm.modules.attention.optimized_attention
    - comfy.ldm.flux.math.apply_rope1
latent_format: sensenova_pixel   # channels 3, compression 1, no patching, no normalisation
```

New latent family `sensenova_pixel` and the new patching strategy both have to
be registered in the Lab before SenseNova stops aborting.

## 5. Roadmap

| Phase | Work | Where |
| --- | --- | --- |
| 0 | MERD YAML + registry entry, Librarian resolves SenseNova | MERD |
| 1 | Detect, report "patching not implemented", no abort | RoPE_Lab |
| 2 | Static per-axis NTK via the three transformer options (2048 vs 4096 duplication test) | RoPE_Lab |
| 3 | Dynamic `dy_ntk` (`ntk_factor = base_ntk ** (dype_scale * t**exp)`), cache-safe | RoPE_Lab |
| 4 | SEGA on the pixel latent (`pass_latent=True`, `H/32 × W/32` grid) | RoPE_Lab |
| 5 | Optional prefix/DiT split loading | this repo (`MODEL_DECOUPLING_ANALYSIS.md`) |
| 6 | Document `sensenova_mot_hook` in `MERD_DATA_Integration_Contract`, add to `EQUIVALENT_CLASS_GROUPS` | MERD |

Quantized checkpoints interact with all of this only through
`Attention._project`: the bridge replaces `Linear` forwards, the RoPE path is
untouched, so a method that works on bf16 works on int8/W4A4/W4A8 too.

## 6. Interaction with the ConvRot port

- Quantization never changes positions, thetas or the mask, so all RoPE research
  findings transfer unchanged to the int8 / W4A8 rungs.
- When a Lab method needs *exact* reference numerics, pin the ops:
  `SENSENOVA_FORCE_BRIDGE=1` reproduces the original ConvRot fork's eager path;
  the default (auto) may use ComfyUI's own convrot kernels, which differ in
  rounding but not in basis.
- RoPE experiments at 4096 px on int8 are the cheapest way to test the
  duplication-reduction claims (17.58 GB int8 vs 35 GB all-BF16 final).
