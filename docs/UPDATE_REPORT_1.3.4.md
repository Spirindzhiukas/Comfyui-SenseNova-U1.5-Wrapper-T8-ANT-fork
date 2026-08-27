# Update Report — Upstream 1.3.4 (Commit 7365700) Port to ConvRot Fork

## What upstream commit does

**Commit:** https://github.com/T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8/commit/73657001179ccb28479e29a09e69aa7f67e4277d
**Title:** Fix CUDA and Triton RoPE compatibility
**Version:** 1.3.3 → 1.3.4
**Date:** 2026-08-25 (same day we fixed tokenizer bug)

CHANGELOG (translated from Chinese):
> - Fixed CUDA 13 / Blackwell environment when `comfy-kitchen` CUDA backend is enabled, split-half RoPE may return finite but incorrect values, causing severe color bias, over-saturation and structure anomalies; language layer RoPE now fixed to use official PyTorch reference formula.
> - Vision RoPE changed to use comfy-kitchen supported standard 4D input and 6D rotation layout, `--enable-triton-backend` no longer fails due to 3D tensor unpack failure.
> - Added backend isolation and accelerated-backend tensor rank regression tests.

### Technical diff in `sensenova_u15/model.py`

**Before (buggy):**
```python
import comfy.quant_ops
...
def _apply_llm_rope(query, key, positions, theta):
    dim = query.shape[-1]
    frequencies = theta ** (-arange(0,dim,2)/dim)
    ...
    rotation = stack(cos,-sin,sin,cos).reshape(*angles.shape,2,2).unsqueeze(2)
    query, key = comfy.quant_ops.ck.apply_rope_split_half(query^T, key^T, rotation)
    return ...

def _apply_interleaved_rope(value, positions, theta):
    ...
    rotation = stack(cos,-sin,sin,cos).reshape(1,*angles.shape,2,2)
    return apply_rope1(value.float(), rotation)
```

**After (fixed):**
```python
# No import comfy.quant_ops

def _apply_llm_rope(query, key, positions, theta):
    dim = query.shape[-1]
    frequencies = theta ** (-arange(0,dim,2)/dim)
    ...
    embedding = cat(angles, angles, dim=-1).unsqueeze(1)
    cosine = embedding.cos().to(query.dtype)
    sine = embedding.sin().to(query.dtype)
    def rotate_half(v): first,second = v.chunk(2,dim=-1); return cat(-second,first,dim=-1)
    # Keep split-half RoPE on reference PyTorch formula. CUDA kernel on CUDA 13 / Blackwell can return finite but numerically incorrect.
    return query*cosine + rotate_half(query)*sine, key*cosine + rotate_half(key)*sine

def _apply_interleaved_rope(value, positions, theta):
    ...
    rotation = stack(cos,-sin,sin,cos).reshape(1,1,*angles.shape,2,2) # 6D layout
    return apply_rope1(value.float().unsqueeze(1), rotation).squeeze(1) # 4D input
```

**Why critical:**
- `comfy.quant_ops.ck.apply_rope_split_half` dispatches to CUDA kernel via comfy-kitchen when CUDA 13 is detected. On Blackwell (RTX 50xx, e.g., 5070/5080/5090, or H100 with CUDA 13), that kernel returns **finite but wrong** values — no error, just corrupted image (severe tint, oversaturation, broken structure).
- `apply_rope1` Triton backend expects 4D input `[B,1,T,D]` and 6D rotation `[1,1,T,D//2,2,2]`. Old code passed 3D `[B,T,D]` and 5D rotation `[1,T,D//2,2,2]` — worked on eager but crashed with `--enable-triton-backend`.

Tests added:
- `test_shared_split_half_rope_matches_upstream_math` now mocks `ck.apply_rope_split_half` to assert it is **not called** — must stay on PyTorch.
- `test_interleaved_rope_uses_accelerated_backend_supported_ranks` asserts input shape `(2,1,7,32)` and rotation `(1,1,7,16,2,2)`.

## Was our ConvRot fork updated?

**No.**

- Fork `Milor123/ComfyUI-SenseNova-U1.5-ConvRot` latest commit at time of our fix: `7e1e320 Recommend INT8 variant...` (before 1.3.4)
- Upstream 1.3.4 commit `7365700` is on `T8mars/main` but not merged into Milor123 fork.
- Our `FIXED` folder from previous session still had old buggy `model.py` with `comfy.quant_ops.ck.apply_rope_split_half`.

Verified via:
```bash
grep -n "apply_rope_split_half" sensenova_u15/model.py
# Original & FIXED both had it before this update
```

## Did we port it now?

**Yes — updated in this session.**

- Copied upstream `sensenova_u15/model.py` (7365700) into `ComfyUI-SenseNova-U1.5-ConvRot-FIXED/sensenova_u15/model.py`
- Preserved our earlier fixes:
  - `loader.py` CRLF-tolerant tokenizer validation (WARNING not crash)
  - `nodes.py` English labels `Reference Image {index}`
  - `guidance.py` English `[Main Edit]`, `[Reference Image Roles]`, etc.
  - `web/sensenova_reference_labels_v131e.js` English
  - `.gitattributes` LF enforcement

Quant bridge unaffected — it still imports `comfy.quant_ops` for linear, not RoPE. RoPE now pure PyTorch, so quantization path is actually **more stable** (no accidental dispatch to broken CUDA kernel).

## Should we implement same fixes?

**Absolutely, mandatory for:**

- Anyone on **CUDA 13** (PyTorch 2.8+ with CUDA 13.0, which is default in new ComfyUI portable)
- **Blackwell GPUs** (RTX 5070/5080/5090, RTX 6000 Ada? Actually Blackwell is 50-series, your 4070 is safe but future users not)
- **Triton backend** (`--enable-triton-backend` flag for speed)

Without it, images look **silently wrong** — user thinks model is bad, but it's kernel bug. With fix, PyTorch reference formula is used for LLM RoPE (split-half), which is slower but correct. Vision RoPE still uses `apply_rope1` but with correct ranks, so Triton works.

**Recommendation:** 
- Deploy updated FIXED folder immediately (overwrite previous fixed deployment).
- Open PR to `Milor123/ComfyUI-SenseNova-U1.5-ConvRot` with this model.py + our tokenizer fix — otherwise fork stays broken for new GPUs.
- Keep version as `1.3.4-convrot-fix` to indicate upstream 1.3.4 + convrot + English UI.

## Verification

```bash
# Should NOT contain quant_ops in model.py
grep "quant_ops" ComfyUI-SenseNova-U1.5-ConvRot-FIXED/sensenova_u15/model.py
# → no results (good)

# Should contain rotate_half
grep "rotate_half" ComfyUI-SenseNova-U1.5-ConvRot-FIXED/sensenova_u15/model.py
# → def rotate_half

# Should contain unsqueeze(1) + squeeze(1) for interleaved
grep "unsqueeze(1)" ComfyUI-SenseNova-U1.5-ConvRot-FIXED/sensenova_u15/model.py
```

All checks pass in current FIXED folder.

## Integration impact for MERD/RoPE Lab

This fix **helps** RoPE Lab integration:

- Old code dispatched to `ck.apply_rope_split_half` — your Lab's `inv_freq_buffer` strategy tried to patch `inv_freq`, but this model had no `inv_freq` buffer, so patching failed.
- New code is **pure PyTorch** with explicit `cosine`, `sine`, `rotate_half` — much easier to monkey-patch via `method_hook`:
  ```python
  def hooked_llm_rope(query, key, positions, theta):
      # Apply NTK scaling: theta * ntk_factor
      ntk_factor = compute_ntk_factor_from_method(...)
      return original(query, key, positions, theta * ntk_factor)
  ```
- For MERD, `cos_sin_arrangement` is now clearly `split_half` (rotate_half pattern) and `dype_output_format` is `rotation_matrix` / `mrope_interleaved` but with PyTorch reference — easier to document.

So upstream fix aligns with our earlier suggestion to make RoPE patchable via `transformer_options`.

---
*Updated 2026-08-26*
