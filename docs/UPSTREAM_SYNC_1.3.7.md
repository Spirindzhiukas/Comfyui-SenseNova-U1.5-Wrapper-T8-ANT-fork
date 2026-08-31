# Upstream synchronization report — T8mars 1.3.7

**Reviewed:** 2026-08-31  
**Upstream:** [`T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8`](https://github.com/T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8)  
**Upstream range:** 1.3.6 → 1.3.7 (`d53a2975` through `5836acbc`)  
**Fork release:** 1.4.3

## Scope

The two new upstream pull requests after this fork's previous synchronization were inspected individually:

| PR | Upstream commits | Purpose | Decision |
| --- | --- | --- | --- |
| [#5 — Fix PyTorch attention mask dtype](https://github.com/T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8/pull/5) | feature `8f322794`; merge `b8e4bf34` | Cast the prefix mask to the Q/K/V dtype before attention dispatch and add a regression test | Integrated, with the test adapted to this fork's RoPE-options signature |
| [#6 — Release SenseNova wrapper 1.3.7](https://github.com/T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8/pull/6) | release `0fa5a11e`; merge `5836acbc` | Record the attention fix, set upstream package version to 1.3.7, and update its version assertion | Semantically integrated; this fork advances 1.4.2 → 1.4.3 instead of downgrading to 1.3.7 |

Upstream PR #4 and its two core workflows were already present in this fork and therefore required no further changes.

## PR #5 investigation and integration

### Root cause

The SenseNova prefix pass can have a float32 attention mask while its projected query, key, and value tensors are bfloat16. Upstream reports that PyTorch SDPA on CUDA can produce finite but numerically incorrect masked-attention results for that dtype combination. Because the result is finite, this can appear as abnormal generations rather than a clear exception.

### Code change

`Attention.forward_prefix` in `sensenova_u15/model.py` now passes:

```python
mask=attention_mask.to(query.dtype)
```

instead of passing the original mask directly. The conversion happens after projection and immediately before `optimized_attention`, so it follows the actual query dtype selected by the model/ComfyUI runtime rather than assuming BF16.

### Fork-specific adaptation

Upstream 1.3.7 calls `_project(hidden_states, indexes, False)`. This fork intentionally extended `_project` with a `transformer_options` argument so that per-axis RoPE bases can be changed and included in prefix-cache behavior. That extension was retained:

```python
query, key, value = self._project(
    hidden_states, indexes, False, transformer_options
)
```

The upstream regression test was also adapted so its `_project` test double accepts and verifies the same `transformer_options` object. This catches both the new dtype requirement and accidental removal of the fork's RoPE plumbing.

### Quantization impact

The fix is independent of weight storage format and is valid for both BF16 and ConvRot-quantized loads: it only aligns the runtime mask with the projected query. No loader, quantization sidecar, operation-selection, ConvRot bridge, or `QuantizedTensor` guard code was changed.

## PR #6 investigation and integration

PR #6 contains release bookkeeping only:

- a five-line upstream 1.3.7 changelog entry;
- `pyproject.toml` version 1.3.6 → 1.3.7;
- the corresponding upstream metadata-test expectation.

This fork was already at 1.4.2. Applying the upstream version literally would be a version regression and would misrepresent the additional ConvRot, CRLF, English-UI, and RoPE-option work. The release intent was therefore preserved by:

- advancing this fork to **1.4.3**;
- updating its metadata regression test to 1.4.3;
- recording the upstream 1.3.7 attention fix in `CHANGELOG.md`;
- updating README and maintenance provenance to say that fixes are synchronized through upstream 1.3.7.

The fork remains unpublished on the ComfyUI Registry; no upstream publication workflow or Registry ownership claim was adopted.

## Preserved fork invariants

The synchronization was reviewed against `memory.md`. The following behavior remains intact:

1. **ConvRot quantization** — format detection, derived checkpoint contracts, `quant_config` wiring, capability probing, bridge operations, cast guards, and post-load invariants are unchanged.
2. **Blackwell-safe RoPE** — split-half RoPE remains pure PyTorch; no `comfy.quant_ops` or `apply_rope_split_half` call was reintroduced.
3. **RoPE Lab readiness** — `transformer_options` still reaches `_project`, per-axis theta resolution, and the prefix-cache key.
4. **CRLF-safe assets** — tokenizer digest normalization/warning behavior and LF attributes are unchanged.
5. **English UI and docs** — translated node labels, prompts, workflows, frontend extension, and English-first README layout are unchanged.
6. **Checkpoint validation** — the bundled 1116-tensor JSON contract and its digest pin are unchanged.
7. **Existing native workflows** — upstream PR #4's core text-to-image and edit workflows remain present.

## Files changed for this synchronization

- `sensenova_u15/model.py` — integrate the mask dtype cast.
- `tests/test_model_structure.py` — add the adapted upstream regression test.
- `pyproject.toml`, `tests/test_metadata.py` — release this fork as 1.4.3.
- `CHANGELOG.md`, `README.md`, `README_CN.md`, `memory.md` — document upstream level, decisions, and maintenance invariants.
- `docs/UPSTREAM_SYNC_1.3.7.md` — this audit.

## Validation

The integration should be accepted only if all of the following pass:

```bash
python -m compileall -q .
ruff check .
python -m unittest discover -s tests -t .
```

Additional invariant checks:

```bash
# Must produce no matches: split-half RoPE remains on the safe PyTorch path.
grep -n "quant_ops\|apply_rope_split_half" sensenova_u15/model.py

# Must show the integrated 1.3.7 fix.
grep -n "attention_mask.to(query.dtype)" sensenova_u15/model.py
```

Large model inference is outside the unit-test environment. The regression test intercepts ComfyUI's attention dispatch and directly verifies that an FP32 input mask arrives as BF16 when the projected query is BF16.
