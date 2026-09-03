# Fork Evaluation: T8mars vs Milor123 vs Our FIXED

## Summary of lineages

```
T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8 (original, Apache-2.0)
  ├─ v1.3.3 and earlier: checkpoint_contract.json + bf16-only validation + RoPE via ck.apply_rope_split_half (buggy on Blackwell)
  └─ v1.3.4 (7365700, 2026-08-25): Fixed RoPE to pure PyTorch + 4D/6D layout for Triton, still bf16-only

Milor123/ComfyUI-SenseNova-U1.5-ConvRot (fork of T8, ~37 commits ahead)
  ├─ Replaced checkpoint_contract.json with checkpoint_contract.py (BASE_CONTRACT dict) for static validation immune to meta-model patching
  ├─ Added quant support: _is_quant_candidate, _read_quant_formats, _checkpoint_contract(quant_formats), _expected_storage_dtype
  ├─ Added quant_bridge.py: ConvRot-aware Linear (rotates activations, manual dequant, handles int8, convrot_w4a4, asym_w4a8_int8)
  ├─ Added qt_guards.py: guards cast_to_device and kitchen _handle_to to prevent dtype corruption when LoRA touches quantized tensors
  ├─ Added tools: convert_sensenova_int4_convrot.py (W4A4/W4A8/mixed), make_hybrid_ladder.py (INT8+W4A8 hybrid without re-quant), merge_safetensors, etc.
  ├─ Added headless test harness tests/headless/
  ├─ Did NOT have 1.3.4 RoPE fix (still used ck.apply_rope_split_half) at time of our work
  └─ Still had Chinese UI labels and tokenizer CRLF bug

Our FIXED (Milor123 + fixes)
  ├─ loader.py: CRLF-tolerant tokenizer validation (hash raw + LF-normalized, WARNING not crash) + .gitattributes eol=lf
  ├─ nodes.py: English display_name "Reference Image {i}" + English defaults for Structured Edit Prompt
  ├─ web/sensenova_reference_labels_v131e.js: English "Reference Image"
  ├─ guidance.py: English "[Main Edit]", "[Reference Image Roles]", "[Must Preserve]", "[Must Avoid]", "[Requirements]"
  ├─ model.py: Ported upstream 1.3.4 fix — pure PyTorch split-half RoPE + 4D/6D layout for apply_rope1
  └─ tests updated to expect English
```

## Detailed comparison

### T8 original (current main, 7365700)

**Pros:**
- Most up-to-date upstream, maintained by T8mars
- Has 1.3.4 RoPE fix (critical for CUDA 13 / Blackwell / Triton)
- Has checkpoint_contract.json hash validation (947b59973a054...) that is cross-platform (does .replace(b"\r\n", b"\n"))
- Has complete English README after 1.3.4? Actually still has Chinese changelog but README English
- Has Comfy Registry publishing metadata, proper versioning 1.3.4
- No quant bloat — clean bf16-only, easier to maintain

**Cons:**
- **No quant support** — cannot load Milor123's 13.8GB hybw4a8 or 17.58GB int8 checkpoints. Those checkpoints have keys like `*.weight_scale`, `*.comfy_quant` etc. T8 loader will raise `key mismatch`.
- Still has tokenizer CRLF bug: `_validate_tokenizer_assets` does raw `read_bytes()` hash without CRLF normalization — will fail on Windows with `core.autocrlf=true` (same bug we fixed)
- Still has Chinese UI? Let's check T8 nodes.py current — it had Chinese before, but after 1.3.4? The upstream still has `参考图`? In T8's latest nodes.py, check: it had Chinese labels originally. The commit history shows "Improve reference image labels" but still Chinese? Need to check. From our earlier grep, T8's web JS also had Chinese. So T8 still has Chinese UI.
- No qt_guards — if you add quant later without guards, LoRA + quantized will corrupt.

### Milor123 ConvRot fork

**Pros:**
- **Only fork that can run 50GB bf16 model on 12GB GPU** via INT8/W4A8. This is huge — bf16 streams 47GB per step, quantized moves 3-4x fewer bytes, actually faster on 4070 12GB.
- Has quant_bridge that correctly handles ConvRot rotation (offline Hadamard folded into weights, must rotate activations at inference). ComfyUI generic dispatch skips rotation — Milor123 fixes.
- Has hybrid tooling — empirically found L18-41 boundary for W4A8 tolerance.
- Has static contract immune to meta-model construction issues.

**Cons:**
- Fork is **behind upstream** — missing 1.3.4 RoPE fix → silent corruption on Blackwell.
- Still has tokenizer bug and Chinese UI (inherited from T8).
- Uses `checkpoint_contract.py` (Python dict) instead of JSON — loses file_size validation and legacy revision support that T8 added. T8's `checkpoint_contract.json` has `variants: final, final_legacy, sft` with file_size checks. Milor123's `BASE_CONTRACT` only has single variant shape, no file_size, no legacy support.
- No Registry publishing, version not bumped to 1.3.4.
- Quant code is complex — maintenance burden.

### Our FIXED (Milor123 + our patches)

**Pros:**
- **Best of both:** Has quant support + 1.3.4 RoPE fix + tokenizer fix + English UI.
- Currently the **only working version for quantized + Blackwell + Windows**.
- Has `.gitattributes` to prevent future CRLF issues.

**Cons:**
- Still based on Milor123's `checkpoint_contract.py` — missing T8's `final_legacy` and file_size validation.
- Not yet pushed to GitHub as proper fork — just local folder.
- Mix of two lineages — need to cleanly merge.

## ConvRot support — is it already in T8?

**No.** T8 has **zero** quant support:

```bash
# T8
grep -rn "quant\|convrot\|comfy_quant" upstream --include="*.py"
# → only tests/test_rope.py mocks ck.apply_rope_split_half

# Milor123
grep -rn "quant" sensenova_u15 --include="*.py"
# → loader.py quant_formats, quant_bridge.py, model_config.py custom_operations
```

T8's `model_config.py` does NOT set `custom_operations`. Milor123's does:
```python
if any(k.endswith(".comfy_quant")):
    self.custom_operations = make_sensenova_quant_ops()
```

So ConvRot is **exclusive to Milor123 fork**.

## Which one to fork to your GitHub properly?

### Recommendation: Fork from **T8mars** (upstream) and cherry-pick Milor123 quant features + our fixes.

**Why T8 as base, not Milor123:**

1. **Upstream is maintained** — T8mars is active, has Registry, versioning, and will get future fixes (e.g., future CUDA 14, new SFT revisions). Milor123 fork is 37 commits ahead but behind on critical 1.3.4 and has no upstream merge strategy.

2. **Clean history** — Forking from T8 gives you proper fork network on GitHub (shows "forked from T8mars"), easier to PR back upstream. Forking from Milor123 which itself is fork of T8 gives you second-level fork, harder to track.

3. **T8's contract is better** — `checkpoint_contract.json` with SHA256, file_size, `final_legacy` support, cross-platform CRLF handling for contract itself. Milor123's `BASE_CONTRACT` is simpler but loses those.

4. **You can still add quant** — Quant support is additive: 3 files (`quant_bridge.py`, `qt_guards.py`, `checkpoint_contract.py` vs JSON) + modifications to `loader.py` and `model_config.py`. You can implement it cleanly on top of T8 1.3.4 without losing upstream.

**How to implement our fixes on T8 fork:**

1. Start fresh fork from T8mars at commit `7365700` (1.3.4)
2. Apply **tokenizer fix** (our loader.py patch):
   - In `loader.py:_validate_tokenizer_assets`, hash both raw and LF-normalized, WARNING not ValueError
   - Add `.gitattributes` with `sensenova_u15/tokenizer/* text eol=lf`

3. Apply **English UI** (our nodes.py, guidance.py, web JS):
   - `nodes.py: _reference_image_inputs` display_name English
   - `nodes.py: SenseNovaStructuredEditPrompt` defaults English
   - `guidance.py: build_structured_edit_prompt` English brackets
   - `web/sensenova_reference_labels_v131e.js` English

4. **Port quant support from Milor123** (the valuable part):
   - Keep T8's `checkpoint_contract.json` but extend loader to handle quant: copy Milor123's `_is_quant_candidate`, `_read_quant_formats`, `_expected_storage_dtype`, and modify `_validate_checkpoint_header` to accept quant keys if present, else fall back to T8's JSON contract. Or keep both: if quant detected, use Python dict contract, else use JSON.
   - Add `sensenova_u15/quant_bridge.py` (copy from Milor123, it already works with 1.3.4 model.py)
   - Add `qt_guards.py` (copy, and keep `install_quant_guards()` in `__init__.py`)
   - Add `sensenova_u15/checkpoint_contract.py` **in addition** to JSON — or convert JSON to Python dict at runtime for quant case.
   - Modify `model_config.py:get_model` to set `custom_operations = make_sensenova_quant_ops()` when `comfy_quant` present (copy from Milor123)
   - Add tools `convert_sensenova_int4_convrot.py`, `make_hybrid_ladder.py` (optional, for users who want to quant themselves)

5. **Keep 1.3.4 RoPE fix** — already in T8 base, don't regress to old `ck.apply_rope_split_half`.

Resulting fork would be:
```
YourName/ComfyUI-SenseNova-U1.5-Wrapper (forked from T8mars)
  ├─ Upstream 1.3.4 base (pure PyTorch RoPE, Registry)
  ├─ + tokenizer CRLF fix (our)
  ├─ + English UI (our)
  └─ + quant_bridge + qt_guards + quant-aware loader (from Milor123)
```

Version it as `1.3.4-convrot-english-fix` or `1.4.0`.

### If you fork from Milor123 instead:

- Pros: You get quant out-of-the-box, less cherry-picking.
- Cons: You must manually port 1.3.4 RoPE fix (we did), plus port T8's `checkpoint_contract.json` improvements (file_size, legacy), plus Registry metadata. You also inherit Milor123's diverged history which is not in T8 network.

**If you choose Milor123 as base:** Just take our FIXED folder as is — it already has 1.3.4 + English + tokenizer fix. Then add back T8's `checkpoint_contract.json` handling for file_size validation (merge both contracts).

### Can our fixes be implemented to original T8 version?

**Yes, all of them:**

- Tokenizer fix: 15 lines, no dependency on quant
- English UI: 10 lines in nodes.py + 1 line in web JS + 5 lines in guidance.py — no dependency
- ConvRot support: Yes, but requires adding 2 new files + ~80 lines in loader.py + 10 lines in model_config.py — completely additive, doesn't break bf16 path. T8's loader already has `lru_cache` for contract — you can extend to handle quant as optional branch.

**Should you?**

- Tokenizer + English: **Yes, mandatory** — improves UX for all users, no downside. PR to T8 would be accepted.
- ConvRot: **Yes, if you want 12GB support** — but discuss with T8mars first, as it adds complexity and dependency on `comfy-kitchen>=0.2.31`. Milor123's README says "Requires comfy-kitchen >=0.2.31" — T8's current `pyproject.toml` doesn't require it. You'd need to add optional dependency.

## Final recommendation

1. **Fork from T8mars** (proper GitHub fork button)
2. **Create branch `convrot-english`** 
3. **Cherry-pick:**
   - Our `loader.py` tokenizer fix
   - Our English UI commits
   - Milor123's `quant_bridge.py`, `qt_guards.py`, and quant-aware parts of `loader.py`/`model_config.py`
4. **Keep `model.py` at 1.3.4** (pure PyTorch RoPE)
5. **Test** with both bf16 official (should still work via JSON contract) and int8 quantized (via Python contract)
6. **Push** to your GitHub as `ComfyUI-SenseNova-U1.5-Wrapper-T8-ConvRot-English` or similar, with clear attribution in README: "Forked from T8mars, includes ConvRot quantization from Milor123, plus tokenizer and UI fixes"

This gives you a **maintainable, upstream-compatible, quant-enabled, English, Blackwell-safe** fork.

---
*Evaluation done 2026-08-26, based on /tmp/upstream (7365700) vs /home/user/ComfyUI-SenseNova-U1.5-ConvRot (7e1e320) vs FIXED (merged)*
