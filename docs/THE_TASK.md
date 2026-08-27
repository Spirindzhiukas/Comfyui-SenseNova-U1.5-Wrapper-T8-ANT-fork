# THE_TASK.md — Maintenance & Continuation Contract for SenseNova U1.5 Fork

**For: Future self (Arena agent) in session where new T8 fork repo is connected**
**Source fork to create:** Fork from `T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8` at commit `7365700` (v1.3.4) — proper GitHub fork via UI, not clone.
**Target fork name suggestion:** `YourName/ComfyUI-SenseNova-U1.5-Wrapper-T8-ConvRot-English` or `ComfyUI-SenseNova-U1.5-Plus`
**Purpose:** Turn fresh T8 fork into amalgamation of: T8 1.3.4 (clean base) + Milor123 ConvRot int8/w4a8 support + our fixes (tokenizer CRLF + English UI + RoPE 1.3.4 preservation) + RoPE Lab / MERD research, **without breaking future T8 updates**.

---

## 0. Context & Provenance (Where we came from)

- Original bug: `ValueError: tokenizer asset digest mismatch: config.json` on Windows due to `core.autocrlf=true` → CRLF vs LF. Provenance: user error log, `loader.py: _validate_tokenizer_assets()` raw `read_bytes()` hash, reproduced via `sha256sum` LF `6497591f...` vs CRLF `9a7324...`.
- Original node pack: `Milor123/ComfyUI-SenseNova-U1.5-ConvRot` (37 commits ahead of T8, forked from T8). It added quant support but was behind on 1.3.4 and had same tokenizer bug + Chinese UI.
- Upstream critical commit: `T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8/commit/73657001179ccb28479e29a09e69aa7f67e4277d` — Fix CUDA and Triton RoPE compatibility (v1.3.3 → 1.3.4). Changes `model.py` from `comfy.quant_ops.ck.apply_rope_split_half` (broken on Blackwell/CUDA13) to pure PyTorch `rotate_half`, and fixes `apply_rope1` rank from 5D/3D to 6D/4D.
- Our work in this chat: Created `ComfyUI-SenseNova-U1.5-ConvRot-FIXED` with:
  - `loader.py` CRLF-tolerant (hash raw + normalized, WARNING not crash) + `.gitattributes`
  - `nodes.py` English `Reference Image {i}` + English defaults for Structured Edit Prompt
  - `web/sensenova_reference_labels_v131e.js` English
  - `guidance.py` English `[Main Edit]` etc.
  - `model.py` ported 1.3.4 fix
  - Research docs: `FIX_REPORT.md`, `UPDATE_REPORT_1.3.4.md`, `MODEL_DECOUPLING_ANALYSIS.md`, `SenseNova_MERD_and_RoPE_Lab_integration_research.md`, `ROPE_AND_EMBEDDING_NODES_PROPOSAL.md`, `FORK_EVALUATION.md`
- RoPE Lab context: User has `ANT_NODES/RoPE_Lab` suite with adapters (dy_ntk, yarn, vision_yarn, hope, sega, sabre, etc.), MERD database v1.7 (23 detectors, 6 patching strategies). SenseNova uses 3D M-RoPE (t,h,w) with axes_dim [64,32,32] and thetas [5M,10k,10k] — currently classified as `mrope_interleaved` which Lab aborts. Needs `method_hook` strategy like MiniMax-H3.
- Model properties: Official `sensenova/SenseNova-U1.5-8B-MoT` (18B params, BF16 50GB), NEO-unify MoT architecture (dual weights `_mot_gen`), Qwen3-based but custom trained (pure_llm=false), pixel VAE dummy (HiDreamO1Pixel, 3ch), text encoder dummy (tokenizer-only Qwen2Tokenizer + chat template).

## 1. Goals for New T8 Fork

**Amalgamation without incompatibility:**

- Keep T8's clean base: `checkpoint_contract.json` (with file_size, final_legacy, cross-platform), Registry publishing, versioning, 1.3.4 RoPE fix.
- Add Milor123's quant: int8 ConvRot (17.58GB), hybw4a8 L18-41 (13.80GB), W4A4, mixed, hybrid ladder tooling — but as **optional additive** path that doesn't break bf16.
- Add our fixes: tokenizer CRLF + English UI + .gitattributes.
- Preserve future mergeability: T8 updates should be mergeable via `git merge upstream/main` without conflicts on our patches.

## 2. Fixes / Changes to Port (If They Apply)

### 2.1 Tokenizer CRLF Fix (MANDATORY, from FIX_REPORT.md)

**File:** `sensenova_u15/loader.py:_validate_tokenizer_assets()`

**T8 current state (to verify in next session):** Does raw `hashlib.sha256(path.read_bytes()).hexdigest()` → will fail on Windows.

**Task:** Replace with:

```python
def _validate_tokenizer_assets():
    asset_dir = Path(__file__).resolve().parent / "tokenizer"
    for name, expected in TOKENIZER_ASSET_SHA256.items():
        path = asset_dir / name
        if not path.is_file():
            raise ValueError(f"tokenizer asset missing: {name}")
        raw = path.read_bytes()
        canonical = raw.replace(b"\r\n", b"\n")
        digest_raw = hashlib.sha256(raw).hexdigest()
        digest_canonical = hashlib.sha256(canonical).hexdigest()
        if digest_raw != expected and digest_canonical != expected:
            print(f"[SenseNova-U1.5] WARNING: tokenizer asset {name} mismatch: expected={expected} got_raw={digest_raw} canonical={digest_canonical}. Continuing. Fix: git config --global core.autocrlf false and re-clone")
            continue
```

**Provenance:** Our fix, verified via `sha256sum` LF vs CRLF.

**Compatibility:** No break — only makes validation tolerant, adds warning.

**Also:** Add `.gitattributes`:
```
sensenova_u15/tokenizer/* text eol=lf
*.py text eol=lf
*.json text eol=lf
*.js text eol=lf
*.txt text eol=lf
```

### 2.2 English UI Translation (MANDATORY, from FIX_REPORT.md)

**Files:**
- `nodes.py:_reference_image_inputs()`
  - From: `display_name=f"参考图 {index} (Image-{index})"`
  - To: `display_name=f"Reference Image {index} (Image-{index})"`
  - Provenance: grep "参考"

- `nodes.py: SenseNovaStructuredEditPrompt`
  - `image_roles` default: From Chinese `参考图作为主画面...` To `Image-1 is the main image to edit. When using multiple references, clearly state what each Image-1, Image-2 provides.`
  - `preserve`: From `保持主体身份...` To `Keep subject identity, pose, composition, background, lighting, camera angle, and aspect ratio unchanged.`
  - `avoid`: From `不要增加无关主体...` To `Do not add unrelated subjects, do not alter unspecified areas, do not generate watermarks or garbled text.`

- `web/sensenova_reference_labels_v131e.js: referenceLabel()`
  - From: `` `参考图 ${number} (Image-${number})` ``
  - To: `` `Reference Image ${number} (Image-${number})` ``
  - This JS overrides Python labels at runtime — main culprit.

- `sensenova_u15/guidance.py: build_structured_edit_prompt()`
  - From: `【主要修改】`, `参考图职责`, `必须保持`, `禁止出现`, `【执行要求】...`
  - To: `[Main Edit]`, `Reference Image Roles`, `Must Preserve`, `Must Avoid`, `[Requirements]\nOnly modify what is explicitly requested above; keep all other areas consistent with the original.`
  - Note: This changes prompt sent to model — model understands English, but test with Chinese prompt to ensure no quality drop. Keep Chinese as comment if needed for future reference.

**Compatibility:** Translation doesn't break future T8 merges if you keep original Chinese as comment above new English line, e.g.:
```python
# Original: 参考图 {index} (Image-{index})
display_name=f"Reference Image {index} (Image-{index})"
```
Then when T8 updates with new Chinese label, you see conflict and can re-translate.

### 2.3 Upstream 1.3.4 RoPE Fix (VERIFY, don't regress)

**File:** `sensenova_u15/model.py`

**T8 current state at 7365700:** Already fixed — pure PyTorch `rotate_half`, no `comfy.quant_ops`, vision RoPE uses `unsqueeze(1)` + 6D rotation.

**Task in next session:** Verify file contains:
```python
def rotate_half(value): ...
query * cosine + rotate_half(query) * sine
...
rotation = stack(...).reshape(1,1,*angles.shape,2,2)
return apply_rope1(value.float().unsqueeze(1), rotation).squeeze(1)
```
And does NOT contain `comfy.quant_ops.ck.apply_rope_split_half`.

If T8 has moved beyond 1.3.4 and reintroduced kitchen kernel, re-apply fix with comment about Blackwell.

**Provenance:** Commit 7365700, CHANGELOG.md, our UPDATE_REPORT_1.3.4.md.

**Compatibility:** Keep pure PyTorch — more patchable for RoPE Lab.

### 2.4 Milor123 ConvRot Quant Support (PORT, but as optional)

**This is the valuable part exclusive to Milor123.**

**Files to add from Milor123 fork:**
- `sensenova_u15/quant_bridge.py` (copy as-is, 125 LOC, handles int8, convrot_w4a4, asym_w4a8_int8)
- `qt_guards.py` (copy as-is, 70 LOC, guards cast_to_device and _handle_to)
- `sensenova_u15/checkpoint_contract.py` (Python dict BASE_CONTRACT) — BUT keep T8's `checkpoint_contract.json` as primary. Strategy: keep both files, use JSON for bf16, Python dict for quant.

**Files to modify:**

- `__init__.py`:
  ```python
  from .qt_guards import install_quant_guards
  install_quant_guards()
  WEB_DIRECTORY = "./web"
  ```
  Keep T8's original `__init__.py` (which had no qt_guards) and add guard installation with env var `SENSENOVA_NO_QT_GUARDS` check (already in qt_guards.py).

- `sensenova_u15/model_config.py:get_model()`:
  ```python
  if any(key.endswith(".comfy_quant") for key in state_dict) and not os.environ.get("SENSENOVA_NO_BRIDGE"):
      from .quant_bridge import make_sensenova_quant_ops
      self.custom_operations = make_sensenova_quant_ops()
  ```
  This is additive — only triggers for quantized checkpoints.

- `sensenova_u15/loader.py`:
  **CRITICAL MERGE STRATEGY TO KEEP COMPATIBILITY:**
  Keep T8's JSON-based validation for bf16 (with file_size, final_legacy), but add quant branch from Milor123:

  ```python
  def _is_quant_candidate(name, shape): ... # from Milor123
  def _read_quant_formats(checkpoint): ... # from Milor123
  def _expected_storage_dtype(...): ... # from Milor123
  def _checkpoint_contract_py(quant_formats): ... # from Milor123's BASE_CONTRACT logic, rename to avoid clash with _checkpoint_contract (JSON)

  def _validate_checkpoint_header(checkpoint, model_path):
      quant_formats = _read_quant_formats(checkpoint)
      if quant_formats: # quantized path
          contract = _checkpoint_contract_py(quant_formats)
          # ... Milor123 validation
      else: # bf16 path — use T8's JSON logic
          contract = _checkpoint_contract_json(profile) # original T8
          # ... T8 validation with file_size
  ```

  This way future T8 updates to `checkpoint_contract.json` (new revisions) still work for bf16, and quant path uses Python dict.

- `tools/`:
  - Copy `convert_sensenova_int4_convrot.py` and `make_hybrid_ladder.py` from Milor123 — optional, for users who quant themselves. Add to `.comfyignore`? Keep.

**Dependencies:**
- Add to `pyproject.toml` or README: `comfy-kitchen>=0.2.31` required for quant, but make it optional — bf16 works without. Use try/except import.

**Compatibility preservation:**
- Quant code only activates when `*.comfy_quant` keys present. For bf16 official checkpoints (no such keys), code path is identical to T8 upstream → future T8 merges won't conflict on bf16 path.
- Keep `checkpoint_contract.json` SHA validation for bf16 — don't delete.
- Keep `qt_guards` behind env var.

## 3. RoPE and RoPE Lab Deliberations (Very Important)

**Include full analysis from `SenseNova_MERD_and_RoPE_Lab_integration_research.md`:**

- SenseNova uses 3D M-RoPE: axes_dim [64,32,32] (pairs [32,16,16]), thetas [5M,10k,10k], axis_identity [t,h,w], spatial [1,2], temporal 0, concat_order t_h_w, base_patch_grid [64,64], cos_sin_arrangement split_half, dype_output_format mrope_interleaved, uses_qk_norm true, uses_attention_mask true.
- Vision RoPE interleaved with apply_rope1, 4D input + 6D rotation.
- Current RoPE Lab aborts on mrope_interleaved (Ideogram4). Needs new strategy `sensenova_mot_hook`.
- Hooks needed:
  - `SenseNovaMotRopeHook` in `core/method_hook.py` that patches `_apply_llm_rope`, `_apply_interleaved_rope`, `Attention._project`
  - Make RoPE functions read from `transformer_options["rope_theta_t"]`, `["rope_theta_hw"]`, `["rope_theta_vision"]` to allow dynamic scaling without monkey-patch.
  - Sampler patch must handle pixel latent [B,3,H,W] for SEGA FFT — compute tokens H/32*W/32.
  - Prefix cache key must include RoPE theta — clear cache when dype changes.
  - Frozen imports: `optimized_attention`, `pad_to_patch_size`, `apply_rope1` — need sys.modules walk.
- Dype math: `base_ntk = scale ** (dim/(dim-2))`, `kappa = dype_scale * t^dype_exp`, `ntk_factor = base_ntk**kappa` — directly applicable per axis.
- SEGA: FFT on pixel latent possible, needs `pass_latent=True`.
- MERD fields needed: `per_axis_theta`, `per_axis_theta_vision`, `mot_enabled`, `mot_dual_weights`, new latent family `sensenova_pixel` (channels 3, compression 1), new patching strategy `sensenova_mot_hook`, `inline_rope_function` list.
- Roadmap Phase 0-6 from research doc.

**Task for next session:** Copy relevant sections from `SenseNova_MERD_and_RoPE_Lab_integration_research.md` and `ROPE_AND_EMBEDDING_NODES_PROPOSAL.md` into new fork's `docs/rope_lab_integration.md` and reference in THE_TASK.md.

## 4. Task List for Next Session (From Scratch Analysis)

**When new T8 fork repo is connected:**

1. **Analyze current state:**
   - `git log --oneline -20` — what version? Is 1.3.4 present? Is there newer?
   - `cat sensenova_u15/model.py | grep -n "quant_ops\|rotate_half"` — verify RoPE fix
   - `cat sensenova_u15/loader.py | grep -n "comfy_quant\|BASE_CONTRACT"` — verify no quant yet
   - `grep -rn "参考" --include="*.py" --include="*.js"` — check Chinese still present?
   - `cat pyproject.toml` — version, dependencies

2. **Create memory.md:**
   - Task: Create `memory.md` in new fork root with coding/updating protocols (see Section 5).

3. **Implement tokenizer fix:**
   - Edit `loader.py`, add `.gitattributes`

4. **Implement English UI:**
   - Edit `nodes.py`, `guidance.py`, `web/*.js` with English, keep Chinese as comment

5. **Implement quant support (additive):**
   - Copy `quant_bridge.py`, `qt_guards.py` from our FIXED folder (`/home/user/ComfyUI-SenseNova-U1.5-ConvRot-FIXED/`)
   - Merge `loader.py` quant branch while keeping JSON contract for bf16
   - Edit `model_config.py`, `__init__.py`
   - Test with both bf16 (should use JSON) and int8 (should use Python dict) — need dummy test without actual 17GB file, use header validation only

6. **Verify 1.3.4 RoPE fix preserved:**
   - Ensure no regression

7. **Add research docs:**
   - Copy `MODEL_DECOUPLING_ANALYSIS.md`, `SenseNova_MERD_and_RoPE_Lab_integration_research.md`, `ROPE_AND_EMBEDDING_NODES_PROPOSAL.md`, `FORK_EVALUATION.md`, `UPDATE_REPORT_1.3.4.md` into `docs/`

8. **Create final amalgamation:**
   - Ensure fork can load official Final (bf16) + Milor123 int8/hybw4a8
   - Ensure English UI
   - Ensure tokenizer works on Windows CRLF
   - Ensure RoPE is pure PyTorch (Blackwell-safe)

9. **Version bump:**
   - `pyproject.toml` version → `1.3.4-convrot-english-fix` or `1.4.0`
   - Update CHANGELOG.md with English entries for our fixes + quant port

10. **Prepare for future merges:**
    - Document in `memory.md` how to merge upstream: `git remote add upstream https://github.com/T8mars/...`, `git fetch upstream`, `git merge upstream/main`, then re-apply translation protocol.

## 5. memory.md — Coding / Updating Protocols

**Task:** Create `memory.md` in new fork root with following protocols:

```markdown
# memory.md — Maintenance Protocols for SenseNova Fork

## 1. Translation Protocol (MANDATORY for any upstream merge)

- NEVER merge code with Chinese labels/comments blindly.
- When upstream (T8mars) or other forks add Chinese:
  1. Keep original Chinese as comment: `# Original: 参考图...`
  2. Translate to English below it.
  3. For UI display_name, use English only.
  4. For structured prompts (guidance.py), use English brackets [Main Edit] etc., but keep Chinese example as comment for quality reference.
- Use translator or ask user if unsure — don't guess.

## 2. Evaluation Against Own Changes

- Before merging any upstream commit, diff against our changes:
  - `loader.py`: Our CRLF fix must stay — if upstream changes tokenizer validation, merge our tolerant logic.
  - `model.py`: Our 1.3.4 pure PyTorch RoPE must stay — if upstream reintroduces `ck.apply_rope_split_half`, re-apply fix with Blackwell comment.
  - `quant_bridge.py`, `qt_guards.py`: These are ours (from Milor123) — if upstream adds its own quant, compare and keep ConvRot-aware version.
  - `nodes.py`, `guidance.py`, `web/*.js`: Our English UI must stay — re-translate any new Chinese.
- Always check `git diff` after merge, ensure no regression.

## 3. Compatibility Preservation

- Keep `checkpoint_contract.json` as primary for bf16 — don't delete.
- Quant support must be optional: only activate when `comfy_quant` keys present.
- Keep `qt_guards` behind env var `SENSENOVA_NO_QT_GUARDS`.
- Keep `custom_operations` behind `SENSENOVA_NO_BRIDGE`.
- Don't add hard dependency on `comfy-kitchen` for bf16 path — use try/except.

## 4. Testing Protocol

- After any merge:
  - `grep -rn "参考\|保持\|不要" --include="*.py" --include="*.js" sensenova_u15 nodes.py web` → should be 0 for main code (tests may have Chinese examples, okay)
  - `grep -n "apply_rope_split_half" sensenova_u15/model.py` → should be 0
  - `grep -n "quant_ops" sensenova_u15/model.py` → should be 0
  - Simulate CRLF: `python -c "import hashlib; raw=open('sensenova_u15/tokenizer/config.json','rb').read(); print(hashlib.sha256(raw).hexdigest()); print(hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest())"` → both should match expected or warning path works
  - If possible, run `pytest tests/test_rope.py` — should pass with new ranks

## 5. Versioning

- Follow T8's versioning: upstream is 1.3.4, we are 1.3.4-convrot-english-fix or 1.4.0
- Update CHANGELOG.md with English entries, keep Chinese changelog as is but add English below.

## 6. RoPE Lab Integration Notes

- SenseNova uses 3D M-RoPE with per_axis_theta [5M,10k,10k], axes_dim [64,32,32], mrope_interleaved, method_hook strategy.
- Future RoPE Lab work needs SenseNovaMotRopeHook — see docs/rope_lab_integration.md
- Don't break `transformer_options["sensenova_prefix_cache"]` — RoPE Lab dynamic methods need to clear it.

## 7. Provenance

- Keep this file updated with where each fix came from (commit hash, file, line).
```

## 6. Maintenance Contract

- This fork is amalgamation — must stay mergeable with T8 upstream.
- When T8 releases new version (e.g., 1.3.5), merge via `git merge upstream/main`, then re-apply translation protocol + verify tokenizer fix + RoPE fix.
- When Milor123 releases new quant (e.g., new hybrid), cherry-pick his `quant_bridge.py` and loader changes, but keep T8's JSON contract.
- Always update `memory.md` with new protocols learned.
- Keep research docs in `docs/` for future sessions.

## 7. Files to Reference in Next Session

- Our FIXED folder: `/home/user/ComfyUI-SenseNova-U1.5-ConvRot-FIXED/` — contains all fixes ready to copy
- Upstream T8: `https://github.com/T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8` at `7365700`
- Milor123: `https://github.com/Milor123/ComfyUI-SenseNova-U1.5-ConvRot` at `7e1e320`
- Research docs: All `*.md` in `/home/user/` — copy to new fork's `docs/`

---

*End of THE_TASK.md — this is your contract for next session when new fork repo is connected.*
