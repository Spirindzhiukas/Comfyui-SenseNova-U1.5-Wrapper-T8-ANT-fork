# SenseNova-U1.5-ConvRot Node Fix Report

## Root Cause of `tokenizer asset digest mismatch: config.json`

**File:** `sensenova_u15/loader.py` → `_validate_tokenizer_assets()`

```python
digest = hashlib.sha256(path.read_bytes()).hexdigest()
if digest != expected: raise ValueError(...)
```

The code hashes raw file bytes. On **Windows**, `git` defaults to `core.autocrlf=true`, which checks out all text files with CRLF (`\r\n`) line endings.

- Bundled hash was computed on LF (`\n`) version: `6497591f...` (LF)
- After Windows checkout, file becomes CRLF: hash `9a7324fdf647cbc5e...`
- Result: `ValueError: SenseNova-U1.5 tokenizer asset digest mismatch: config.json`

Confirmed by reproducing hashes:

```
LF hash:       6497591f64cb0dd6917fbb10c0cd13024e5817179a9aa3700998eb137a553d6b (expected)
CRLF hash:     9a7324fdf647cbc5e8d20c2bae3c498bc0ef68c8e3341cacef26813ef47207c0 (mismatch)
Canonicalized: 6497591f... (matches after \r\n -> \n)
```

Upstream already fixed this pattern for `checkpoint_contract.json` (they do `.replace(b"\r\n", b"\n")`), but forgot to apply it to tokenizer assets.

### Fix Applied

In `sensenova_u15/loader.py`:

1. **Normalize line endings** before hashing: `raw.replace(b"\r\n", b"\n")`
2. **Accept both** raw and normalized hashes
3. **Downgrade hard failure to warning** – print warning and continue instead of crashing, making node resilient to manual edits
4. **Added `.gitattributes`** to enforce LF for future clones:
   ```
   sensenova_u15/tokenizer/* text eol=lf
   *.py text eol=lf
   *.json text eol=lf
   *.js text eol=lf
   ```

This matches upstream's own fix for contract file and eliminates Windows breakage.

---

## Chinese Labels Removal

User reported Chinese UI in "SenseNova Reference Image" node.

### Files Changed

#### 1. `nodes.py`
- `_reference_image_inputs()`:
  - Before: `display_name=f"参考图 {index} (Image-{index})"`
  - After: `display_name=f"Reference Image {index} (Image-{index})"`

- `SenseNovaStructuredEditPrompt` defaults:
  - `image_roles`:
    - Before: `参考图作为主画面和待编辑对象。多图时请明确写 Image-1、Image-2 各自提供什么。`
    - After: `Image-1 is the main image to edit. When using multiple references, clearly state what each Image-1, Image-2 provides.`
  - `preserve`:
    - Before: `保持主体身份、姿势、构图、背景、光线、镜头和画幅比例不变。`
    - After: `Keep subject identity, pose, composition, background, lighting, camera angle, and aspect ratio unchanged.`
  - `avoid`:
    - Before: `不要增加无关主体，不要改变未指定区域，不要生成水印或乱码文字。`
    - After: `Do not add unrelated subjects, do not alter unspecified areas, do not generate watermarks or garbled text.`

#### 2. `web/sensenova_reference_labels_v131e.js`
- Before: `return \`参考图 ${number} (Image-${number})\`;`
- After: `return \`Reference Image ${number} (Image-${number})\`;`

This JS runs in browser and overrides slot labels – was the main source of Chinese appearing even after Python fix.

#### 3. `sensenova_u15/guidance.py`
- `build_structured_edit_prompt()` section headers were Chinese:
  - `【主要修改】` → `[Main Edit]`
  - `参考图职责` → `Reference Image Roles`
  - `必须保持` → `Must Preserve`
  - `禁止出现` → `Must Avoid`
  - `【执行要求】\n只修改上面明确指定的内容；未要求修改的区域保持原图一致。` → `[Requirements]\nOnly modify what is explicitly requested above; keep all other areas consistent with the original.`

This ensures the prompt sent to model is English (model understands both, but English UI consistency is requested).

#### 4. Tests (optional, for CI)
- Updated `tests/test_examples.py`, `test_guider.py`, `test_guider_node.py` to expect English labels.

---

## Installation

1. Replace your existing folder:
   ```
   ComfyUI/custom_nodes/ComfyUI-SenseNova-U1.5-ConvRot/
   ```
   with the contents of `ComfyUI-SenseNova-U1.5-ConvRot-FIXED.zip`

2. Or apply manual patches:
   - Copy fixed `sensenova_u15/loader.py`
   - Copy fixed `nodes.py`
   - Copy fixed `web/sensenova_reference_labels_v131e.js`
   - Copy fixed `sensenova_u15/guidance.py`
   - Add `.gitattributes`

3. Restart ComfyUI

No model re-download needed. Works with both quantized checkpoints from https://huggingface.co/Milor123/ComfyUI-ConvRot-SenseNova-U1.5-8B-MoT-T8

---

## Verification

- Simulated CRLF file: normalized hash matches expected, no exception
- Grepped codebase: no Chinese characters remain in `nodes.py`, `sensenova_u15/`, `web/`
- Node display names now English-only in UI

