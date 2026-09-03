# memory.md — Maintenance Protocols for This SenseNova Fork

Repository: `Spirindzhiukas/Comfyui-SenseNova-U1.5-Wrapper-T8-ANT-fork`
Base: `T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8` (started at **1.3.6**; features synchronized through **1.5.2**)
Self-version: `1.6.1` (see `CHANGELOG.md`)
Quant source: `Milor123/ComfyUI-SenseNova-U1.5-ConvRot` @ `7e1e320` (v1.3.1 base)

This fork is an *amalgamation*: T8mars' clean base + Milor123's ConvRot int8 /
W4A4 / W4A8 support + the tokenizer-CRLF and English-UI fixes. It must stay
mergeable with upstream.

---

## 1. Translation Protocol (MANDATORY for any upstream merge)

- NEVER merge code with Chinese user-facing labels blindly.
- When upstream (T8mars) or another fork adds Chinese:
  1. Keep the original Chinese as the line right above the new value:
     `# Original (upstream T8mars): 参考图 {index} (Image-{index})`
  2. Translate below it.
  3. Node UI (`display_name`, tooltips, defaults): English only.
  4. Structured prompts (`sensenova_u15/guidance.py`): English bracket titles
     (`[Main Edit]`, `[Reference Image Roles]`, `[Must Preserve]`, `[Must Avoid]`,
     `[Requirements]`), with the official Chinese headers kept in the docstring as
     the quality reference. The model follows both; re-test a Chinese prompt after
     any wording change.
- Use a translator or ask the user when unsure — do not guess a translation.
- Files that currently carry translated strings: `nodes.py`,
  `sensenova_u15/guidance.py`, `web/sensenova_reference_labels_v131e.js`,
  `examples/edit_workflow.json`, `examples/multi_reference_edit_workflow.json`,
  `examples/sft_edit_workflow.json`. `examples/t2i_8step_workflow.json` keeps its
  Chinese prompt **on purpose** (it demos rendering Chinese text in the image).
- README layout differs from upstream on purpose: upstream keeps the Chinese
  document at `README.md` and English at `README_EN.md`; **this fork makes
  English the default** (`README.md`) and moved the Chinese to `README_CN.md`.
  When upstream edits its `README_EN.md`, apply those changes here to `README.md`
  and translate them into `README_CN.md`; when it edits its `README.md`, apply
  them to `README_CN.md` only.

## 2. Evaluation Against Our Own Changes

Before merging any upstream commit, diff it against these invariants:

- `sensenova_u15/loader.py::_validate_tokenizer_assets` — the CRLF-tolerant
  digest check (`_tokenizer_digest_kind`) must survive. If upstream changes
  tokenizer validation, re-apply tolerance; the "asset missing" error stays fatal.
- `sensenova_u15/loader.py::load_sensenova_model` — the **quantization wiring**
  must stay: `detect_quant_config(state_dict)` → `model_config.quant_config`, and
  the manual-cast call that passes `None` as the weight dtype for quantized files.
  ComfyUI only consumes `*.comfy_quant` / `*.weight_scale` under
  `comfy.ops.mixed_precision_ops`, which `pick_operations` selects from
  `quant_config`. Losing it is silent and catastrophic: the packed int8 payload
  becomes the weight and the image is a regular checkerboard. Keep
  `_validate_quantized_weights_loaded()` (post-load `QuantizedTensor` check) as the
  tripwire, and keep `comfy.utils.convert_old_quants` for legacy layouts.
  Provenance: reported by the fork owner on 2026-08-27 with Milor123's
  `SenseNova-U1.5-8B-MoT-T8-int8-convrot-tagged.safetensors`.
- `sensenova_u15/quant_bridge.py::kitchen_honours_int8_convrot` — the ops decision
  must stay a **measurement**. Never downgrade it to a version or signature check:
  comfy-kitchen 0.2.28 and 0.2.31 both *accept* `convrot` on `int8_linear`, but only
  0.2.31 applies it, and an ignored flag is again a silent wrong-basis image. Keep
  the "unmeasurable -> use the bridge" direction (validated) and keep
  `_validate_quantized_weights_loaded()` as the backstop.
- `sensenova_u15/model.py` — the pure-PyTorch split-half RoPE (1.3.4 fix) must
  stay. `grep -n "quant_ops\|apply_rope_split_half" sensenova_u15/model.py` must
  stay empty; if upstream re-introduces the comfy-kitchen kernel, re-apply the
  fix and keep the Blackwell comment. The 1.3.7 prefix attention fix must also
  stay: `forward_prefix` casts `attention_mask` to `query.dtype` before calling
  `optimized_attention`. Keep passing this fork's `transformer_options` into
  `_project` while applying future upstream edits.
- `sensenova_u15/quant_bridge.py`, `sensenova_u15/qt_guards.py` — ours (ported
  from Milor123). If upstream grows its own quant path, compare and keep the
  ConvRot-aware one only where upstream is weaker (see §3).
- `nodes.py`, `guidance.py`, `web/*.js`, `examples/*.json` — English UI must stay;
  re-translate any newly added Chinese.
- `checkpoint_contract.json` — upstream-owned. Never edit by hand; regenerate with
  `tools/build_checkpoint_contract.py` and update `CHECKPOINT_CONTRACT_SHA256`.
- Always read `git diff` after a merge and re-run §7.
- `.gitattributes` declares `text eol=lf`, so every file this fork edits gets
  normalised to LF when it is staged (this is what kills the CRLF class of
  bugs). Unedited upstream files keep their own blobs; do not run
  `git add --renormalize .` repo-wide — it would add whitespace-only noise to
  every future upstream merge.

## 3. Compatibility Preservation

- `checkpoint_contract.json` stays the primary contract for bf16 (with
  `file_size`, `final_legacy`, cross-platform digests). Do not delete it.
- Quantization must stay optional and additive:
  - detection is `*.comfy_quant` keys → no such key means the code path is
    byte-identical to upstream (this is what makes future merges cheap);
  - the file-size check only runs for non-quantized checkpoints;
  - `quant_bridge` is installed only when the running ComfyUI/comfy-kitchen does
    not handle convrot itself (`core_supports_convrot`), so a modern core keeps
    its own GPU kernels;
  - `qt_guards` installs for any quantized load (Milor123 installed it at package
    import; we do it from `get_model`, so bf16 streaming keeps its hot path clean
    and 4-bit weights stay intact on hardware that requests a manual cast).
- Env switches (all documented in `README.md`, the English default):
  `SENSENOVA_NO_QUANT=1` (reject quant files), `SENSENOVA_NO_BRIDGE=1` (never install
  custom ops), `SENSENOVA_FORCE_BRIDGE=1` (always install, to reproduce the reference
  numbers of the original ConvRot fork), `SENSENOVA_NO_QT_GUARDS=1` (no cast guards),
  `SENSENOVA_NO_CONVROT_PROBE=1` (skip the kernel measurement, always use the bridge).

## 4. Testing Protocol

After any merge or packaging change:

```bash
# 1. no Chinese left in user-visible strings. The only allowed hits are the
#    "Original (upstream T8mars)" comments and the reference block in the
#    guidance.py docstring, so grep for the *code* patterns, not the characters:
grep -rn 'display_name=f"参考\|default="参考\|default="保持\|default="不要\|return `参考图' \
    --include="*.py" --include="*.js" sensenova_u15 nodes.py web   # -> empty
# 2. RoPE stays pure torch (Blackwell-safe)
grep -n "quant_ops\|apply_rope_split_half" sensenova_u15/model.py   # -> empty
# 3. bf16 contract untouched
python - <<'PY'
import hashlib, pathlib, re
root = pathlib.Path(".")
src = (root / "sensenova_u15/loader.py").read_text()
pin = re.search(r'CHECKPOINT_CONTRACT_SHA256 = "([0-9a-f]{64})"', src).group(1)
raw = (root / "sensenova_u15/checkpoint_contract.json").read_bytes()
assert hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() == pin
for m in re.finditer(r'"([^"]+)": "([0-9a-f]{64})"', src):
    path = root / "sensenova_u15/tokenizer" / m.group(1)
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == m.group(2) or \
           hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest() == m.group(2), m.group(1)
print("digest pins ok")
PY
# 4. the real suite (needs ComfyUI; the node must live in ComfyUI/custom_nodes)
python -m compileall -q . && ruff check .
python -m unittest discover -s tests -t .
```

Windows/CRLF regression to re-check by hand if `loader.py` changes: copy
`sensenova_u15/tokenizer/*` to a directory with CRLF endings, point
`loader.TOKENIZER_DIR` at it and confirm `_validate_tokenizer_assets()` prints a
note and returns.

Quantized checkpoints are validated **header-only** in tests — a fake contract and
fake slices, never a 17 GB file.

Field status: verified working on 2026-08-27 (maintainer report) for the
ConvRot quantized checkpoints in both text-to-image and reference-editing
workflows, after `1.4.1` (quantization wiring) and `1.4.2` (measured capability
probe). Keep the checks below as the regression tripwires — the failure mode is
silent bad images, not a crash.

A real quantized load must print these console lines (missing them is the
checkerboard bug):

```text
Found quantization metadata version 1        <- ComfyUI core, from quant_config
Using mixed precision operations             <- ComfyUI core, pick_operations
[SenseNova-U1.5] quantized checkpoint detected (...); loading with mixed-precision quantization ops.
[sensenova-u15] quantized weights: ComfyUI native mixed-precision operations | SenseNova ConvRot operations
```

`comfy-kitchen` note: its INT8 kernel only honours `convrot` from 0.2.31 (0.2.28
accepts and ignores the kwargs). `core_supports_convrot()` probes the signature and
falls back to our bridge, which reproduces Milor123's validated behaviour.

## 5. Versioning

- This fork keeps its own monotonic release line after the original feature drop.
  Do not downgrade it to a later upstream version; increment the fork version and
  document which upstream fixes were synchronized.
- `tests/test_metadata.py` pins the version string; update it together with
  `pyproject.toml`.
- `CHANGELOG.md` keeps its Chinese body and adds an English block per fork release.
- Keep `node_list.json` in sync with the V3 node ids (ComfyUI-Manager uses it).

## 6. Provenance

| Change | Where | Came from |
| --- | --- | --- |
| CRLF-tolerant tokenizer digest | `sensenova_u15/loader.py::_tokenizer_digest_kind`, `_validate_tokenizer_assets` | this fork, `docs/FIX_REPORT.md`; verified with `sha256sum` LF `6497591f…` vs CRLF `9a7324…` |
| `.gitattributes` LF pin | `.gitattributes` | this fork (upstream had one line for the contract) |
| English labels / defaults | `nodes.py::_reference_image_inputs`, `SenseNovaStructuredEditPrompt` | this fork |
| English frontend label | `web/sensenova_reference_labels_v131e.js::referenceLabel` | this fork (this JS overrides the Python label at runtime) |
| English prompt sections | `sensenova_u15/guidance.py::build_structured_edit_prompt` | this fork; Chinese kept in the docstring |
| Pure-PyTorch split-half RoPE + 6D vision rotation | `sensenova_u15/model.py::_apply_llm_rope`, `_apply_interleaved_rope` | upstream `T8mars` commit `73657001` (1.3.4); verified, not re-applied |
| Prefix attention mask cast to query dtype | `sensenova_u15/model.py::Attention.forward_prefix` | upstream T8mars PR #5 / commit `8f322794` (released in 1.3.7) |
| Upstream 1.3.7 sync audit | `docs/UPSTREAM_SYNC_1.3.7.md` | this fork, 2026-08-31; covers upstream PRs #5 and #6 and retained invariants |
| Upstream 1.5.2 sync audit | `docs/UPSTREAM_SYNC_1.5.2.md` | this fork, 2026-09-03; covers upstream PRs #7–#12, GGUF, thinking/interleave and retained invariants |
| Quant header contract | `sensenova_u15/loader.py` (`QUANT_*`, `_read_quant_formats`, `_quant_checkpoint_contract`, `_validate_quant_header`) | adapted from Milor123 `sensenova_u15/loader.py` + `checkpoint_contract.py`, rebuilt on T8's JSON contract |
| Quant ops wiring (`quant_config`, post-load invariant) | `sensenova_u15/loader.py::detect_quant_config`, `_validate_quantized_weights_loaded` | this fork, 2026-08-27, after the int8 checkerboard report; mirrors `comfy.sd.load_diffusion_model` + `comfy.model_detection` |
| INT8 convrot capability probe (measured, not versioned) | `sensenova_u15/quant_bridge.py::kitchen_honours_int8_convrot` | this fork, 2026-08-27; reference maths cross-checked against comfy-kitchen v0.2.31 `backends/cuda/__init__.py::int8_linear` + `backends/eager/quantization.py` |
| Injector `--variant auto` (source-revision aware tagging) | `tools/inject_sensenova_metadata.py::detect_variant` | this fork, 2026-08-27: Milor123's published int8/int4 files carry the **legacy** revision `1f6ec604`, so re-tagging them as `final` would break the non-quantized dtype checks |
| Quant dtype profile cross-check | `sensenova_u15/loader.py::_quant_checkpoint_contract` reuses `checkpoint_contract.json[profile]` | verified 2026-08-27 to equal Milor123's `_storage_dtype` rule on all 1116 tensors for `final_legacy` (0 differences), and to differ on 519 keys for `final` — which is the real all-BF16 revision, as intended |
| ConvRot Linear forwards | `sensenova_u15/quant_bridge.py` | Milor123 @ `7e1e320` (`quant_bridge.py`), plus our `quant_bridge_needed` / `core_supports_convrot` gate |
| Quantized-tensor cast guards | `sensenova_u15/qt_guards.py` | Milor123 @ `7e1e320` (`qt_guards.py`), relocated + defensive lookups |
| ops hook for quantized loads | `sensenova_u15/model_config.py::get_model` | Milor123's `model_config.py` hook, made capability-aware |
| Converters | `tools/convert_sensenova_int4_convrot.py`, `tools/make_hybrid_ladder.py` | Milor123 @ `7e1e320` (hard-coded Windows paths removed) |
| Metadata tagger | `tools/inject_sensenova_metadata.py` | Milor123's tool, re-pinned to the 1.3.6 revisions (`final`, `final_legacy`, `sft`) |
| Tests | `tests/test_fork_quant_checkpoint.py`, `tests/test_fork_tokenizer_assets.py` | this fork |

Keep this table updated with commit hashes whenever a fix moves.
