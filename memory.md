# memory.md — Maintenance Protocols for This SenseNova Fork

Repository: `Spirindzhiukas/Comfyui-SenseNova-U1.5-Wrapper-T8-ANT-fork`
Base: `T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8` **1.3.6** (upstream RoPE fix from 1.3.4 included)
Self-version: `1.4.0` (see `CHANGELOG.md`)
Contract this file implements: `docs/THE_TASK.md`
Quant source: `Milor123/ComfyUI-SenseNova-U1.5-ConvRot` @ `7e1e320` (v1.3.1 base)

This fork is an *amalgamation*: T8mars' clean base + Milor123's ConvRot int8 /
W4A4 / W4A8 support + the tokenizer-CRLF and English-UI fixes + the MERD / RoPE
Lab research. It must stay mergeable with upstream.

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
- `sensenova_u15/model.py` — the pure-PyTorch split-half RoPE (1.3.4 fix) must
  stay. `grep -n "quant_ops\|apply_rope_split_half" sensenova_u15/model.py` must
  stay empty; if upstream re-introduces the comfy-kitchen kernel, re-apply the
  fix and keep the Blackwell comment.
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
  - `qt_guards` installs only for quantized loads (Milor123 installed it at
    package import; we do not, so bf16 streaming keeps its hot path clean).
- Env switches (all documented in `README.md`, the English default):
  `SENSENOVA_NO_QUANT=1` (reject quant files), `SENSENOVA_NO_BRIDGE=1`
  (never install custom ops), `SENSENOVA_FORCE_BRIDGE=1` (always install, to
  reproduce the reference numbers of the original ConvRot fork),
  `SENSENOVA_NO_QT_GUARDS=1` (no cast guards).
- No hard dependency on `comfy-kitchen`: every quant import is guarded and
  optional; the bf16 path must import fine on a plain ComfyUI.
- Deliberate deviations from `docs/THE_TASK.md` (recorded so nobody "restores"
  them by accident):
  1. `qt_guards.py` lives in `sensenova_u15/` (not the repo root) so it also
     imports when `sensenova_u15` is loaded standalone by the unit tests, and it
     is installed from `SenseNovaModelConfig.get_model` instead of `__init__.py`.
  2. No duplicate frozen tensor table: the quant contract is *derived* from
     `checkpoint_contract.json`, so Milor123's `sensenova_u15/checkpoint_contract.py`
     was not ported. Upstream contract revisions then cover quant for free.
  3. Upstream moved to 1.3.6 while this contract targeted 1.3.4; the RoPE section
     of the task was therefore verified, not re-applied.
  4. `docs/ROPE_AND_EMBEDDING_NODES_PROPOSAL.md` referenced by the task was not
     present in the source folder; only the research doc exists
     (`docs/SenseNova_MERD_and_RoPE_Lab_integration_research.md`), and
     `docs/rope_lab_integration.md` distils it.

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

## 5. Versioning

- Upstream version wins the first three components; this fork bumps the minor on
  a feature drop (`1.3.6` → `1.4.0`) and says which upstream it is based on.
- `tests/test_metadata.py` pins the version string; update it together with
  `pyproject.toml`.
- `CHANGELOG.md` keeps its Chinese body and adds an English block per fork release.
- Keep `node_list.json` in sync with the V3 node ids (ComfyUI-Manager uses it).

## 6. RoPE / RoPE Lab Integration Notes

- SenseNova-U1.5 uses 3D M-RoPE: `axes_dim [64, 32, 32]` (pairs 32/16/16),
  `per_axis_theta [5e6, 1e4, 1e4]`, axis identity `t,h,w`, spatial axes `[1, 2]`,
  temporal axis `0`, concat order `t_h_w`, base patch grid `64 x 64` at
  2048 px (`MERGED_PATCH_SIZE = 32`), split-half `cos/sin` for the LLM path and
  interleaved 4D-value/6D-rotation `apply_rope1` for vision;
  `uses_qk_norm = true`, block-causal prefix mask `true`.
- `dype_output_format` is `mrope_interleaved`, which makes RoPE Lab abort
  (Ideogram4 branch). The patching strategy must be a new `sensenova_mot_hook`
  (`core/method_hook.py`), not `embedder_replacement`.
- The fork already removed the main obstacle: the three bases are read through
  `resolve_rope_thetas(transformer_options)` from
  `sensenova_rope_theta_t / _hw / _vision`, so a Lab wrapper only has to set
  `transformer_options` — no monkey-patching of `model.py`.
- Never break `transformer_options["sensenova_prefix_cache"]`: dynamic methods
  change thetas per timestep, and the theta tuple is part of the cache key, so a
  changed scale correctly misses the cached prefix KV. Do not "optimise" that key.
- Frozen imports a hook must handle: `optimized_attention`, `pad_to_patch_size`,
  `apply_rope1` (they are imported into `sensenova_u15.model`'s namespace, so a
  `sys.modules` walk or patching `sensenova_u15.model.<name>` is required).
- See `docs/rope_lab_integration.md` for the MERD fields, dype math and the
  Phase 0-6 roadmap.

## 7. Provenance

| Change | Where | Came from |
| --- | --- | --- |
| CRLF-tolerant tokenizer digest | `sensenova_u15/loader.py::_tokenizer_digest_kind`, `_validate_tokenizer_assets` | this fork, `docs/FIX_REPORT.md`; verified with `sha256sum` LF `6497591f…` vs CRLF `9a7324…` |
| `.gitattributes` LF pin | `.gitattributes` | this fork (upstream had one line for the contract) |
| English labels / defaults | `nodes.py::_reference_image_inputs`, `SenseNovaStructuredEditPrompt` | this fork, task §2.2 |
| English frontend label | `web/sensenova_reference_labels_v131e.js::referenceLabel` | this fork (this JS overrides the Python label at runtime) |
| English prompt sections | `sensenova_u15/guidance.py::build_structured_edit_prompt` | this fork; Chinese kept in the docstring |
| Pure-PyTorch split-half RoPE + 6D vision rotation | `sensenova_u15/model.py::_apply_llm_rope`, `_apply_interleaved_rope` | upstream `T8mars` commit `73657001` (1.3.4); verified, not re-applied |
| Quant header contract | `sensenova_u15/loader.py` (`QUANT_*`, `_read_quant_formats`, `_quant_checkpoint_contract`, `_validate_quant_header`) | adapted from Milor123 `sensenova_u15/loader.py` + `checkpoint_contract.py`, rebuilt on T8's JSON contract |
| ConvRot Linear forwards | `sensenova_u15/quant_bridge.py` | Milor123 @ `7e1e320` (`quant_bridge.py`), plus our `quant_bridge_needed` / `core_supports_convrot` gate |
| Quantized-tensor cast guards | `sensenova_u15/qt_guards.py` | Milor123 @ `7e1e320` (`qt_guards.py`), relocated + defensive lookups |
| ops hook for quantized loads | `sensenova_u15/model_config.py::get_model` | Milor123's `model_config.py` hook, made capability-aware |
| Converters | `tools/convert_sensenova_int4_convrot.py`, `tools/make_hybrid_ladder.py` | Milor123 @ `7e1e320` (hard-coded Windows paths removed) |
| Metadata tagger | `tools/inject_sensenova_metadata.py` | Milor123's tool, re-pinned to the 1.3.6 revisions (`final`, `final_legacy`, `sft`) |
| RoPE theta plumbing | `sensenova_u15/model.py::resolve_rope_thetas` + call sites | this fork, from `docs/SenseNova_MERD_and_RoPE_Lab_integration_research.md` §9.3 |
| Tests | `tests/test_fork_quant_checkpoint.py`, `tests/test_fork_tokenizer_assets.py`, `tests/test_fork_rope_theta.py` | this fork |

Keep this table updated with commit hashes whenever a fix moves.
