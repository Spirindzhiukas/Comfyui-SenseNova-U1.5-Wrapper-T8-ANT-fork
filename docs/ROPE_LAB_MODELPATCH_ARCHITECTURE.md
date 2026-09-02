# SenseNova RoPE Lab — loader-agnostic MODEL patch architecture

## Status and ownership

This is a **handoff specification** for the separately maintained ANT RoPE Lab project. It does not add a new node to this SenseNova wrapper.

The current fork temporarily carries three SenseNova-specific `transformer_options` keys. The goal is to move RoPE policy and compatibility adapters into RoPE Lab so the same patch node works with:

1. T8mars' custom SenseNova wrapper, including its thinking/interleave implementation;
2. this ConvRot/GGUF maintenance fork during migration;
3. native ComfyUI SenseNova from merged core PR #15922;
4. native thinking/interleave after ComfyUI PR #16032 or equivalent lands.

Once the external implementation passes the compatibility matrix below, this repository can remove its direct RoPE changes without losing the feature.

## Proposed user-facing node

```text
MODEL ──> ANT SenseNova RoPE Patch ──> MODEL
```

Suggested inputs:

| Input | Purpose |
| --- | --- |
| `model` | Any MODEL produced by the custom BF16/ConvRot/GGUF loader or core `CheckpointLoaderSimple` |
| `method` | `none`, fixed-theta, NTK, YaRN, SEGA, or another RoPE Lab strategy |
| `time_scale` | Policy input for the SenseNova temporal/token axis |
| `spatial_scale` | Shared or separate height/width policy input |
| `vision_scale` | Policy input for the vision embedding RoPE |
| method-specific advanced inputs | Context length, original length, alpha/beta, interpolation factor, etc. |

The node must clone the incoming ModelPatcher and must not mutate the loader's original MODEL.

## Why a normal model patch is preferable

Loader-specific source edits are fragile:

- T8mars' wrapper and ComfyUI core use different module paths.
- The original 1.3.x wrapper, 1.5.x thinking wrapper and core implementation have different prefix/decode structures.
- BF16, ConvRot and GGUF loaders should not each own a RoPE implementation.
- Core will eventually make the duplicated custom model implementation unnecessary.

A MODEL-to-MODEL patch keeps policy in RoPE Lab and treats model loaders as interchangeable providers.

## Runtime design

### 1. Adapter registry

Resolve an adapter from the loaded diffusion model class, not from a node id or loader name. At minimum recognize:

| Implementation | Expected class/module |
| --- | --- |
| T8mars/custom fork | `sensenova_u15.model.SenseNovaU15` (package prefix may vary) |
| Native ComfyUI | `comfy.ldm.sensenova.model.SenseNovaU15` |

Do not require an exact repository folder name. Inspect the actual diffusion-model instance reachable from `ModelPatcher` and use capability checks for helper functions and methods.

Unknown model classes must fail at patch-node execution with a clear “unsupported model” message, not silently return an unmodified model when a non-`none` method was requested.

### 2. Scoped execution context

Use `contextvars.ContextVar` for the active immutable RoPE configuration. Never store the active policy in a process-global mutable dictionary.

Register wrappers once per Python module, but make them no-ops unless the context variable is active. This allows patched and unpatched SenseNova models to execute concurrently without leaking settings.

The safest scope is an `OUTER_SAMPLE` ModelPatcher wrapper because native core may precompute text/reference prefixes in `BaseModel.extra_conds` before the diffusion model's ordinary forward wrapper runs. The outer wrapper should:

1. set the context variable;
2. establish the cache namespace for the effective policy;
3. call the wrapped sampler;
4. reset the context in `finally`, including cancellation and exceptions.

Thinking/interleave nodes that call model helper methods outside the standard sampler path need an equivalent scoped entry point. Prefer a reusable context manager exposed by the adapter rather than duplicating global state logic.

### 3. Helper interception

Both current implementations import or define RoPE helpers in their module namespace. Patch the module-local call sites that are actually used:

- `_apply_llm_rope(query, key, positions, theta)` or its future equivalent;
- `_apply_interleaved_rope(value, positions, theta)`;
- any prepared-RoPE helper introduced by core, such as `_prepare_llm_rope` / `_prepare_mrope`.

The wrapper should obtain the active method from the context, derive an effective frequency schedule from axis identity, positions, head dimension and original theta, and then execute the original implementation or a RoPE Lab reference implementation.

Do not patch `comfy.quant_ops` globally. SenseNova's split-half language RoPE must retain the pure-PyTorch correctness behavior required on CUDA 13/Blackwell. Quantized linear operations are unrelated to RoPE policy.

### 4. Axis identification

SenseNova U1.5 uses 3D M-RoPE:

- concatenation order: time/token, height, width;
- axis dimensions: 64, 32, 32 per attention head;
- official theta: `5_000_000`, `10_000`, `10_000`;
- vision embedding: interleaved x/y RoPE with theta `10_000`;
- merged image patch: 32 px.

Prefer explicit call-site metadata from an adapter. As a compatibility fallback, map the official theta and dimension/call order only after confirming the model is SenseNova. Never infer axes globally from theta alone because other models may use the same constants.

### 5. Prefix cache isolation

Changing RoPE invalidates prefix K/V. This is a correctness requirement, not an optimization detail.

Create a stable policy fingerprint from:

- adapter/version id;
- method name;
- all method parameters;
- effective time/spatial/vision schedules;
- any timestep- or resolution-dependent inputs.

For custom wrappers using `transformer_options["sensenova_prefix_cache"]`, provide a namespaced mapping or per-fingerprint cache bucket. For a method whose effective frequencies change by denoising timestep, select the bucket before every model call; never reuse prefix K/V from a different effective schedule.

For native core precomputed conditions, either include the fingerprint in core's condition/cache identity through the patcher API or force recomputation under the active outer-sample context. If neither can be guaranteed, reject dynamic methods with a clear error rather than returning numerically inconsistent output.

### 6. Backward compatibility bridge

During migration, support the existing keys as an adapter input format:

```text
sensenova_rope_theta_t
sensenova_rope_theta_hw
sensenova_rope_theta_vision
```

The RoPE Lab node may set these keys when it detects this fork, but must not depend on them for native core. The long-term implementation lives in the adapter/helper interception layer.

## Recommended package layout

```text
rope_lab/
├── nodes/sensenova_rope_patch.py
├── adapters/
│   ├── sensenova_common.py
│   ├── sensenova_t8.py
│   └── sensenova_comfy_core.py
├── runtime/
│   ├── context.py
│   ├── module_hooks.py
│   └── cache_namespace.py
└── tests/
    ├── test_sensenova_adapter_t8.py
    ├── test_sensenova_adapter_core.py
    ├── test_sensenova_cache_isolation.py
    └── test_sensenova_scope.py
```

Keep method mathematics separate from compatibility adapters. Adapters describe where and when to apply a schedule; method modules calculate the schedule.

## Acceptance criteria for the RoPE Lab session

### Functional

- One patch node accepts MODEL from custom BF16, ConvRot and GGUF loaders.
- The same node accepts MODEL from native core `CheckpointLoaderSimple`.
- `method=none` is byte-identical to an unpatched run.
- Explicit official thetas are byte-identical to defaults.
- Changing each axis independently changes only its intended RoPE calls.
- Thinking token decode and interleave image feedback remain operational.
- An unpatched model executed after a patched model observes official defaults.

### Cache and concurrency

- Two policies in one process never share prefix K/V.
- Re-running one policy may reuse only its own entries.
- Cancellation resets context and leaves no active global policy.
- Concurrent executions in separate threads/tasks remain isolated.
- Dynamic policies either recompute/namespace correctly per effective schedule or fail explicitly.

### Compatibility matrix

Test against:

1. T8mars 1.5.2 custom wrapper;
2. this fork 1.6.0 with BF16;
3. this fork 1.6.0 with at least one ConvRot checkpoint;
4. this fork 1.6.0 with Q3_K_M or Q6_K GGUF;
5. ComfyUI commit containing merge `3216c62e` (PR #15922);
6. ComfyUI PR #16032 head or its eventual merge commit.

### Numerical safety

- Split-half language RoPE matches the PyTorch reference at official settings.
- No call to `apply_rope_split_half` is introduced for the language path.
- Tests include BF16 queries and multiple nonzero positions.
- Vision tests cover canonical 4D input / 6D rotation backend ranks.

## Migration plan for this repository

1. **Current release:** retain direct option-key plumbing as a compatibility bridge.
2. **RoPE Lab implementation:** build and validate the model patch node against the matrix above.
3. **Dual-path release:** prefer external adapter when present; retain direct keys with a deprecation notice.
4. **Removal release:** delete `resolve_rope_thetas`, option threading and fork-specific cache tuple entries from this node only after external tests pass for custom and core loaders.
5. Keep this document or a link to the RoPE Lab implementation so users can migrate workflows.

## Explicit non-goals

- Do not move ConvRot loading or GGUF dequantization into RoPE Lab.
- Do not make RoPE Lab responsible for checkpoint validation.
- Do not replace ComfyUI's model loader.
- Do not permanently fork ComfyUI core model files.
- Do not rely on repository names, node display labels or filesystem paths to identify the model.
