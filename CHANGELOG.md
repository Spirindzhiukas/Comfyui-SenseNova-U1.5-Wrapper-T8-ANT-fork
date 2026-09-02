# Changelog

本文件记录 ComfyUI 节点本身的版本变化。模型权重的下载和说明见 Hugging Face 模型页。

## [1.6.1] - 2026-09-03

- 删除不再使用的三轴 RoPE `transformer_options` 覆盖桥接；模型实现恢复为 T8mars 1.5.2 上游代码，同时保留 CUDA 13 / Blackwell 安全的纯 PyTorch split-half RoPE。
- prefix cache 本身继续保留，用于普通、Thinking 与交错生成的执行期 KV 复用；仅删除已无消费者的 theta override cache identity。
- 删除已完成的一次性集成规划与过时研究材料及其代码/文档引用；量化、GGUF、CRLF 与英文 UI 功能不变。

### English

- Removed the unused per-axis RoPE `transformer_options` override bridge. The model implementation now matches T8mars 1.5.2 while retaining its CUDA 13/Blackwell-safe pure-PyTorch split-half RoPE.
- Kept the execution-local prefix cache for normal, thinking and interleaved generation; only the unconsumed theta-override cache identity was removed.
- Removed completed one-time integration plans, obsolete research material and related code/documentation references. Quantization, GGUF, CRLF and English-UI behavior are unchanged.

## [1.6.0] - 2026-09-03

- 同步 T8mars 上游 `1.3.8`～`1.5.2`（PR #7～#12）：采用已合并 ComfyUI core PR #15922 的原生工作流节点，新增严格校验的 Q2_K / Q3_K_M / Q5_K_M / Q6_K / Q8_0 GGUF Loader、Thinking 生图、文本/图像交错生成、KV 回填与顺序化预览，并包含 ComfyUI 0.33 预览兼容及编号图片定位修复。
- 保留本分支的 ConvRot INT8/W4A4/W4A8 检测、严格派生 contract、量化 ops/guard、实测 comfy-kitchen 能力回退、CRLF tokenizer 容错、全英文 UI 与工作流。
- 将新的 `language_model.lm_head.weight` 模型结构与 Thinking/Interleave 路径作为上游基座，量化操作只在存在 `*.comfy_quant` 侧车时叠加；GGUF 和 BF16 不会导入 ConvRot bridge。
- 此版本曾暂时保留三轴 RoPE `transformer_options` 兼容桥接；该桥接已在 1.6.1 清理。
- 完整合并审计见 [`docs/UPSTREAM_SYNC_1.5.2.md`](docs/UPSTREAM_SYNC_1.5.2.md)。

### English

- Synchronized T8mars upstream 1.3.8 through 1.5.2 (PRs #7–#12): native workflows for merged ComfyUI core PR #15922, strictly validated Q2_K/Q3_K_M/Q5_K_M/Q6_K/Q8_0 GGUF loading, thinking generation, interleaved text/image generation with KV feedback, ordered previews, ComfyUI 0.33 preview compatibility, and numbered-image placement fixes.
- Preserved this fork's ConvRot INT8/W4A4/W4A8 detection, derived contracts, quantized operations and guards, measured comfy-kitchen fallback, CRLF-safe tokenizer handling, and English UI/workflows.
- Adopted upstream's LM-head thinking/interleave model as the base and layers quantized operations only when `*.comfy_quant` sidecars exist; BF16 and GGUF do not import the ConvRot bridge.
- This release temporarily retained a three-axis RoPE `transformer_options` compatibility bridge; it was removed in 1.6.1.
- Full merge audit: [`docs/UPSTREAM_SYNC_1.5.2.md`](docs/UPSTREAM_SYNC_1.5.2.md).

## [1.4.3] - 2026-08-31

- 调查并集成 T8mars 上游 PR [#5](https://github.com/T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8/pull/5)：prefix attention mask 在调用 ComfyUI attention 后端前转换为 query dtype，避免 PyTorch SDPA 在 FP32 mask 配合 BF16 Q/K/V 时产生有限但数值错误的输出；新增回归测试。
- 审查上游发布 PR [#6](https://github.com/T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8/pull/6)：保留本分支较新的 `1.4.x` 版本线而不降级为上游 `1.3.7`，本次发布为 `1.4.3`；上游 changelog 内容已合并到本条目。
- 合并时保留本分支的 ConvRot 量化路径、CRLF 容错、英文 UI 和文档布局；完整审计见 [`docs/UPSTREAM_SYNC_1.3.7.md`](docs/UPSTREAM_SYNC_1.3.7.md)。
- 两份 README 增加完整的署名与来源说明：逐文件列出哪些代码来自 T8mars（最初的上游 1.3.6 基座，修复同步至 1.3.7）、
  哪些移植自 `Milor123/ComfyUI-SenseNova-U1.5-ConvRot@7e1e320`、以及本分支改了什么；
  `NOTICE` 同步补充两段上游署名、社区模型仓库与 ConvRot / comfy-kitchen /
  convert_to_quant / ComfyUI 的致谢。
- 模型下载改为“按仓库区分”：bf16 与官方精度权重在 `t8star/SenseNova-U1.5-Comfy`，
  量化 ConvRot 权重在 `Milor123/ComfyUI-ConvRot-SenseNova-U1.5-8B-MoT-T8`；并说明
  8-step LoRA 在两个仓库里字节一致、Milor123 的量化文件按 `final_legacy` 校验、
  W4A4 没有现成文件需自行转换。
- 删除 ComfyUI Registry / ComfyUI-Manager 安装说明（本分支未发布到 Registry），安装改为
  Git 克隆本仓库，徽章、Releases 链接与示例命令一并指向本仓库。
- 模型校验补充 Milor123 两个量化文件的大小与 SHA256；支持矩阵补上“量化 Final”一行。
- `tests/test_metadata.py` 的本地链接检查现在会忽略 `#` 锚点部分，便于跨文档跳转。

### English

- Investigated and integrated T8mars upstream PR [#5](https://github.com/T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8/pull/5): the prefix attention mask is cast to the query dtype before ComfyUI dispatches to an attention backend, preventing finite but numerically incorrect PyTorch SDPA output with an FP32 mask and BF16 Q/K/V. Added a regression test.
- Reviewed upstream release PR [#6](https://github.com/T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8/pull/6): retained this fork's newer `1.4.x` version line instead of downgrading to upstream `1.3.7`, and released this integration as `1.4.3`; the upstream changelog content is represented here.
- Preserved this fork's ConvRot quantization, CRLF tolerance, English UI, and documentation layout during integration. The complete audit is in [`docs/UPSTREAM_SYNC_1.3.7.md`](docs/UPSTREAM_SYNC_1.3.7.md).
- Both READMEs gained a full "Credits and provenance" section: a file-by-file
  breakdown of what comes from T8mars (initial upstream 1.3.6 base, fixes synchronized through 1.3.7), what was ported from
  `Milor123/ComfyUI-SenseNova-U1.5-ConvRot@7e1e320`, and what this fork changed.
  `NOTICE` now carries the same attribution for both upstreams, the community model
  repositories, and ConvRot / comfy-kitchen / convert_to_quant / ComfyUI.
- "Download the models" is now organised per repository: bf16 and official-precision
  checkpoints live in `t8star/SenseNova-U1.5-Comfy`, quantized ConvRot checkpoints in
  `Milor123/ComfyUI-ConvRot-SenseNova-U1.5-8B-MoT-T8`, including the notes that the
  8-step LoRA is byte-identical in both, that Milor123's quants validate as
  `final_legacy`, and that no W4A4 file is published.
- Removed the ComfyUI Registry / ComfyUI-Manager installation instructions (this fork
  is not on the Registry); installation is now a Git clone of this repository, and the
  badge, Releases link and example commands point here too.
- Model verification lists size and SHA256 for Milor123's two quantized files, and the
  support matrix gained a "quantized Final" row.
- `tests/test_metadata.py` now ignores the `#anchor` part when checking local links.

## [1.4.2] - 2026-08-27

- INT8 ConvRot 的算子选择改为**实测**：加载时在目标设备上用小规模确定性权重调用 `comfy_kitchen.int8_linear`，比较"旋转激活"与"不旋转"两种参考结果，据此决定使用 ComfyUI 原生 mixed-precision 还是本节点的 ConvRot 实现；不再依赖版本号或函数签名（0.2.28 与 0.2.31 都接收 `convrot` 参数，但只有后者真正执行）。可用 `SENSENOVA_NO_CONVROT_PROBE=1` 跳过实测。
- 若量化权重最终不是 `QuantizedTensor`，加载直接报错（此前只会安静地产出方格图），并提示改用 `SENSENOVA_FORCE_BRIDGE=1`。
- `tools/inject_sensenova_metadata.py` 新增 `--variant auto`（默认）：沿用文件自身的来源标签。社区 int8/int4 权重是由旧版混合精度 Final 转换而来（`1f6ec604`），必须保持 `final_legacy`，否则未量化张量的 dtype 校验会失败。
- 量化加载现在总会安装 `QuantizedTensor` 转换保护（不再只在 bridge 模式下），因为无 BF16 的硬件会请求手动 dtype 转换。
- 加载日志会打印检测到的量化格式、实测结论与最终所选算子；两份 README 增加"如何确认量化路径已生效"。
- 已由本分支维护者于 2026-08-27 实测确认：两套 ConvRot 量化权重在文生图与参考图编辑工作流中均正常出图。

### English

- The INT8 ConvRot ops decision is now **measured**: at load time the pack calls `comfy_kitchen.int8_linear` on a small deterministic weight and compares the rotated against the unrotated reference before deciding between ComfyUI's native mixed-precision ops and this node's ConvRot forwards. Version numbers and signatures are not trustworthy here — 0.2.28 and 0.2.31 both *accept* `convrot`, only 0.2.31 applies it. `SENSENOVA_NO_CONVROT_PROBE=1` skips the measurement (always bridge).
- Loading now fails loudly if a quantized layer did not end up as a `QuantizedTensor`, instead of silently rendering a checkerboard, and the message points at `SENSENOVA_FORCE_BRIDGE=1`.
- `tools/inject_sensenova_metadata.py` gained `--variant auto` (the default), which keeps the file's own source tags. The community int8/int4 files were converted from the legacy mixed-precision Final (`1f6ec604`) and must stay `final_legacy`, otherwise the non-quantized tensors fail their dtype check.
- The `QuantizedTensor` cast guards now install for every quantized load, not only bridge loads, since hardware without BF16 asks for a manual cast.
- Load logs report the detected formats, the probe verdict and the selected ops; both READMEs document how to confirm the quantized path.
- Verified by the fork maintainer on 2026-08-27: both ConvRot quantized checkpoints generate correctly in text-to-image and in the editing workflows.

## [1.4.1] - 2026-08-27

- 修复量化权重实际未被解包的问题：加载 ConvRot 检查点时现在会像 `comfy.sd.load_diffusion_model` 一样设置 `model_config.quant_config`（并调用 `comfy.utils.convert_old_quants`），否则 ComfyUI 会选择普通 Linear，把 `int8` 打包数据直接当权重使用——推理照常运行，但画面是规则方格噪声，同时控制台出现大量 `unet unexpected: [... weight_scale ... comfy_quant]`。
- 量化模型现在按核心规则忽略存储精度来选择 `manual cast`，并在控制台打印检测到的量化格式与所选算子（原生 mixed-precision 或本分支 ConvRot bridge）。
- 新增加载后校验：若带 `comfy_quant` 的层最终不是 `QuantizedTensor`，直接报错并提示更新 ComfyUI 或使用 `SENSENOVA_FORCE_BRIDGE=1`，不再静默产出坏图。

### English

- Fixed quantized checkpoints silently loading unpacked. Loading a ConvRot file now sets `model_config.quant_config` (and runs `comfy.utils.convert_old_quants`) exactly like `comfy.sd.load_diffusion_model`; without it ComfyUI picks plain `Linear` ops, treats the packed int8 payload as the weight, and inference happily produces a regular checkerboard while the console fills with `unet unexpected: [... weight_scale ... comfy_quant]`.
- The manual-cast decision now follows core's rule of ignoring the stored weight dtype for quantized checkpoints, and the loader logs the detected formats plus which ops were selected (native mixed-precision vs this fork's ConvRot bridge).
- New post-load invariant: if a layer that carries a `comfy_quant` sidecar did not end up as a `QuantizedTensor`, loading fails with guidance (update ComfyUI, or `SENSENOVA_FORCE_BRIDGE=1`) instead of producing broken images.

## [1.4.0] - 2026-08-27（ANT 分支，基线 = 上游 T8mars 1.3.6）

- Windows 换行符兼容：tokenizer 资产摘要现在同时比对原始字节和 CRLF→LF 规范化结果，只有两者都不匹配时打印警告而不再阻止加载；`.gitattributes` 把 `sensenova_u15/tokenizer/*` 以及 `*.py/*.json/*.js/*.txt` 固定为 LF。
- 全英文界面：参考图插槽名、`SenseNova Structured Edit Prompt` 默认值、前端扩展标签、内置工作流示例都改为英文；上游中文原文保留为注释，方便继续合并 T8mars 更新。
- 新增可选 ConvRot 量化支持：`int8_tensorwise`、`convrot_w4a4`、`asym_w4a8_int8`（含按层混合格式与 hybrid 阶梯产物）通过每层 `*.comfy_quant` 侧车键被识别，并按派生自 `checkpoint_contract.json` 的 contract 严格校验；官方 bf16 检查点仍走完全不变的上游 JSON 分支，文件大小校验只对 bf16 生效。
- `sensenova_u15/quant_bridge.py`（移植自 Milor123/ComfyUI-SenseNova-U1.5-ConvRot@7e1e320）只在当前 ComfyUI/comfy-kitchen 不会自行完成 convrot 激活旋转时启用；`sensenova_u15/qt_guards.py` 也只在真正加载量化权重时安装，因此 bf16 用户的行为和性能完全不变。可用 `SENSENOVA_NO_QUANT`、`SENSENOVA_NO_BRIDGE`、`SENSENOVA_FORCE_BRIDGE`、`SENSENOVA_NO_QT_GUARDS` 覆盖。
- 保留上游 1.3.4 的纯 PyTorch RoPE 修复（CUDA 13 / Blackwell 安全）；此版本加入的实验性三轴 theta override 后于 1.6.1 删除。
- 主 README 改为英文：原 `README_EN.md` 内容移到 `README.md`，中文文档改名 `README_CN.md`，仓库与 ComfyUI-Registry 首页默认显示英文。
- 新增量化脚本 `tools/convert_sensenova_int4_convrot.py`、`tools/inject_sensenova_metadata.py`、`tools/make_hybrid_ladder.py`、维护契约 `memory.md`，以及量化检查点和 CRLF 相关测试。

### English

- CRLF-safe tokenizer check: the asset digest now compares both the raw bytes and the LF-normalised bytes, warns instead of aborting on a mismatch, and `.gitattributes` pins the packaged text files to LF, so a Windows clone with `core.autocrlf=true` loads again.
- English UI: reference-image slot labels, the structured edit prompt defaults, the frontend extension label and the shipped examples are English; the upstream Chinese wording stays next to them as a comment so future T8mars merges stay reviewable.
- Optional ConvRot quantization: `int8_tensorwise`, `convrot_w4a4` and `asym_w4a8_int8` checkpoints (including per-layer mixes and the hybrid ladder output) are detected through their per-layer `comfy_quant` sidecars and validated against a contract derived from the bundled `checkpoint_contract.json`. Official bf16 files keep the untouched upstream path, including the file-size check.
- The ported `quant_bridge.py` installs itself only when the running ComfyUI/comfy-kitchen cannot rotate convrot activations itself, and `qt_guards.py` installs only for quantized loads, so bf16 behaviour and speed are unchanged. Override with `SENSENOVA_NO_QUANT`, `SENSENOVA_NO_BRIDGE`, `SENSENOVA_FORCE_BRIDGE` or `SENSENOVA_NO_QT_GUARDS`.
- The upstream 1.3.4 pure-PyTorch RoPE fix is preserved. The experimental three-axis theta override added in this release was later removed in 1.6.1.
- Added quant tooling in `tools/`, the `memory.md` maintenance contract, and tests for quantized headers and CRLF assets.
- **English is now the default README**: `README.md` holds the English documentation (was `README_EN.md`) and the Chinese translation moved to `README_CN.md`, so the repository and ComfyUI-Registry front pages open in English.

## [1.3.6] - 2026-08-27

- 添加官方 ComfyUI-Manager 使用的 `node_list.json`，让 V3 扩展入口注册的全部 8 个节点能被“安装缺失节点”功能可靠识别。
- 增加节点清单与 V3 schema ID 的一致性测试，防止新增或重命名节点时遗漏 Manager 映射。

## [1.3.5] - 2026-08-26

- 支持官方 revision `19bc874e` 的全 BF16 Final 权重及约 35 GB 的新版 ComfyUI 单文件，同时继续严格校验并兼容原有约 50 GB 的混合精度 Final；两者都可使用现有 8-step LoRA。
- `Empty SenseNova Pixel Latent` 新增官方建议的 1:1、16:9、9:16、2:3、3:2 分辨率预设，保留原有自定义宽高和旧工作流输入顺序。
- CI 更新到 ComfyUI v0.34.0，覆盖 Python 3.10～3.14，并加入 Ruff 静态检查。
- 添加 GitHub Bug/Feature Issue 表单与 GitHub Actions Dependabot 更新配置；主分支启用必需 CI、禁止强推和删除保护。

## [1.3.4] - 2026-08-25

- 修复 CUDA 13 / Blackwell 环境启用 `comfy-kitchen` CUDA 后端时，split-half RoPE 可能返回有限但错误的数值，导致生成结果严重偏色、过饱和和结构异常的问题；语言层 RoPE 现在固定使用与官方一致的 PyTorch 参考公式。
- 视觉 RoPE 改用 `comfy-kitchen` 支持的标准 4D 输入和 6D rotation 布局，启用 `--enable-triton-backend` 时不再因 3D tensor 解包失败。
- 新增后端隔离和 accelerated-backend tensor rank 回归测试。

## [1.3.3] - 2026-08-24

- 修复部分环境加载官方 Final/SFT 单文件时，把 `timestep_embedder` 和 `noise_scale_embedder` 错报为多余键的问题（#1、#2）。
- 节点现在随包携带从正式 Final/SFT 转换清单生成的固定 1116 tensor contract，不再根据当前 PyTorch/ComfyUI 环境临时推导权重结构。
- 加载前检查模型 metadata、文件大小、全部 tensor 名称、shape 和各版本 dtype；错误信息会显示实际模型与 loader 路径，便于发现旧节点或重复安装。
- CI 不再因缺少本地大模型 manifest 而跳过结构测试，并新增 Python 3.13 + ComfyUI 0.33.1 组合。

## [1.3.2] - 2026-08-24

- CI 同时验证最低支持的 ComfyUI 0.31 和当前稳定版 ComfyUI 0.33.1。
- Registry 发布后自动写入对应版本的更新说明。
- 补全 Registry、模型下载、问题反馈和发布记录入口。
- 同步 Hugging Face 模卡中的仓库地址、批量生成和多参考图说明。

本版本不修改模型加载、采样或图像生成逻辑，推理结果与 1.3.1 一致。

## [1.3.1] - 2026-08-22

- 修复普通参考图节点的插槽命名和默认工作流连线。
- 旧工作流导入时自动迁移历史插槽名称。
- 批量生成时复用文本和参考图 prefix，降低重复计算。

## [1.3.0] - 2026-08-22

- 支持一次生成 1～16 张不同结果。
- 新增 CFG Norm、CFG 生效区间和编辑提示词整理节点。
- 新增 1～10 张参考图的高级节点。
