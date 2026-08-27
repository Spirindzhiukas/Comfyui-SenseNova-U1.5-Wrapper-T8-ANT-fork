# Changelog

本文件记录 ComfyUI 节点本身的版本变化。模型权重的下载和说明见 Hugging Face 模型页。

## [1.4.0] - 2026-08-27（ANT 分支，基线 = 上游 T8mars 1.3.6）

- Windows 换行符兼容：tokenizer 资产摘要现在同时比对原始字节和 CRLF→LF 规范化结果，只有两者都不匹配时打印警告而不再阻止加载；`.gitattributes` 把 `sensenova_u15/tokenizer/*` 以及 `*.py/*.json/*.js/*.txt` 固定为 LF。
- 全英文界面：参考图插槽名、`SenseNova Structured Edit Prompt` 默认值、前端扩展标签、内置工作流示例都改为英文；上游中文原文保留为注释，方便继续合并 T8mars 更新。
- 新增可选 ConvRot 量化支持：`int8_tensorwise`、`convrot_w4a4`、`asym_w4a8_int8`（含按层混合格式与 hybrid 阶梯产物）通过每层 `*.comfy_quant` 侧车键被识别，并按派生自 `checkpoint_contract.json` 的 contract 严格校验；官方 bf16 检查点仍走完全不变的上游 JSON 分支，文件大小校验只对 bf16 生效。
- `sensenova_u15/quant_bridge.py`（移植自 Milor123/ComfyUI-SenseNova-U1.5-ConvRot@7e1e320）只在当前 ComfyUI/comfy-kitchen 不会自行完成 convrot 激活旋转时启用；`sensenova_u15/qt_guards.py` 也只在真正加载量化权重时安装，因此 bf16 用户的行为和性能完全不变。可用 `SENSENOVA_NO_QUANT`、`SENSENOVA_NO_BRIDGE`、`SENSENOVA_FORCE_BRIDGE`、`SENSENOVA_NO_QT_GUARDS` 覆盖。
- 保留上游 1.3.4 的纯 PyTorch RoPE 修复（CUDA 13 / Blackwell 安全），并把 t / hw / vision 三条 RoPE 基频改为可从 `transformer_options` 读取，prefix cache key 也包含基频，为 ANT RoPE_Lab 的动态缩放留出接口。
- 主 README 改为英文：原 `README_EN.md` 内容移到 `README.md`，中文文档改名 `README_CN.md`，仓库与 ComfyUI-Registry 首页默认显示英文。
- 新增量化脚本 `tools/convert_sensenova_int4_convrot.py`、`tools/inject_sensenova_metadata.py`、`tools/make_hybrid_ladder.py`，维护契约 `memory.md` 与设计文档 `docs/rope_lab_integration.md`，以及量化检查点、CRLF、RoPE 基频相关测试。

### English

- CRLF-safe tokenizer check: the asset digest now compares both the raw bytes and the LF-normalised bytes, warns instead of aborting on a mismatch, and `.gitattributes` pins the packaged text files to LF, so a Windows clone with `core.autocrlf=true` loads again.
- English UI: reference-image slot labels, the structured edit prompt defaults, the frontend extension label and the shipped examples are English; the upstream Chinese wording stays next to them as a comment so future T8mars merges stay reviewable.
- Optional ConvRot quantization: `int8_tensorwise`, `convrot_w4a4` and `asym_w4a8_int8` checkpoints (including per-layer mixes and the hybrid ladder output) are detected through their per-layer `comfy_quant` sidecars and validated against a contract derived from the bundled `checkpoint_contract.json`. Official bf16 files keep the untouched upstream path, including the file-size check.
- The ported `quant_bridge.py` installs itself only when the running ComfyUI/comfy-kitchen cannot rotate convrot activations itself, and `qt_guards.py` installs only for quantized loads, so bf16 behaviour and speed are unchanged. Override with `SENSENOVA_NO_QUANT`, `SENSENOVA_NO_BRIDGE`, `SENSENOVA_FORCE_BRIDGE` or `SENSENOVA_NO_QT_GUARDS`.
- The upstream 1.3.4 pure-PyTorch RoPE fix is preserved, and the three RoPE bases (time, spatial, vision) can now be overridden per sampling pass through `transformer_options`; the prefix cache key includes them so a rescaled run never reuses stale KV.
- New quant tooling in `tools/`, a maintenance contract in `memory.md`, `docs/rope_lab_integration.md`, plus tests for quantized headers, CRLF assets and the RoPE bases.
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
