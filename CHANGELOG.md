# Changelog

本文件记录 ComfyUI 节点本身的版本变化。模型权重的下载和说明见 Hugging Face 模型页。

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
