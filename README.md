# SenseNova-U1.5 ComfyUI 节点

这是 SenseNova-U1.5 的 ComfyUI 原生节点，支持：

- 文生图
- 单张参考图编辑
- ComfyUI 原生 `KSampler`
- 自定义 `img_cfg` 的三路图像引导

节点只读取本地模型，运行时不会联网下载文件。

## 下载

- **模型（Hugging Face）：** [t8star/SenseNova-U1.5-Comfy](https://huggingface.co/t8star/SenseNova-U1.5-Comfy/)
- **模型网盘：** [夸克网盘](https://pan.quark.cn/s/c9c267081fbf)
- **节点仓库：** [T8mars/SenseNova-U1.5-Wrapper-T8](https://github.com/T8mars/SenseNova-U1.5-Wrapper-T8)

模型是一个约 50 GB 的 `.safetensors` 文件。下载后放到：

```text
ComfyUI/models/diffusion_models/
```

不要把模型放进本节点目录。

## 安装

进入 ComfyUI 的 `custom_nodes` 目录后运行：

```bash
git clone https://github.com/T8mars/SenseNova-U1.5-Wrapper-T8.git
```

重启 ComfyUI。搜索 `SenseNova`，能看到下面 5 个节点就说明安装成功：

- `SenseNova U1.5 Loader`
- `SenseNova Sampling Options`
- `Empty SenseNova Pixel Latent`
- `SenseNova Reference Image`
- `SenseNova Edit Guider`

本项目没有额外的 Python 依赖。

## 文生图怎么用

推荐参数：

```text
steps: 50
CFG: 4
shift: 3
sampler: euler
scheduler: normal
denoise: 1
```

基本连接顺序：

1. 用 `SenseNova U1.5 Loader` 加载模型。
2. 用两个 ComfyUI 自带的 `CLIP Text Encode` 编码提示词和空提示词。
3. `MODEL` 先连接 `SenseNova Sampling Options`，shift 保持 3。
4. 用 `Empty SenseNova Pixel Latent` 设置宽高。
5. 接入 ComfyUI 自带的 `KSampler`，最后使用 `VAE Decode` 输出图片。

可以直接参考 [examples/t2i_api.json](examples/t2i_api.json)。这是 API 格式示例，不是前端拖拽工作流。

## 图像编辑怎么用

参考图需要经过 `SenseNova Reference Image`，不要把参考图直接当成 latent。

最简单的设置是：

```text
CFG: 4
img_cfg: 1
steps: 50
shift: 3
```

`img_cfg=1` 时可以继续使用普通 `KSampler`。参考 [examples/edit_two_way_api.json](examples/edit_two_way_api.json)。

如果需要设置其他 `img_cfg`，请使用 `SenseNova Edit Guider` 和 ComfyUI 自带的 `SamplerCustomAdvanced`。参考 [examples/edit_api.json](examples/edit_api.json)。

如果编辑结果颜色太艳，先检查提示词里是否用了 `bright`、`vivid`、`neon`、`highly saturated` 一类词。可以把 CFG 降到 3～3.5，`img_cfg` 建议先保持 1。

## 运行要求

当前实测环境：

- ComfyUI `v0.31.0-8`，commit `cbbc9dab1f03d0d9a6caa8a8be7d77a7e37e1e44`
- NVIDIA CUDA
- BF16
- RTX 5090 Laptop 24 GB
- 64 GB 系统内存

2048×2048、50 步文生图和编辑都能在 24 GB 显存下完成，CUDA 峰值分配约 20 GiB。模型本身约 50 GB，加载和卸载时还会占用较多系统内存，建议准备 64 GB RAM 和足够的虚拟内存。

## 当前限制

- 只支持单张参考图
- batch size 只支持 1
- 只验证了 NVIDIA CUDA + BF16
- 不支持运行时自动下载模型
- 多参考图、量化、8-step LoRA、CFG norm、bbox/marker 和 think mode 暂未开放
- FP16、ROCm、MPS、DirectML、XPU、NPU 暂未验证

## 模型校验

当前单文件模型信息：

```text
大小：50,222,155,152 bytes
SHA256：2e5c4451969a8af9d7bcbf9d00a0fe463b15ed44149d8d79f31409e671587615
tensor：1116
revision：1f6ec60423d29939dde4202fd82ae340b144e280
```

节点会检查模型 metadata、tensor 名称、shape 和存储 dtype。如果下载不完整或模型版本不对，会直接报错，不会静默加载错误权重。

## 其他链接

- [B站](https://space.bilibili.com/385085361)
- [YouTube](https://www.youtube.com/@T8star-Aix/)
- [AI API](https://api.seedance.nz/sign-up?aff=5f4w)
- [在线 AI 应用](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121)
- [ComfyUI 整合包](https://pan.quark.cn/s/264edb7e36bd)
- [模型网盘](https://pan.quark.cn/s/c9c267081fbf)
- [Hugging Face 主页](https://huggingface.co/t8star)

## 说明

SenseNova-U1.5 模型和参考实现来自 [OpenSenseNova/SenseNova-U1](https://github.com/OpenSenseNova/SenseNova-U1)，原项目使用 Apache License 2.0。

本仓库只提供 ComfyUI 本地推理适配，不包含模型权重。详细归因见 [NOTICE](NOTICE)。
