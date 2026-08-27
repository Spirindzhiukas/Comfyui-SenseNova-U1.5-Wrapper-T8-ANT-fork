"""ConvRot-aware Linear forwards for quantized SenseNova checkpoints.

Provenance: ported from ``Milor123/ComfyUI-SenseNova-U1.5-ConvRot`` at commit
``7e1e320``, where it was validated against the official bf16 outputs with
per-layer traces (int8-ConvRot and ConvRot W4A4/W4A8 conversions).

ComfyUI core stores convrot metadata on quantized weights but its generic
dispatch computes plain (dequantized) linears without rotating activations on
older comfy-kitchen builds, so offline-folded checkpoints evaluate against the
wrong basis. This factory subclasses ComfyUI's mixed-precision ops and routes
convrot weights through comfy-kitchen kernels (which rotate internally); if
anything upstream already materialized the weight as float, it rotates the
activation explicitly - the math is identical.

Current ComfyUI releases (0.31.x with comfy-kitchen >= 0.2.31) pass the convrot
flags to the kernels themselves, so :func:`quant_bridge_needed` only installs
this on stacks where that support is missing. Force it with
``SENSENOVA_FORCE_BRIDGE=1`` to reproduce the reference numbers of the original
ConvRot fork, and disable it with ``SENSENOVA_NO_BRIDGE=1``.
"""

import json
import logging
import os

import torch

import comfy.ops

try:  # comfy.quant_ops exists in every supported ComfyUI release, but the
    # QuantizedTensor types are only real ones when comfy-kitchen is installed.
    from comfy.quant_ops import QUANT_ALGOS, QuantizedTensor, TensorWiseINT8Layout
except Exception:  # pragma: no cover - unsupported ComfyUI core
    QUANT_ALGOS = {}
    QuantizedTensor = None
    TensorWiseINT8Layout = None

QUANT_METADATA_SUFFIX = ".comfy_quant"
_REPORTED_NAN = set()


def _warn_nan(name, where, out):
    if (torch.isnan(out).any() or torch.isinf(out).any()) and name not in _REPORTED_NAN:
        _REPORTED_NAN.add(name)
        logging.warning(
            f"[sensenova-quant] NaN/Inf detected in {where} output of '{name}' "
            f"(first occurrence; further occurrences suppressed)"
        )


def _rotate_input(x, group_size):
    from comfy_kitchen.backends.eager.convrot_w4a4 import _build_hadamard

    h = _build_hadamard(group_size, x.device, x.dtype)
    features = x.shape[-1]
    rotated = torch.matmul(x.reshape(-1, features // group_size, group_size), h)
    return rotated.reshape(x.shape)


def _layer_config(tensor):
    """Decode a ``*.comfy_quant`` payload from a state dict or header tensor."""
    try:
        return json.loads(bytes(tensor.numpy()).decode("utf-8"))
    except Exception:
        return {}


def state_dict_quant_formats(state_dict):
    """``{layer stem: format}`` for every quantized layer, empty for bf16 files."""
    formats = {}
    for key in state_dict:
        if not key.endswith(QUANT_METADATA_SUFFIX):
            continue
        config = _layer_config(state_dict[key])
        formats[key[: -len(QUANT_METADATA_SUFFIX)]] = config.get("format")
    return formats


def _kitchen_available():
    try:
        import comfy_kitchen  # noqa: F401
    except Exception:
        return False
    return True


CONVROT_PROBE_GROUP_SIZE = 256
_CONVROT_PROBE_CACHE = {}


def _rotate_groups(value, hadamard, group_size):
    features = value.shape[-1]
    rotated = torch.matmul(value.reshape(-1, features // group_size, group_size), hadamard)
    return rotated.reshape(value.shape)


def kitchen_honours_int8_convrot(device=None):
    """Measure whether the installed comfy-kitchen INT8 kernel rotates activations.

    A signature check is not enough here: some builds accept ``convrot`` on
    ``ck.int8_linear`` and quietly ignore it, which evaluates ConvRot-folded
    weights against the wrong basis. That is exactly the "runs fine, output is a
    checkerboard" failure, so the capability is measured instead: a small
    deterministic INT8 weight is fed to the kernel with ``convrot=True`` and the
    result is compared against the rotated reference and the unrotated one.

    Returns True (kernel rotates), False (it does not / cannot be asked to), or
    None when no measurement is possible; the caller treats None as "not
    supported" so the fallback is the known-good path.
    """
    key = str(device)
    if key in _CONVROT_PROBE_CACHE:
        return _CONVROT_PROBE_CACHE[key]
    result = None
    try:
        import comfy_kitchen

        if not os.environ.get("SENSENOVA_NO_CONVROT_PROBE"):
            from comfy_kitchen.backends.eager.convrot_w4a4 import _build_hadamard

            group = CONVROT_PROBE_GROUP_SIZE
            rows = columns = group * 2
            weight = torch.randn(rows, columns, generator=torch.Generator().manual_seed(0))
            scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-6) / 127.0
            qweight = (weight / scale).round().clamp(-127, 127).to(torch.int8)
            activations = torch.randn(8, columns, generator=torch.Generator().manual_seed(1))

            target = device if device is not None else torch.device("cpu")
            qweight = qweight.to(device=target)
            scale = scale.to(device=target)
            activations = activations.to(device=target)
            weight_float = qweight.float() * scale.float()

            # the reference must see the same bf16 input the kernel is fed, so
            # only the activation quantizer can show up as error
            kernel_input = activations.to(torch.bfloat16)
            out = comfy_kitchen.int8_linear(
                kernel_input, qweight, scale,
                out_dtype=torch.float32, convrot=True, convrot_groupsize=group,
            ).float()
            hadamard = _build_hadamard(group, target, torch.float32)
            activations = kernel_input.float()
            rotated = _rotate_groups(activations, hadamard, group)
            error_rotated = torch.linalg.vector_norm(out - rotated @ weight_float.t())
            error_plain = torch.linalg.vector_norm(out - activations @ weight_float.t())
            error_rotated = float(error_rotated) / max(float(torch.linalg.vector_norm(rotated @ weight_float.t())), 1e-6)
            error_plain = float(error_plain) / max(float(torch.linalg.vector_norm(activations @ weight_float.t())), 1e-6)
            result = error_rotated < 0.2 and error_rotated < error_plain / 4.0
            logging.info(
                "[sensenova-u15] comfy-kitchen INT8 convrot probe on %s: relative error %.4f rotated / "
                "%.4f unrotated -> %s",
                target, error_rotated, error_plain, "honours convrot" if result else "ignores convrot",
            )
    except Exception as exc:  # a kernel that refuses the kwargs simply gets the bridge
        logging.debug("[sensenova-u15] convrot probe unavailable: %s", exc)
        result = None
    _CONVROT_PROBE_CACHE[key] = result
    return result


def core_supports_convrot(formats=(), device=None):
    """Is the running ComfyUI + comfy-kitchen able to rotate activations itself?

    ``comfy.ops`` gained per-format convrot handling for ``convrot_w4a4`` and
    ``asym_w4a8_int8``, which the kitchen layouts apply through their own
    kernels. Rotated INT8 additionally needs a comfy-kitchen build whose
    ``int8_linear`` really consumes ``convrot``, which is measured rather than
    assumed. Anything else has to use this bridge to stay numerically correct.
    """
    if QuantizedTensor is None or TensorWiseINT8Layout is None:
        return False
    supported = set(QUANT_ALGOS)
    for quant_format in formats or ("int8_tensorwise", "convrot_w4a4", "asym_w4a8_int8"):
        if quant_format not in supported:
            return False
    if "int8_tensorwise" not in set(formats or ("int8_tensorwise",)):
        return True
    return kitchen_honours_int8_convrot(device) is True


def quant_bridge_needed(state_dict, device=None):
    """Decide whether this state dict needs the ConvRot operations.

    Raises for a quantized checkpoint that the current install cannot run, so
    the failure happens at load time instead of mid-generation.
    """
    formats = state_dict_quant_formats(state_dict)
    if not formats:
        return False  # plain bf16 / fp32 checkpoint: never touch the ops
    if os.environ.get("SENSENOVA_NO_BRIDGE"):
        return False
    if not _kitchen_available():
        raise ValueError(
            "SenseNova-U1.5 quantized checkpoints need comfy-kitchen (>= 0.2.31) for the "
            "ConvRot kernels. Install it in the ComfyUI environment, or load the official "
            "bf16 checkpoint instead."
        )
    if os.environ.get("SENSENOVA_FORCE_BRIDGE"):
        return True
    return not core_supports_convrot(set(value for value in formats.values() if value), device)


def make_sensenova_quant_ops():
    base = comfy.ops.mixed_precision_ops({"mixed_ops": True})

    class _Linear(base.Linear):
        def _load_from_state_dict(self, *args, **kwargs):
            prefix = args[1] if len(args) > 1 else kwargs.get("prefix", "")
            super()._load_from_state_dict(*args, **kwargs)
            params = getattr(self.weight, "_params", None)
            fmt = getattr(self, "quant_format", None)
            rotated = params is not None and (
                getattr(params, "convrot", False) or fmt == "convrot_w4a4"
            )
            self._sensenova_convrot_gs = int(getattr(params, "convrot_groupsize", 256)) if rotated else None
            self._sensenova_name = prefix.rstrip(".")

        def _forward(self, input, weight, bias):
            gs = getattr(self, "_sensenova_convrot_gs", None)
            if gs is None:
                return torch.nn.functional.linear(input, weight, bias)

            fmt = getattr(self, "quant_format", None)
            if QuantizedTensor is not None and isinstance(weight, QuantizedTensor):
                if fmt == "asym_w4a8_int8":
                    # TRUE W4A8: 4-bit weights, int8 runtime activations.
                    # Eager forced for the same consistency reason as w4a4.
                    from comfy_kitchen.backends.eager.w4a8_int8 import (
                        w4a8_int8_linear as eager_w4a8,
                    )
                    from comfy_kitchen.tensor.w4a8_int8 import AsymW4A8Int8Layout

                    qdata, s_rel, s_ch, corr, cb = AsymW4A8Int8Layout.get_plain_tensors(weight)
                    out = eager_w4a8(
                        input, qdata, s_rel, s_ch,
                        codebook=cb, correction=corr, bias=bias,
                        group_size=int(getattr(weight._params, "group_size", 16)),
                        convrot_groupsize=gs,
                        out_dtype=input.dtype,
                    )
                    _warn_nan(getattr(self, "_sensenova_name", "?"), "w4a8 eager", out)
                    return out
                if fmt == "convrot_w4a4":
                    # Kitchen's W4A4 linear rotates activations internally.
                    # The EAGER implementation is forced deliberately: the CUDA
                    # kernel diverges from eager by ~15% per call on this
                    # build, which compounds destructively across 42 layers.
                    from comfy_kitchen.backends.eager.convrot_w4a4 import (
                        convrot_w4a4_linear as eager_w4a4_linear,
                    )
                    from comfy_kitchen.tensor.convrot_w4a4 import TensorCoreConvRotW4A4Layout

                    qdata, wscales = TensorCoreConvRotW4A4Layout.get_plain_tensors(weight)
                    out = eager_w4a4_linear(
                        input, qdata, wscales, bias,
                        convrot_groupsize=gs,
                        quant_group_size=int(getattr(weight._params, "quant_group_size", 64)),
                        linear_dtype=getattr(weight._params, "linear_dtype", "int4"),
                    )
                    _warn_nan(getattr(self, "_sensenova_name", "?"), "w4a4 eager", out)
                    return out
                # NOTE: comfy-kitchen builds before 0.2.31 accept convrot kwargs
                # on ck.int8_linear but ignore them, so int8 goes through the
                # exact float path below instead.

            # Exact path: rotate activations into the folded basis, then a
            # plain linear against the rotated weight. Dequantization is done
            # MANUALLY from raw qdata/scales: QuantizedTensor.dequantize()
            # dispatch is unreliable on CPU in current builds (wrong results
            # or native crashes) and weight streaming hits CPU constantly.
            #
            # Float materializations differ per format: comfy's int8 dequant
            # keeps the ROTATED basis (input must rotate), while kitchen's
            # w4a4 dequant already restores the ORIGINAL basis (plain linear).
            if QuantizedTensor is not None and isinstance(weight, QuantizedTensor):
                qdata, scale = TensorWiseINT8Layout.get_plain_tensors(weight)
                weight_float = qdata.to(input.dtype) * scale.to(input.dtype).reshape(-1, 1)
                return torch.nn.functional.linear(_rotate_input(input, gs), weight_float, bias)
            if fmt == "convrot_w4a4" or fmt == "asym_w4a8_int8":
                # Both kitchen formats dequantize back to the ORIGINAL basis.
                return torch.nn.functional.linear(input, weight, bias)
            return torch.nn.functional.linear(_rotate_input(input, gs), weight, bias)

    class Ops(base):
        Linear = _Linear

    return Ops()
