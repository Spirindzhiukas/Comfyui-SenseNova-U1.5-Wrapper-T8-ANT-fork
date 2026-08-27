"""Runtime guards that keep QuantizedTensors intact through ComfyUI weight streaming.

Provenance: ported from ``Milor123/ComfyUI-SenseNova-U1.5-ConvRot`` at commit
``7e1e320``.

ComfyUI's patcher feeds patched weights through ``cast_to_device`` /
``tensor.to(dtype)`` whenever a LoRA touches a layer. On packed quantized
tensors those calls either relabel ``orig_dtype`` while keeping the packed
bytes or route the packing into float math, which silently corrupts 4-bit
formats (int8 survives because its values are representable after a cast).
The guards strip dtype requests for QuantizedTensors so they reach kernels
pristine, matching the fix the community validated for other convrot models.

The guards are installed lazily from
:meth:`sensenova_u15.model_config.SenseNovaModelConfig.get_model`, i.e. only
once a quantized checkpoint is actually loaded, so a bf16 session never pays
for the extra ``isinstance`` checks on the streaming hot path. (Upstream
Milor123 keeps this module at the node root; inside the package it also stays
importable when ``sensenova_u15`` is loaded as a top-level package by the unit
tests.)
Set ``SENSENOVA_NO_QT_GUARDS=1`` to keep the stock ComfyUI behaviour.
"""

import logging
import os

import torch

_guard_installed = False


def _strip_dtype_args(args):
    head, rest = args[:1], args[1:]
    return head + tuple(a for a in rest if not isinstance(a, torch.dtype))


def guards_installed():
    return _guard_installed


def install_quant_guards():
    """Patch dtype-stripping wrappers around QuantizedTensor conversion paths.

    Returns True when the guards are active. Every lookup is defensive: a
    missing comfy-kitchen, a renamed private helper, or an already-patched
    module must degrade to "guards not installed" instead of breaking the node
    pack import for users on stock bf16 checkpoints.
    """
    global _guard_installed
    if _guard_installed:
        return True
    if os.environ.get("SENSENOVA_NO_QT_GUARDS"):
        return False

    try:
        from comfy import model_management
        from comfy_kitchen.tensor import base as kitchen_base
        from comfy_kitchen.tensor.base import QuantizedTensor
    except Exception:
        return False

    orig_cast_to_device = getattr(model_management, "cast_to_device", None)
    orig_handle_to = getattr(kitchen_base, "_handle_to", None)
    orig_handle_empty_like = getattr(kitchen_base, "_handle_empty_like", None)
    if orig_cast_to_device is None or orig_handle_to is None:
        return False

    def cast_to_device_qt_safe(tensor, device, dtype=None, copy=False):
        if isinstance(tensor, QuantizedTensor):
            dtype = None
        return orig_cast_to_device(tensor, device, dtype, copy)

    model_management.cast_to_device = cast_to_device_qt_safe

    def handle_to_dtype_safe(qt, args, kwargs, force_copy=False):
        if isinstance(qt, QuantizedTensor):
            args = _strip_dtype_args(args)
            kwargs = {k: v for k, v in kwargs.items() if k != "dtype"}
        return orig_handle_to(qt, args, kwargs, force_copy=force_copy)

    kitchen_base._handle_to = handle_to_dtype_safe

    handle_empty_like = None
    if orig_handle_empty_like is not None:
        def handle_empty_like_dtype_safe(qt, args, kwargs):
            if isinstance(qt, QuantizedTensor):
                kwargs = {k: v for k, v in kwargs.items() if k != "dtype"}
            return orig_handle_empty_like(qt, args, kwargs)

        kitchen_base._handle_empty_like = handle_empty_like_dtype_safe
        handle_empty_like = handle_empty_like_dtype_safe

    dispatch = getattr(kitchen_base, "_DISPATCH_TABLE", None)
    if isinstance(dispatch, dict):
        for op_key, handler in list(dispatch.items()):
            if handler is orig_handle_to:
                dispatch[op_key] = handle_to_dtype_safe
            elif handle_empty_like is not None and handler is orig_handle_empty_like:
                dispatch[op_key] = handle_empty_like

    _guard_installed = True
    logging.info("[sensenova-u15] QuantizedTensor dtype guards installed.")
    return True
