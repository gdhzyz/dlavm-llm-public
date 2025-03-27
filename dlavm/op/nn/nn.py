from ...adr.base import Call, Op, DataType, DataEnum
from . import (
    _nn,
    _ohbm
)
from .attrs import *


def conv2d(*args, **kwargs):
    return Call(Op.Get("nn.conv2d"), args, Conv2dAttrs(kwargs))


def mvm_f16xi4(*args, **kwargs):
    return Call(Op.Get("nn.mvm_f16xi4"), args, MVMF16xI4Attrs(kwargs))


def mvm_f16xf16(*args, **kwargs):
    return Call(Op.Get("nn.mvm_f16xf16"), args, MVMF16xF16Attrs(kwargs))


def layer_norm(*args, **kwargs):
    kwargs["rms"] = 0
    return Call(Op.Get("nn.norm"), args, NormAttrs(kwargs))


def rms_norm(*args, **kwargs):
    kwargs["rms"] = 1
    return Call(Op.Get("nn.norm"), args, NormAttrs(kwargs))


def softmax(*args, **kwargs):
    if kwargs.get("mask", False):
        kwargs["auto_mask"] = False
    elif kwargs.get("auto_mask", True):
        kwargs["auto_mask"] = True
    return Call(Op.Get("nn.softmax"), args, SoftmaxAttrs(kwargs))


def add(*args, **kwargs):
    kwargs["mode"] = ElementwiseMode.add
    return Call(Op.Get("nn.elementwise"), args, ElementwiseAttrs(kwargs))


def mul(*args, **kwargs):
    kwargs["mode"] = ElementwiseMode.mul
    return Call(Op.Get("nn.elementwise"), args, ElementwiseAttrs(kwargs))


def sub(*args, **kwargs):
    kwargs["mode"] = ElementwiseMode.sub
    return Call(Op.Get("nn.elementwise"), args, ElementwiseAttrs(kwargs))


def activate(*args, **kwargs):
    return Call(Op.Get("nn.activate"), args, Attrs(kwargs))


def kcache2hbm(data, **kwargs):
    kwargs["k_mode"] = True
    node = Call(Op.Get("nn.kvcache2hbm"), [data], KvcacheAttrs(kwargs))
    node.prefix = "hbm_cache"
    return node


def vcache2hbm(data, **kwargs):
    kwargs["k_mode"] = False
    node = Call(Op.Get("nn.kvcache2hbm"), [data], KvcacheAttrs(kwargs))
    node.prefix = "hbm_cache"
    return node


def rope_glm(*args, **kwargs):
    kwargs["mode"] = RoPEMode.glm
    return Call(Op.Get("nn.rope"), args, RoPEAttrs(kwargs))


def rope_qwen(*args, **kwargs):
    kwargs["mode"] = RoPEMode.qwen
    return Call(Op.Get("nn.rope"), args, RoPEAttrs(kwargs))


def empty_f16_hbm(shape, device):
    dtype = DataType(DataEnum.fp16, DataEnum.hbm)
    kwargs = {
        "shape": shape,
        "dtype": dtype,
        "device": device,
    }
    return Call(Op.Get("nn.empty"), [], EmptyAttrs(kwargs))