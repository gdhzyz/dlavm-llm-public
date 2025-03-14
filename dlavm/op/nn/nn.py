from ...adr.base import Call, Op
from . import (
    _nn,
    _ohbm
)
from .attrs import *


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
