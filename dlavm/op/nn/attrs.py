from ...adr.base import Attrs
from dlavm.device import ohbm_accel


class Conv2dAttrs(Attrs):

    default = {
        "relu": False,
        "padding" : [0, 0],
        "strides" : [1, 1],
        # attention
    }


class MVMF16xI4Attrs(Attrs):

    default = {
        "relu": False,
        "argmax": False,
        # attention
        "out_heads": None, # want list, [Feature_Head, Weight_Head]
        "ch_head": ohbm_accel.OHBM.MAX_CH_per_HEAD,
    }


class MVMF16xF16Attrs(Attrs):

    default = {
        "w_trp": False,
    }


class NormAttrs(Attrs):

    default = {
    }


class SoftmaxAttrs(Attrs):

    default = {
        "mask": False,
        "auto_mask": False,
    }


class ElementwiseMode:

    init = -1
    add = 0
    sub = 1
    mul = 2
    sub_mul = 3

class ElementwiseAttrs(Attrs):

    default = {
        "mode": ElementwiseMode.init,
    }


class KvcacheAttrs(Attrs):

    default = {
        "k_mode": None,
        "cache_len": 0,
        "cache_size": ohbm_accel.OHBM.MAX_TOKEN,
    }


class RoPEMode:

    init = -1
    qwen = 0
    glm = 1

class RoPEAttrs(Attrs):

    default = {
        "mode": RoPEMode.init,
        "last_token": 0,
    }
