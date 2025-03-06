from ...adr.base import Attrs


class MVMF16xI4Attrs(Attrs):

    default = {
        "relu": False,
        "arg_max": False,
        # attention
        "out_head": False,
        "ch_head": 128,
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
        "mask": False
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
