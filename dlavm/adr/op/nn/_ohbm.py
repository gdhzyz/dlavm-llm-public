from dlavm.device import ohbm_accel
from dlavm.adr.base import Op, DataEnum, DataType
from ._nn import (
    MVMRel
)

@Op.RegisterAttrs("nn.mvm", "rel", ohbm_accel.OHBM)
def MVMOHBMRel(args, attrs):
    check = MVMRel(args, attrs)
    if not check[0]:
        return check[0], check[1]
    for arg in args:
        if arg.dtype.mapped != DataEnum.hbm:
            return False, "dtype of args must be " + DataEnum.hbm
    if args[0].dtype.dtype != DataEnum.fp16:
        return False, "dtype of data must be " + DataEnum.fp16
    if args[1].dtype.dtype != DataEnum.int4:
        return False, "dtype of weight must be " + DataEnum.int4
    if len(args) > 2:
        for arg in args[1:]:
            if arg.dtype.dtype != DataEnum.fp16:
                return False, "dtype of args must be " + DataEnum.fp16
    return True, check[1]


