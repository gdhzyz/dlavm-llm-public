from ..base import Op, Call, Var, Constant, DataEnum, DataType
from ...device import BoothSparse


def var_ddr(name, shape, dtype=DataEnum.int8, device=BoothSparse):
    dtype = DataType(dtype, DataEnum.ddr)
    return Var(name, shape, dtype, device)


def const_ddr(name, data, shape=None, dtype=DataEnum.int8, device=BoothSparse):
    dtype = DataType(dtype, DataEnum.ddr)
    return Constant(name, data, shape, dtype, device)


def conv2d(data, weight, stride, padding, widths, sparsity, scales, relu=0):
    attrs = {
        "stride": stride,
        "padding": padding,
        "widths": widths,
        "sparsity": sparsity,
        "scales": scales,
        "relu_en": relu
    }
    return Call(Op.Get("accel.sparse.conv2d"), [data, weight], attrs)
