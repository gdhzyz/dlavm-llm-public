from ..base import Op, Call, Var, Constant, DataEnum, DataType


def var_ddr(name, shape, dtype=DataEnum.int8):
    dtype = DataType(dtype, DataEnum.ddr)
    return Var(name, shape, dtype)


def const_ddr(name, data, shape=None, dtype=DataEnum.int8):
    dtype = DataType(dtype, DataEnum.ddr)
    return Constant(name, data, shape, dtype)


def conv2d(data, weight, stride, padding, l0_dw, l1_dw, scales, relu=0):
    attrs = {
        "stride": stride,
        "padding": padding,
        "l0_dw": l0_dw,
        "l1_dw": l1_dw,
        "scales": scales,
        "relu": relu
    }
    return Call(Op.Get("accel.booth.conv2d"), [data, weight], attrs)
   