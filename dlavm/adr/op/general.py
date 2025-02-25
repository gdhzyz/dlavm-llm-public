from ..base import Op, VM, Call, Var, Constant, DataEnum, DataType


def var_hbm(name, shape, dtype=DataEnum.fp16):
    dtype = DataType(dtype, DataEnum.hbm)
    expr = Var(name, shape, dtype)
    expr.prefix = "runtime"
    return expr


def var_ddr(name, shape, dtype=DataEnum.fp16):
    dtype = DataType(dtype, DataEnum.ddr)
    expr = Var(name, shape, dtype)
    expr.prefix = "runtime"
    return expr


def const_ddr(name, data, shape=None, dtype=DataEnum.fp16):
    dtype = DataType(dtype, DataEnum.ddr)
    expr = Constant(name, data, shape, dtype)
    expr.prefix = "weight"
    return expr


def const_hbm(name, data, shape=None, dtype=DataEnum.int4):
    dtype = DataType(dtype, DataEnum.hbm)
    expr = Constant(name, data, shape, dtype)
    expr.prefix = "hbm"
    return expr


def split(data, axis, new_chs=[], dynamic=True):
    attrs = {
        "axis": axis,
        "new_chs": new_chs,
        "dynamic": dynamic
    }
    return VM(Op.Get("accel.split"), [data], attrs)


def reshape(data, new_shape, force=0):
    attrs = {
        "new_shape": new_shape,
        "force": force
    }
    return VM(Op.Get("accel.reshape"), [data], attrs)


def realloc(data, new_shape):
    attrs = {
        "new_shape": new_shape
    }
    return VM(Op.Get("accel.realloc"), [data], attrs)


def tuple(exprs):
    attrs = {}
    return VM(Op.Get("accel.tuple"), exprs, attrs)


def concat(exprs, dim=0, **kwattrs):
    attrs = {
        "dim": dim,
        **kwattrs
    }
    return VM(Op.Get("accel.concat"), exprs, attrs)
