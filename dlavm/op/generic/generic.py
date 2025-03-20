from ...adr.base import Call, VM, Op, Var, Constant, DataEnum, DataType
from . import (
    _generic,
    _ohbm,
)
from .attrs import *


def var_hbm(name, shape, dtype=DataEnum.fp16, prefix="runtime"):
    dtype = DataType(dtype, DataEnum.hbm)
    expr = Var(name, shape, dtype)
    expr.prefix = prefix
    return expr


def var_ddr(name, shape, dtype=DataEnum.fp16, prefix="runtime"):
    dtype = DataType(dtype, DataEnum.ddr)
    expr = Var(name, shape, dtype)
    expr.prefix = prefix
    return expr


def const_ddr(name, data, shape=None, dtype=DataEnum.fp16, prefix="weight"):
    dtype = DataType(dtype, DataEnum.ddr)
    expr = Constant(name, data, shape, dtype)
    expr.prefix = prefix
    return expr


def const_hbm(name, data, shape=None, dtype=DataEnum.int4, prefix="hbm"):
    dtype = DataType(dtype, DataEnum.hbm)
    expr = Constant(name, data, shape, dtype)
    expr.prefix = prefix
    return expr


def reshape(data, new_shape:list):
    return VM(Op.Get("reshape"), [data], Attrs({"new_shape":new_shape}))


# TODO: this node should be Call node but there is not impl. function for it
def transpose(data, new_axis:list):
    return VM(Op.Get("transpose"), [data], Attrs({"new_axis":new_axis}))


def split(data, axis:int, size:list):
    attrs = Attrs({"axis":axis, "size":size})
    return VM(Op.Get("split"), [data], attrs)


def concat(*args):
    '''
    @brief: concat op just support axis -1
    '''
    attrs = Attrs({"axis":-1})
    return VM(Op.Get("split"), args, attrs)