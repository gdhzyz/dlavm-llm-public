from ..base import Op, Call, Var, Constant, DataEnum, DataType


def cache(data, last_data):
    attrs = {}
    return Call(Op.Get("cpu.cache"), [data, last_data], attrs)