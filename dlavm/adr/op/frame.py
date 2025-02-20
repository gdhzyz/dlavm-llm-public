from ..base import Op, Call, Var, Constant, DataEnum, DataType


def onnx_expand(data, shape):
    attrs = {"shape": shape}
    return Call(Op.Get("onnx.expand"), [data], attrs)


def onnx_transpose(data, perm):
    attrs = {"perm": perm}
    return Call(Op.Get("onnx.transpose"), [data], attrs)


def onnx_attention(query, key, value):
    attrs = {}
    return Call(Op.Get("onnx.attention"), [query, key, value], attrs)


def onnx_slice(data, starts, ends, axes):
    attrs = {"starts": starts, "ends": ends, "axes": axes}
    return Call(Op.Get("onnx.slice"), [data], attrs)


def onnx_sigmoid(data):
    attrs = {}
    return Call(Op.Get("onnx.sigmoid"), [data], attrs)