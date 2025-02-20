from ..base import Op, Tensor, Tuple, DataEnum, DataType
from functools import reduce


def CHECK_ERROR(judge, error_str):
    if judge:
        print(error_str)
        exit(-1)


def CastRel(args, attrs):
    if len(args) != 2:
        return False, []
    dshape, dtype = args[0].shape, args[0].dtype  # H, W, C
    return True, Tensor(dshape, dtype)


Op.Register("onnx.cast", CastRel)


def UnsqueezeRel(args, attrs):
    dshape, dtype = args[0].shape, args[0].dtype  # H, W, C
    oshape = dshape.insert(attrs["axes"], 1)
    return True, Tensor(oshape, dtype)


Op.Register("onnx.unsqueeze", UnsqueezeRel)


def ExpandRel(args, attrs):
    dshape, dtype, device = args[0].shape, args[0].dtype, args[0].device
    if len(dshape) != len(attrs["shape"]):
        return False, "shape ndim does not match"
    oshape = [dshape[i]*int(attrs["shape"][i]) for i in range(len(dshape))]
    return True, Tensor(oshape, dtype, device)


Op.Register("onnx.expand", ExpandRel)


def TransposeRel(args, attrs):
    dshape, dtype, device = args[0].shape, args[0].dtype, args[0].device
    if len(dshape) != len(attrs["perm"]):
        return False, "shape ndim does not match"
    oshape = []
    for i in range(len(dshape)):
        oshape.append(dshape[attrs["perm"][i]])
    return True, Tensor(oshape, dtype, device)


Op.Register("onnx.transpose", TransposeRel)


def AttentionRel(args, attrs):
    qshape, qtype, device = args[0].shape, args[0].dtype, args[0].device
    kshape, ktype, device = args[1].shape, args[1].dtype, args[1].device
    vshape, vtype, device = args[2].shape, args[2].dtype, args[2].device
    return True, Tensor(qshape, qtype, device)


Op.Register("onnx.attention", AttentionRel)


def SliceRel(args, attrs):
    qshape, qtype, device = args[0].shape, args[0].dtype, args[0].device
    starts, ends, axes = attrs["starts"], attrs["ends"], attrs["axes"]
    oshape = [i for i in qshape]
    ends = [ends] if isinstance(ends, int) else ends
    starts = [starts] if isinstance(starts, int) else starts
    oshape[axes] = ends[0] - starts[0]
    return True, Tensor(oshape, qtype, device)


Op.Register("onnx.slice", SliceRel)


def SigmoidRel(args, attrs):
    dshape, dtype, device = args[0].shape, args[0].dtype, args[0].device
    return True, Tensor(dshape, dtype, device)


Op.Register("onnx.sigmoid", SigmoidRel)