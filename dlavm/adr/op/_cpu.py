from ..base import Op, Tensor, Tuple, DataEnum, DataType
from functools import reduce


def CHECK_ERROR(judge, error_str):
    if judge:
        print(error_str)
        exit(-1)


def CacheRel(args, attrs):
    if len(args) != 2:
        return False, []
    dshape, dtype = args[0].shape, args[0].dtype  # H, W, C
    cshape, ctype = args[1].shape, args[1].dtype  # H, W, C
    if dtype.mapped != DataEnum.ddr or dtype.dtype != DataEnum.fp16:
        return False, "input data type error"
    if ctype.mapped != DataEnum.ddr or ctype.dtype != DataEnum.fp16:
        return False, "cache data type error"
    if dshape[2] != cshape[2]:
        return False, f"the concatenate channel not same, {dshape[2]} vs {cshape[2]}"
    if dshape[0] != cshape[0] or dshape[0] != 1:
        return False, "the first ndim should be 1"
    oshape = [dshape[0], cshape[1] + dshape[1], dshape[2]]
    return True, Tensor(oshape, dtype)


Op.Register("cpu.cache", CacheRel)