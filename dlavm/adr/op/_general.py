from ..base import Op, Tensor, Tuple, DataEnum
from functools import reduce


def CHECK_ERROR(judge, error_str):
    if judge:
        print(error_str)
        exit(-1)


def ReshapeRel(args, attrs):
    if len(args) != 1:
        return False, []
    device = args[0].device
    dshape, dtype = args[0].shape, args[0].dtype  # H, W, C
    if dtype.mapped != DataEnum.ddr or dtype.dtype != DataEnum.fp16:
        return False, []

    def mul(x, y):
        return x * y
    new_shape = attrs["new_shape"]
    if attrs["force"]:
        bytesize = args[0].get_bytesize()
        new_bytesize = device.malloc_bytes(new_shape, dtype)
        if new_bytesize <= bytesize:
            return True, Tensor(new_shape, dtype, device)
        return False, f"force reshape needs enable byte size, needs {new_bytesize} but got {bytesize}"
    else:
        if reduce(mul, new_shape) < 0:
            for i in range(len(new_shape)):
                if new_shape[i] == -1:
                    new_shape[i] = reduce(mul, dshape) // (-1*reduce(mul, new_shape))
        new_shape = [i for i in new_shape]
        if reduce(mul, new_shape) == reduce(mul, dshape):
            return True, Tensor(new_shape, dtype, device)
        return False, "new_shape mismatch"


Op.Register("accel.reshape", ReshapeRel)


def SplitRel(args, attrs):
    if len(args) != 1:
        return False, []
    device = args[0].device
    dshape, dtype = args[0].shape, args[0].dtype  # H, W, C
    if dtype.mapped != DataEnum.ddr or dtype.dtype != DataEnum.fp16:
        return False, []
    if attrs["axis"] == 0:
        CHECK_ERROR(sum(attrs["new_chs"]) != dshape[0], "Check attrs of split Error")
        new_chs = attrs["new_chs"]
        otensors = []
        tmp_shape = [i for i in dshape[1:]]
        for new_ch in new_chs:
            otensors.append(Tensor([new_ch] + tmp_shape, dtype, device))
        return True, Tuple(otensors)
    if attrs["axis"] == 2 or attrs["axis"] == -1:
        CHECK_ERROR(sum(attrs["new_chs"]) != dshape[2], "Check attrs of split Error")
        new_chs = attrs["new_chs"]
        otensors = []
        tmp_shape = [i for i in dshape[:-1]]
        for new_ch in new_chs:
            otensors.append(Tensor(tmp_shape + [new_ch], dtype, device))
        return True, Tuple(otensors)
    else:
        CHECK_ERROR(sum(attrs["new_chs"]) != dshape[attrs["axis"]], "Check attrs of split Error")
        new_chs = attrs["new_chs"]
        otensors = []
        for new_ch in new_chs:
            tmp_shape = [i for i in dshape]
            tmp_shape[attrs["axis"]] = new_ch
            otensors.append(Tensor(tmp_shape, dtype, device))
        return True, Tuple(otensors)
    axis = attrs["axis"]
    return False, f"split attr axis={axis} is not support, 0 and -1 is available"


Op.Register("accel.split", SplitRel)


def ReallocRel(args, attrs):
    if len(args) != 1:
        return False, []
    device = args[0].device
    dshape, dtype = args[0].shape, args[0].dtype  # H, W, C
    if dtype.mapped != DataEnum.ddr or dtype.dtype != DataEnum.fp16:
        return False, "accel.realloc doesn't support this type"
    new_bytesize = args[0].device.malloc_bytes(attrs["new_shape"], dtype)
    args[0].bytesize = new_bytesize
    return True, args[0]


Op.Register("accel.realloc", ReallocRel)


def TupleRel(args, attrs):
    tensors = []
    for i in range(len(args)):
        device = args[i].device
        dshape, dtype = args[i].shape, args[i].dtype  # H, W, C
        tensors.append(Tensor(dshape, dtype, device))
    return True, Tuple(tensors)


Op.Register("accel.tuple", TupleRel)


def ConcatRel(args, attrs):
    device = args[0].device
    ranks = [len(arg.shape) for arg in args]
    dshape, dtype = args[0].shape, args[0].dtype
    if ranks != [ranks[0]]*len(ranks):
        return False, "concat should be same rank tensors"
    if attrs["dim"] >= len(dshape):
        return False, "dim set error: " + str(attrs["dim"])
    oshape = [i for i in dshape]
    oshape[attrs["dim"]] = sum([arg.shape[attrs["dim"]] for arg in args])
    return True, Tensor(oshape, dtype, device)


Op.Register("accel.concat", ConcatRel)