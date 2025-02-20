from ..adr import Op, Tuple, Tensor


def SplitDriver(args, output, attrs):
    offset = args[0].offset
    new_tensors = []
    for t in output.tensors:
        t.offset = offset
        new_tensors.append(t)
        offset += t.get_bytesize(dynamic=attrs["dynamic"])
    return Tuple(new_tensors)

Op.Get("accel.split").attrs["driver"] = SplitDriver


def ReshapeDriver(args, output, attrs):
    output.offset = args[0].offset
    return output

Op.Get("accel.reshape").attrs["driver"] = ReshapeDriver


def ReallocDriver(args, output, attrs):
    output.offset = args[0].offset
    return output

Op.Get("accel.realloc").attrs["driver"] = ReallocDriver


def TupleDriver(args, output, attrs):
    return output

Op.Get("accel.tuple").attrs["driver"] = TupleDriver