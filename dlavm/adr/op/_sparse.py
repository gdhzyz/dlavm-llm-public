from ..base import Op, Tensor, Tuple, DataEnum, DataType
from functools import reduce


def CHECK_ERROR(judge, error_str):
    if judge:
        print(error_str)
        exit(-1)


def Conv2DRel(args, attrs):
    device = args[0].device
    dshape, dtype = args[0].shape, args[0].dtype  # H, W, C
    wshape, wtype = args[1].shape, args[1].dtype  # O, I, H, W
    if dtype.mapped != DataEnum.ddr:
        return False, "feature data mapped is error, only ddr support"
    if wtype.mapped != DataEnum.ddr:
        return False, "weight data mapped is error, only ddr support"
    if dshape[-1] != wshape[1]:
        return False, "feature shape and weight shape (OIHW) are not match"
    if 2 in attrs["widths"]:
        CHECK_ERROR(wtype.dtype != DataEnum.int2, "Weight Bit Width not Match!")
    Sy, Sx = attrs["stride"]
    Py, Px = attrs["padding"]
    Ky, Kx = wshape[2], wshape[3]
    out_height = (dshape[0]+2*Py-Ky)//Sy+1
    out_width = (dshape[1]+2*Px-Kx)//Sx+1
    oshape = [out_height, out_width, wshape[0]]
    return True, Tensor(oshape, dtype, device)


Op.Register("accel.sparse.conv2d", Conv2DRel)