from ..base import Op, Tensor, Tuple, DataEnum, DataType
from functools import reduce


def CHECK_ERROR(judge, error_str):
    if judge:
        print(error_str)
        exit(-1)


def MVMRel(args, attrs):
    if len(args) != attrs["skip"]+1:
        return False, "skip should be 1"
    device = args[0].device
    dshape, dtype = args[0].shape, args[0].dtype  # H, W, C
    wshape, wtype = args[1].shape, args[1].dtype  # Cin, Cout
    if dtype.mapped != DataEnum.ddr or dtype.dtype != DataEnum.fp16:
        return False, "data type error, should be ddr with fp16"
    if wtype.mapped != DataEnum.hbm or wtype.dtype != DataEnum.int4:
        return False, "weight type error, should be ddr with int4"
    if dshape[-1] != wshape[0]:
        return False, "weight shape should be [in_channels, out_channels]"
    if len(args) > 2:
        for n in range(len(args)-2):
            if args[1] != args[2+n]:
                return False, "not support"
    if attrs.get("arg_max", 0):
        return False, "arg max not support now"
    oshape = [i for i in dshape]
    oshape[-1] = wshape[-1]
    return True, Tensor(oshape, dtype, device)


Op.Register("accel.hbm.mvm", MVMRel)


def TRPMVMRel(args, attrs):
    if len(args) != 2:
        return False, "only 2 arguments is necessary"
    device = args[0].device
    dshape, dtype = args[0].shape, args[0].dtype  # FH, H, W, C
    wshape, wtype = args[1].shape, args[1].dtype  # WH, O, C
    if dtype.mapped != DataEnum.ddr or dtype.dtype != DataEnum.fp16:
        return False, "args[0] gets wrong data type, it should be ddr and fp16"
    if wtype.mapped != DataEnum.hbm or wtype.dtype != DataEnum.fp16:
        return False, "args[1] gets wrong data type, it should be hbm and fp16"
    if dshape[-1] != wshape[-1]:
        return False, "shape not match, please check"
    oshape = [i for i in dshape]
    oshape[-1] = wshape[-2]
    return True, Tensor(oshape, dtype, device)


Op.Register("accel.hbm.trp_mvm", TRPMVMRel)


def F2WMVMRel(args, attrs):
    if len(args) != 2:
        return False, "only 2 arguments is necessary"
    device = args[0].device
    dshape, dtype = args[0].shape, args[0].dtype  # FH, H, W, C
    wshape, wtype = args[1].shape, args[1].dtype  # WH, O, C
    if dtype.mapped != DataEnum.ddr or dtype.dtype != DataEnum.fp16:
        return False, "accel.hbm.mvm_afterF2W gets wrong data type"
    if wtype.mapped != DataEnum.hbm or wtype.dtype != DataEnum.fp16:
        return False, "accel.hbm.mvm_afterF2W gets wrong weight type"
    if dshape[-1] != wshape[-2]:
        return False, "accel.hbm.mvm_afterF2W needs same cache"
    oshape = [i for i in dshape]
    oshape[-1] = wshape[-1]
    return True, Tensor(oshape, dtype, device)


Op.Register("accel.hbm.f2w_mvm", F2WMVMRel)


def Dat2HBMRel(args, attrs):
    if len(args) != 1:
        return False, "only 1 arguments is necessary"
    device = args[0].device
    dshape, dtype = args[0].shape, args[0].dtype  # FH, H, W, C
    if dtype.mapped != DataEnum.ddr or dtype.dtype != DataEnum.fp16:
        return False, "accel.hbm.dat2hbm gets wrong data type"
    oshape = [i for i in dshape]
    tensor = Tensor(oshape, DataType(DataEnum.fp16, DataEnum.hbm), device)
    tensor.bytesize = device.MAX_TOKEN * device.MAX_CH_per_HEAD * 16 // 8 * dshape[0] // device.HBM_Port
    return True, tensor


Op.Register("accel.hbm.dat2hbm", Dat2HBMRel)


def MVMafterTRPRel(args, attrs):
    if len(args) != 2:
        return False, "only 2 arguments is necessary"
    device = args[0].device
    dshape, dtype = args[0].shape, args[0].dtype  # FH, H, W, C
    wshape, wtype = args[1].shape, args[1].dtype  # WH, O, C
    if dtype.mapped != DataEnum.ddr or dtype.dtype != DataEnum.fp16:
        return False, "args[0] gets wrong data type, it should be ddr and fp16"
    if wtype.mapped != DataEnum.ddr or wtype.dtype != DataEnum.fp16:
        return False, "args[1] gets wrong data type, it should be ddr and fp16"
    if dshape[-1] != wshape[-1] or dshape[-2] != wshape[-2]:
        return False, "shape not match, please check"
    oshape = [i for i in dshape]
    oshape[-1] = wshape[-2]
    return True, Tensor(oshape, dtype, device)


Op.Register("accel.hbm.mvm_afterTRP", MVMafterTRPRel)


def MVMafterF2WRel(args, attrs):
    if len(args) != 2:
        return False, "only 2 arguments is necessary"
    device = args[0].device
    dshape, dtype = args[0].shape, args[0].dtype  # FH, H, W, C
    wshape, wtype = args[1].shape, args[1].dtype  # WH, O, C
    if dtype.mapped != DataEnum.ddr or dtype.dtype != DataEnum.fp16:
        return False, "accel.hbm.mvm_afterF2W gets wrong data type"
    if wtype.mapped != DataEnum.ddr or wtype.dtype != DataEnum.fp16:
        return False, "accel.hbm.mvm_afterF2W gets wrong weight type"
    if dshape[-2] != wshape[-2]:
        return False, "accel.hbm.mvm_afterF2W needs same token"
    oshape = [i for i in dshape]
    oshape[-1] = wshape[-1]
    return True, Tensor(oshape, dtype, device)


Op.Register("accel.hbm.mvm_afterF2W", MVMafterF2WRel)


def MVMBNRel(args, attrs):
    if len(args) != attrs["skip"]+2:
        return False, []
    device = args[0].device
    dshape, dtype = args[0].shape, args[0].dtype  # H, W, C
    wshape, wtype = args[1].shape, args[1].dtype  # Cin, Cout
    bshape, btype = args[-1].shape, args[-1].dtype  # H, W, C
    if dtype.mapped != DataEnum.ddr or dtype.dtype != DataEnum.fp16:
        return False, "data type error, it should be fp16"
    if wtype.mapped != DataEnum.hbm or wtype.dtype != DataEnum.int4:
        return False, "weight type error, it should be int4"
    if btype.mapped != DataEnum.ddr or btype.dtype != DataEnum.fp16:
        return False, "bias type error, it should be fp16"
    if dshape[-1] != wshape[0]:
        return False, []
    if bshape[-1] != wshape[1]*2:
        return False, []
    if bshape[:-1] != [1 for i in range(len(bshape)-1)]:
        return False, []
    oshape = [i for i in dshape]
    oshape[-1] = wshape[1]
    if attrs.get("arg_max", 0):
        arg_max_tensor = Tensor([1, oshape[-2]], dtype, device)
        setattr(arg_max_tensor, "csb_read", 40)
        tensors = Tuple([Tensor(oshape, dtype, device), arg_max_tensor])
        return True, tensors
        # return True, Tuple([Tensor(oshape, dtype, device), Tensor(oshape, dtype, device)])
    else:
        return True, Tensor(oshape, dtype, device)


Op.Register("accel.hbm.mvm_bn", MVMBNRel)


def MVMBNResRel(args, attrs):
    if len(args) != attrs["skip"]+3:
        return False, []
    device = args[0].device
    dshape, dtype = args[0].shape, args[0].dtype  # H, W, C
    wshape, wtype = args[1].shape, args[1].dtype  # Cin, Cout
    bshape, btype = args[-2].shape, args[-2].dtype  # H, W, C
    rshape, rtype = args[-1].shape, args[-1].dtype  # H, W, C
    if dtype.mapped != DataEnum.ddr or dtype.dtype != DataEnum.fp16:
        return False, "data check error: " + str(args[0])
    if wtype.mapped != DataEnum.hbm or wtype.dtype != DataEnum.int4:
        return False, "weight check error: " + str(args[1])
    if btype.mapped != DataEnum.ddr or btype.dtype != DataEnum.fp16:
        return False, "bn check error: " + str(args[-2])
    if rtype.mapped != DataEnum.ddr or rtype.dtype != DataEnum.fp16:
        return False, "res check error: " + str(args[-1])
    if dshape[-1] != wshape[0]:
        return False, f"input channel check error: {dshape[2]} and {wshape[0]}" 
    if bshape[-1] != wshape[1]*2:
        return False, f"bn channel check error: {bshape[-1]} and {wshape[1]}"
    if bshape[:-1] != [1 for i in range(len(bshape)-1)]:
        return False, "bn shape is error!"
    if len(args) > 4:
        for n in range(len(args)-4):
            if args[1] != args[2+n]:
                return False, "weights do not match!"
    oshape = [dshape[0], dshape[1], wshape[1]]
    # if rshape[0] != oshape[0] or rshape[1] != oshape[1] or rshape[2] != oshape[2]:
    if rshape[2] != oshape[2]:
        rshape_str = "(" + ", ".join([str(n) for n in rshape]) + ")"
        oshape_str = "(" + ", ".join([str(n) for n in oshape]) + ")"
        return False, f"res add data{rshape_str} is not same with output data{oshape_str}"
    if attrs.get("arg_max", 0):
        arg_max_tensor = Tensor([1, oshape[-2]], dtype, device)
        setattr(arg_max_tensor, "csb_read", 40)
        tensors = Tuple([Tensor(oshape, dtype, device), arg_max_tensor])
        return True, tensors
    else:
        return True, Tensor(oshape, dtype, device)


Op.Register("accel.hbm.mvm_bn_res", MVMBNResRel)


def AddRel(args, attrs):
    if len(args) != 2:
        return False, []
    device = args[0].device
    dshape, dtype = args[0].shape, args[0].dtype  # H, W, C
    wshape, wtype = args[1].shape, args[1].dtype
    if dtype.mapped != DataEnum.ddr or dtype.dtype != DataEnum.fp16:
        return False, []
    if wtype.mapped != DataEnum.ddr or wtype.dtype != DataEnum.fp16:
        return False, []
    if dshape != wshape and dshape[-1] != wshape[-1]:
        return False, "two data shapes not match"
    return True, Tensor(dshape, dtype, device)


Op.Register("accel.hbm.add", AddRel)


def MulRel(args, attrs):
    if len(args) != 2:
        return False, []
    device = args[0].device
    dshape, dtype = args[0].shape, args[0].dtype  # H, W, C
    wshape, wtype = args[1].shape, args[1].dtype
    if dtype.mapped != DataEnum.ddr or dtype.dtype != DataEnum.fp16:
        return False, []
    if wtype.mapped != DataEnum.ddr or wtype.dtype != DataEnum.fp16:
        return False, []
    if dshape != wshape:
        return False, []
    return True, Tensor(dshape, dtype, device)


Op.Register("accel.hbm.mul", MulRel)


def LayerNormRel(args, attrs):
    if len(args) != 2:
        return False, []
    device = args[0].device
    dshape, dtype = args[0].shape, args[0].dtype  # H, W, C
    wshape, wtype = args[1].shape, args[1].dtype
    if dtype.mapped != DataEnum.ddr or dtype.dtype != DataEnum.fp16:
        return False, []
    if wtype.mapped != DataEnum.ddr or wtype.dtype != DataEnum.fp16:
        return False, []
    if dshape[-1]*2 != wshape[-1]:
        return False, "input channels not match"
    if wshape[:-1] != [1 for i in range(len(wshape)-1)]:
        return False, "weight first ndim should be 1"
    if attrs.get("kvcache_token", 0):
        oshape = [i for i in dshape]
        oshape[-2] = 1
        return True, Tensor(oshape, dtype, device)
    else:
        return True, Tensor(dshape, dtype, device)


Op.Register("accel.hbm.layer_norm", LayerNormRel)


def SoftmaxRel(args, attrs):
    if len(args) != 1:
        return False, []
    device = args[0].device
    dshape, dtype = args[0].shape, args[0].dtype  # H, W, C
    if dtype.mapped != DataEnum.ddr or dtype.dtype != DataEnum.fp16:
        return False, []
    return True, Tensor(dshape, dtype, device)


Op.Register("accel.hbm.softmax", SoftmaxRel)


def PosEmbRel(args, attrs):
    if len(args) != 2:
        return False, []
    device = args[0].device
    dshape, dtype = args[0].shape, args[0].dtype  # H, W, C
    wshape, wtype = args[1].shape, args[1].dtype
    if dtype.mapped != DataEnum.ddr or dtype.dtype != DataEnum.fp16:
        return False, "input data type is error"
    if wtype.mapped != DataEnum.ddr or wtype.dtype != DataEnum.fp16:
        return False, "weight data type is error"
    if dshape[-1] != wshape[-1]*2:
        return False, "input channel not match"
    return True, Tensor(dshape, dtype, device)


Op.Register("accel.hbm.pos_emb", PosEmbRel)


def TransposeRel(args, attrs):
    if len(args) != 1:
        return False, []
    device = args[0].device
    dshape, dtype = args[0].shape, args[0].dtype  # H, W, C
    if dtype.mapped != DataEnum.ddr or dtype.dtype != DataEnum.fp16:
        return False, "input data type is error"
    if dshape[:-2] != [1 for i in range(len(dshape)-2)]:
        return False, "the first ndim should be 1"
    oshape = [dshape[-1], dshape[-2]]
    return True, Tensor(oshape, DataType(DataEnum.int4, DataEnum.hbm), device)


Op.Register("accel.hbm.transpose", TransposeRel)


def Feature2WeightRel(args, attrs):
    if len(args) != 1:
        return False, []
    device = args[0].device
    dshape, dtype = args[0].shape, args[0].dtype  # H, W, C
    if dtype.mapped != DataEnum.ddr or dtype.dtype != DataEnum.fp16:
        return False, "input data type is error"
    if dshape[:-2] != [1 for i in range(len(dshape)-2)]:
        return False, "the first ndim should be 1"
    oshape = [dshape[-2], dshape[-1]]
    return True, Tensor(oshape, DataType(DataEnum.int4, DataEnum.hbm), device)


Op.Register("accel.hbm.feature2weight", Feature2WeightRel)


def ActivateRel(args, attrs):
    if len(args) != 2:
        return False, []
    device = args[0].device
    dshape, dtype = args[0].shape, args[0].dtype  # H, W, C
    wshape, wtype = args[1].shape, args[1].dtype
    if dtype.mapped != DataEnum.ddr or dtype.dtype != DataEnum.fp16:
        return False, "input data type is error"
    if wtype.mapped != DataEnum.ddr:
        return False, "Activate weight should be DDR data"
    if reduce(lambda x, y: x * y, wshape) < 48:
        return False, "Maybe activate weight is too small"
    return True, Tensor(dshape, dtype, device)


Op.Register("accel.hbm.activate", ActivateRel)


def Conv2dRel(args, attrs):
    pad_y, pad_x = attrs["padding"]
    stride_y, stride_x = attrs["strides"]
    if len(args) != 2:
        return False, []
    device = args[0].device
    dshape, dtype = args[0].shape, args[0].dtype  # H, W, C
    wshape, wtype = args[1].shape, args[1].dtype  # HWIO
    if dtype.mapped != DataEnum.ddr or dtype.dtype != DataEnum.fp16:
        return False, "feature data type should be DDR FP16"
    if wtype.mapped != DataEnum.hbm or wtype.dtype != DataEnum.int4:
        return False, "weight data type should be HBM INT4"
    if dshape[-1] != wshape[-2]:
        return False, "input channel not match for data and weight, weight layout should be HWIO"
    oshape = [i for i in dshape]
    oshape[-1] = wshape[-1]
    oshape[-2] = (dshape[-2] + 2*pad_x - wshape[1]) // stride_x + 1
    oshape[-3] = (dshape[-3] + 2*pad_y - wshape[0]) // stride_y + 1
    return True, Tensor(oshape, dtype, device)


Op.Register("accel.hbm.conv2d", Conv2dRel)


def Dat2HBM_Rel(args, attrs):
    if len(args) != 1:
        return False, "only 1 arguments is necessary"
    device = args[0].device
    dshape, dtype = args[0].shape, args[0].dtype  # FH, H, W, C
    if dtype.mapped != DataEnum.ddr or dtype.dtype != DataEnum.fp16:
        return False, "accel.hbm.dat_hbm gets wrong data type"
    oshape = [i for i in dshape]
    last_token = attrs["last_token"]
    if last_token is None:
        last_token = 0
    oshape[-2] = oshape[-2] + last_token
    tensor = Tensor(oshape, DataType(DataEnum.fp16, DataEnum.hbm), device)
    tensor.bytesize = device.MAX_TOKEN * device.MAX_CH_per_HEAD * 16 // 8 * dshape[0] // device.HBM_Port
    return True, tensor


Op.Register("accel.hbm.dat_hbm", Dat2HBM_Rel)

