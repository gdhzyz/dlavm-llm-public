from dlavm.device import ohbm_accel
from dlavm.adr.base import Op, Tensor, Tuple, DataEnum, DataType
from ._nn import (
    MVMRel,
    MVMF16xF16Rel,
    SoftmaxRel,
    ActivateRel,
    Kvcache2hbmRel,
    RoPosEmbRel,
)

@Op.RegisterAttrs("nn.mvm_f16xi4", "rel", ohbm_accel.OHBM)
def MVMOHBMRel(args, attrs):
    if len(args) not in [2, 3]:
        return False, "too more arguments! support [2, 3], found " + str(len(args))
    device = args[0].device
    dtype, wtype = args[0].dtype, args[1].dtype
    dshape, wshape = args[0].shape, args[1].shape
    if len(dshape) < 3:
        return False, "dim of data shape should be [head, win, channel]"
    if hasattr(args[0], "heads"):
        if dshape[-1] != 128 or args[0].heads[-1]*dshape[-1] != wshape[1]:
            return False, "weight shape should be [out_channels, in_channels] and the head of dshape does not match after padding: " + str(args[0].heads)
        oshape = [1, dshape[-2], wshape[0]]
    else:
        if dshape[-1] != wshape[1]:
            return False, "weight shape should be [out_channels, in_channels]"
        oshape = [i for i in dshape]
        oshape[-1] = wshape[0]
    if len(args) > 2:
        if args[2].shape[-1] != wshape[0]*2 and args[2].shape[-1] != wshape[0]:
            return False, "bn weight shape should be [out_channels*2] or [out_channels]: " + str(args[2].shape[-1]) + " and " + str(wshape[0])
    for arg in args:
        if arg.dtype.mapped != DataEnum.hbm:
            return False, "dtype of args must be " + DataEnum.hbm
    if args[0].dtype.dtype != DataEnum.fp16:
        return False, "dtype of data must be " + DataEnum.fp16
    if args[1].dtype.dtype != DataEnum.int4:
        return False, "dtype of weight must be " + DataEnum.int4
    if len(args) > 2:
        for arg in args[2:]:
            if arg.dtype.dtype != DataEnum.fp16:
                return False, "dtype of args must be " + DataEnum.fp16
    if attrs.get("argmax", 0):
        arg_max_tensor = Tensor([1, oshape[-2]], dtype, device)
        setattr(arg_max_tensor, "csb_read", 40)
        tensors = Tuple([Tensor(oshape, dtype, device), arg_max_tensor])
        return True, tensors
    if attrs.get("out_heads") is not None:
        if len(attrs.get("out_heads")) not in [2]:
            return False, "too more elements in out_heads! support [Feature_Head, Weight_Head], found " + str(len(attrs.get("out_heads")))
        Feature_Head, Weight_Head = attrs.get("out_heads")
        MAX_CH_per_HEAD         = attrs.get("ch_head")
        Head_x_CHin             = (Feature_Head//Weight_Head*MAX_CH_per_HEAD)
        Head_x_CHin_div_LTout   = ((Head_x_CHin+device.L_Tout-1)//device.L_Tout)
        Feature_Head_in_Padding = (Head_x_CHin_div_LTout*device.L_Tout//MAX_CH_per_HEAD) * Weight_Head
        if wshape[0] != Feature_Head_in_Padding * MAX_CH_per_HEAD:
            return False, "weight should be padded according to Feature Head Padding: " + str([Feature_Head, Weight_Head, Feature_Head_in_Padding]) + \
                          ", found weight shape: " + str(wshape) + ", want " + str([Feature_Head_in_Padding * MAX_CH_per_HEAD, wshape[1]])
        oshape = [Feature_Head_in_Padding, oshape[1], MAX_CH_per_HEAD]
        tensor = Tensor(oshape, dtype, device)
        setattr(tensor, "heads", [Feature_Head, Weight_Head, Feature_Head_in_Padding])
        return True, tensor
    return True, Tensor(oshape, dtype, device)


@Op.RegisterAttrs("nn.mvm_f16xf16", "rel", ohbm_accel.OHBM)
def MVMOHBMRel(args, attrs):
    check = MVMF16xF16Rel(args, attrs)
    if not check[0]:
        return check[0], check[1]
    for arg in args:
        if arg.dtype.mapped != DataEnum.hbm:
            return False, "dtype of args must be " + DataEnum.hbm
    return True, check[1]


@Op.RegisterAttrs("nn.norm", "rel", ohbm_accel.OHBM)
def NormOHBMRel(args, attrs):
    if len(args) not in [2]:
        return False, "error length arguments! support 2, found " + str(len(args))
    device = args[0].device
    dtype = args[0].dtype
    dshape, wshape = args[0].shape, args[1].shape
    if dshape[-1]*2 != wshape[-1] or len(wshape) > 1:
        return False, "weight shape should be [2*in_channels]"
    for arg in args:
        if arg.dtype.mapped != DataEnum.hbm or arg.dtype.dtype != DataEnum.fp16:
            return False, "dtype of args must be " + DataEnum.hbm + " and " + DataEnum.fp16
    oshape = [i for i in dshape]
    return True, Tensor(oshape, dtype, device)


@Op.RegisterAttrs("nn.softmax", "rel", ohbm_accel.OHBM)
def SoftmaxOHBMRel(args, attrs):
    check = SoftmaxRel(args, attrs)
    if not check[0]:
        return check[0], check[1]
    for arg in args:
        if arg.dtype.mapped != DataEnum.hbm or arg.dtype.dtype != DataEnum.fp16:
            return False, "dtype of args must be " + DataEnum.hbm + " and " + DataEnum.fp16
    return True, check[1]


@Op.RegisterAttrs("nn.activate", "rel", ohbm_accel.OHBM)
def ActivateOHBMRel(args, attrs):
    check = ActivateRel(args, attrs)
    if not check[0]:
        return check[0], check[1]
    if args[0].dtype.mapped != DataEnum.hbm or args[0].dtype.dtype != DataEnum.fp16:
        return False, "dtype of args must be " + DataEnum.hbm + " and " + DataEnum.fp16
    return True, check[1]


@Op.RegisterAttrs("nn.kvcache2hbm", "rel", ohbm_accel.OHBM)
def Kvcache2HBMOHBMRel(args, attrs):
    check = Kvcache2hbmRel(args, attrs)
    if not check[0]:
        return check[0], check[1]
    if args[0].dtype.mapped != DataEnum.hbm or args[0].dtype.dtype != DataEnum.fp16:
        return False, "dtype of args must be " + DataEnum.hbm + " and " + DataEnum.fp16
    return True, check[1]


@Op.RegisterAttrs("nn.rope", "rel", ohbm_accel.OHBM)
def RoPEOHBMRel(args, attrs):
    check = RoPosEmbRel(args, attrs)
    if not check[0]:
        return check[0], check[1]
    if args[0].dtype.mapped != DataEnum.hbm or args[0].dtype.dtype != DataEnum.fp16:
        return False, "dtype of args must be " + DataEnum.hbm + " and " + DataEnum.fp16
    return True, check[1]
