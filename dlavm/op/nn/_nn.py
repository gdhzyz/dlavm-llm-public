from ...adr.base import Op, Tensor, Tuple, DataEnum
from ... import ne

'''
@brief: MatMul op
  @args[0]: input feature data, [1, token, chin]
  @args[1]: weight data, [chout, chin]
  @Optional<args[2]>: batch norm weight, [chout*2]
  @Optional<args[3]>: res input feature data, [1, token, chout]
  @output: Tensor, [1, token, chout]
'''
def MVMRel(args, attrs):
    if len(args) not in [2, 3, 4]:
        return False, "too more arguments! support [2, 3, 4], found " + str(len(args))
    device = args[0].device
    dtype = args[0].dtype
    dshape, wshape = args[0].shape, args[1].shape
    if hasattr(args[0], "heads"):
        if dshape[-1] != 128 or args[0].heads[0]*dshape[-1] != wshape[1]:
            return False, "weight shape should be [out_channels, in_channels] and the head of dshape does not match: " + str(args[0].heads)
        oshape = [1, dshape[-2], wshape[0]]
    else:
        if dshape[-1] != wshape[1]:
            return False, "weight shape should be [out_channels, in_channels]"
        oshape = [i for i in dshape]
        oshape[-1] = wshape[0]
    if len(args) > 2:
        if args[2].shape[-1] != wshape[0]*2 and args[2].shape[-1] != wshape[0]:
            return False, "bn weight shape should be [out_channels*2] or [out_channels]"
    if attrs.get("argmax"):
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
        oshape = [Feature_Head_in_Padding, oshape[1], MAX_CH_per_HEAD]
        tensor = Tensor(oshape, dtype, device)
        setattr(tensor, "heads", attrs.get("out_heads"))
        return True, tensor
    return True, Tensor(oshape, dtype, device)

Op.Register("nn.mvm", MVMRel)
Op.Register("nn.mvm_f16xi4", MVMRel)


'''
@brief: MatMul F16xF16 op
  @args[0]: input 1 feature data, [head, token, chin]/[head, token, token]
  @args[1]: input 2 feature data, [head, token, chin]
  @attrs: w_trp, if input 2 should transpose before mvm fp16xfp16
  @output: Tensor, [head, token, token]
'''
def MVMF16xF16Rel(args, attrs):
    if len(args) not in [2]:
        return False, "too more arguments! support [2], found " + str(len(args))
    device = args[0].device
    dtype, wtype = args[0].dtype, args[1].dtype
    dshape, wshape = args[0].shape, args[1].shape
    if len(dshape) < 3 or len(wshape) < 3:
        return False, "the shape of arguments should be [head, token, chin] or [head, 1, token, chin]"
    oshape = [i for i in dshape]
    if wshape[-1] not in [128]:
        return False, "error head per channels! support 128, found " + str(wshape[-1])
    if attrs.get("w_trp", 0):
        if dshape[-1] not in [128]:
            return False, "error head per channels! support 128, found " + str(dshape[-1])
        if dshape[-1] != wshape[-1]:
            return False, "the channel does not match, found: " + str(dshape[-1]) + " and " + str(wshape[-1])
        oshape[-1] = wshape[-2]
    else:
        if dshape[-1] != wshape[-2]:
            return False, "the channel does not match, found: " + str(dshape[-1]) + " and " + str(wshape[-2])
        oshape[-1] = wshape[-1]
    if dtype.dtype != DataEnum.fp16 or wtype.dtype != DataEnum.fp16:
        return False, "the dtype of arguments shoul be " + DataEnum.fp16
    Feature_Head, Weight_Head = dshape[0], wshape[0]
    Feature_Head_in_Padding = Feature_Head
    if hasattr(args[0], "heads"):
        oshape[0] = args[0].heads[0]
        Feature_Head = args[0].heads[0]
    elif not attrs.get("w_trp"):
        MAX_CH_per_HEAD         = wshape[-1]
        Head_x_CHin             = (Feature_Head//Weight_Head*MAX_CH_per_HEAD)
        Head_x_CHin_div_LTout   = ((Head_x_CHin+device.L_Tout-1)//device.L_Tout)
        Feature_Head_in_Padding = (Head_x_CHin_div_LTout*device.L_Tout//MAX_CH_per_HEAD) * Weight_Head
        oshape[0] = Feature_Head_in_Padding
    tensor = Tensor(oshape, dtype, device)
    setattr(tensor, "heads", [Feature_Head, Weight_Head, Feature_Head_in_Padding])
    return True, tensor

Op.Register("nn.mvm_f16xf16", MVMRel)


'''
@brief: Norm op
  @args[0]: input feature data, [1, token, chin]
  @args[1]: weight data, [chin]
  @args[2]: bias data, [chin]
  @output: Tensor, [1, token, chin]
'''
def NormRel(args, attrs):
    if len(args) not in [3]:
        return False, "error length arguments! support 3, found " + str(len(args))
    device = args[0].device
    dtype = args[0].dtype
    dshape, wshape, bshape = args[0].shape, args[1].shape, args[2].shape
    if dshape[-1] != wshape[-1] or len(wshape) > 1 or wshape[-1] != bshape[-1] or len(bshape):
        return False, "weight and bias shape should be [in_channels]"
    oshape = [i for i in dshape]
    return True, Tensor(oshape, dtype, device)

Op.Register("nn.norm", NormRel)


'''
@brief: Softmax op
  @args[0]: input feature data, [1, token, chin]
  @output: Tensor, [1, token, chin]
'''
def SoftmaxRel(args, attrs):
    if len(args) not in [1]:
        return False, "error length arguments! support 1, found " + str(len(args))
    device = args[0].device
    dtype = args[0].dtype
    dshape = args[0].shape
    oshape = [i for i in dshape]
    return True, Tensor(oshape, dtype, device)

Op.Register("nn.softmax", SoftmaxRel)


'''
@brief: Elementwise op
  @args[0]: input feature data 1, [x]
  @args[1]: input feature data 2, [x]
  @output: Tensor, [x]
'''
def ElementwiseRel(args, attrs):
    if len(args) not in [2]:
        return False, "error length arguments! support 2, found " + str(len(args))
    device = args[0].device
    dtype = args[0].dtype
    dshape, wshape = args[0].shape, args[1].shape
    if len(dshape) != len(wshape):
        return False, "arguments should have same shape"
    for i in range(len(dshape)):
        if isinstance(dshape[i], ne.Expr):
            # TODO: check dynamic shape
            continue
        elif dshape[i] != wshape[i]:
            return False, "arguments should have same shape"
    oshape = [i for i in dshape]
    return True, Tensor(oshape, dtype, device)

Op.Register("nn.elementwise", ElementwiseRel)


'''
@brief: Activate op
  @args[0]: input feature data, [head, win, channels]
  @args[1]: activate weight, [16x3]
  @output: Tensor, [head, win, channels]
'''
def ActivateRel(args, attrs):
    if len(args) not in [2]:
        return False, "error length arguments! support 2, found " + str(len(args))
    device = args[0].device
    dtype = args[0].dtype
    dshape, wshape = args[0].shape, args[1].shape
    oshape = [i for i in dshape]
    return True, Tensor(oshape, dtype, device)

Op.Register("nn.activate", ActivateRel)


'''
@brief: KVCache2HBM op
  @args[0]: input feature data, [head, win, channels]
  @output: Tensor, [head, win+cache, channels]
'''
def Kvcache2hbmRel(args, attrs):
    if len(args) not in [1]:
        return False, "error length arguments! support 1, found " + str(len(args))
    device = args[0].device
    dtype = args[0].dtype
    dshape = args[0].shape
    if dshape[-1] not in [128]:
        return False, "error head per channels! support 128, found " + str(dshape[-1])
    oshape = [i for i in dshape]
    oshape[1] = oshape[1] + attrs.get("cache_len")
    tensor = Tensor(oshape, dtype, device)
    tensor.bytesize = attrs.get("cache_size") * dshape[-1] * 16 // 8 * dshape[0] // device.HBM_Port
    WT_HEAD_STRIDE = ((device.MAX_DAT_DW*dshape[-1]//8)*attrs.get("cache_size")//device.HBM_Port)
    WT_LINE_STRIDE = ((device.MAX_DAT_DW*device.Tout//8)*attrs.get("cache_size")//device.HBM_Port)
    setattr(tensor, "strides", [WT_HEAD_STRIDE, WT_LINE_STRIDE, WT_LINE_STRIDE])
    return True, tensor

Op.Register("nn.kvcache2hbm", Kvcache2hbmRel)


'''
@brief: RoPE op
  @args[0]: input feature data, [Head, token, chin]
  @args[1]: processed data, [x]
  @output: Tensor, [Head, token, chin]
'''
def RoPosEmbRel(args, attrs):
    if len(args) not in [2]:
        return False, "error length arguments! support 1, found " + str(len(args))
    device = args[0].device
    dtype = args[0].dtype
    dshape = args[0].shape
    if dshape[-1] not in [128]:
        return False, "error head per channels! support 128, found " + str(dshape[-1])
    oshape = [i for i in dshape]
    tensor = Tensor(oshape, dtype, device)
    if hasattr(args[0], "heads"):
        setattr(tensor, "heads", args[0].heads)
    return True, tensor

Op.Register("nn.rope", RoPosEmbRel)


'''
@brief: Conv2d op
  @args[0]: input feature data, [Hin, Win, CHin]
  @args[1]: processed data, [Ky, Kx, CHin, CHout]
  @output: Tensor, [Head, token, chin]
'''
def Conv2dRel(args, attrs):
    if len(args) not in [3]:
        return False, "error length arguments! support [3], found " + str(len(args))
    device = args[0].device
    dtype = args[0].dtype
    dshape = args[0].shape
    wshape = args[1].shape
    Hin, Win, CHin = dshape
    Ky, Kx, wCHin, CHout = wshape
    if CHin != wCHin:
        return False, f"CHin in weight({wCHin}) should same with CHin in data({CHin})"
    Py, Px = attrs.get("padding")
    Sy, Sx = attrs.get("strides")
    Wout = ((Win+2*Px-Kx)//Sx+1)
    Hout = ((Hin+2*Py-Ky)//Sy+1)
    oshape = [Hout, Wout, CHout]
    tensor = Tensor(oshape, dtype, device)
    return True, tensor

Op.Register("nn.conv2d", Conv2dRel)

