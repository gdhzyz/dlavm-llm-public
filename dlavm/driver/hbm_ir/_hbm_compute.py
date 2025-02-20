from dlavm import ne
from dlavm.adr import Tensor, Op
from .. import ir
from ..basic import Ceil
from .tasks_0922 import *


def get_vars(targets):
    vars = []
    func = lambda n : [i for i in n if i not in vars]
    for n in targets:
        if isinstance(n, list):
            vars += func(get_vars(n))
        elif isinstance(n, dict):
            vars += func(get_vars(n.values()))
        elif isinstance(n, ne.Expr):
            vars += func(n.get_vars(True))
    return vars


def replace(list, num, target):
    new_list = [i for i in list]
    new_list[num] = target
    return new_list


def MVM(args, outputs, full=None, **attrs):
    device = args[0].device
    Hin, Win = 1, args[0].shape[-2]
    CHin, CHout = args[1].shape
    PixelBytes, Tin = device.Pixel_Data_Bytes, device.base_Tin
    DAT_BRAM_DEPTH = device.DAT_BRAM_DEPTH
    strides = [Hin*Win*PixelBytes, Win*PixelBytes, PixelBytes]
    if hasattr(args[0], "strides"):
        strides = args[0].strides
    WT_CHin_div_Tin = Ceil(CHin, Tin)
    if DAT_BRAM_DEPTH < WT_CHin_div_Tin:
        raise RuntimeError("Could not split now for MVM")
    w_slice = DAT_BRAM_DEPTH // WT_CHin_div_Tin
    if full is None:
        full = False
        if isinstance(Win, ne.Expr) and not isinstance(Win, ne.Numb):
            if Win.simplify(1) > w_slice:
                full = True
    with ir.Function(get_vars([args[0].shape, attrs])) as func:
        out_w_slice            = w_slice
        min_dat_depth          = Win*WT_CHin_div_Tin
        Wout_Split_Times_minus1= (min_dat_depth+(out_w_slice*WT_CHin_div_Tin)-1)//(out_w_slice*WT_CHin_div_Tin) -1
        Wout_Split_Times_minus1= func.assign("wout_split_times_minus1", Wout_Split_Times_minus1, "int")
        Wout_Split_Times       = Wout_Split_Times_minus1 + 1
        out_w_slice_last       = (min_dat_depth-(Wout_Split_Times_minus1)*out_w_slice*WT_CHin_div_Tin)//WT_CHin_div_Tin
        out_w_slice_last       = func.assign("out_w_slice_last", out_w_slice_last, "int")
        with ir.For("out_w", 0, Wout_Split_Times) as w:
            w_slice = ne.If(w.var < Wout_Split_Times_minus1, out_w_slice, out_w_slice_last)
            dshape = replace(args[0].shape, -2, w_slice)
            data = args[0].gen_tensor(shape=dshape, offset=w.var*out_w_slice*PixelBytes+args[0].offset)
            setattr(data, "strides", strides)
            oshape = replace(outputs[0].shape, -2, w_slice)
            output = Tensor(oshape, outputs[0].dtype, device)
            output = outputs[0].gen_tensor(shape=oshape, offset=w.var*out_w_slice*PixelBytes+outputs[0].offset)
            setattr(output, "strides", strides)
            Tasks.Get("atom.hbm.mvm", device)(w, replace(args, 0, data), replace(outputs, 0, output), device, **attrs)
        func += w
    return func


@Op.RegisterAttrs("accel.hbm.layer_norm", "compute")
def LayerNorm(args, outputs, **attrs):
    device = args[0].device
    with ir.Function(get_vars([args[0].shape, attrs])) as func:
        new_args = [arg for arg in args]
        if attrs.get("kvcache_token", 0):
            PixelBytes = device.Pixel_Data_Bytes
            Hin, Win, CHin = args[0].shape
            dshape = replace(args[0].shape, -2, 1)
            in_strides = [Hin*Win*PixelBytes, Win*PixelBytes, PixelBytes]
            data = args[0].gen_tensor(shape=dshape, offset=(Win-1)*PixelBytes+args[0].offset)
            setattr(data, "strides", in_strides)
            new_args = replace(new_args, 0, data)
        Tasks.Get("atom.hbm.layer_norm", device)(func, new_args, outputs, device, **attrs)
    return func


@Op.RegisterAttrs("accel.hbm.pos_emb", "compute")
def PosEmb(args, outputs, **attrs):
    device = args[0].device
    with ir.Function(get_vars([args[0].shape, attrs])) as func:
        Tasks.Get("atom.hbm.pos_emb", device)(func, args, outputs, device, **attrs)
    return func


@Op.RegisterAttrs("accel.hbm.softmax", "compute")
def Softmax(args, outputs, **attrs):
    device = args[0].device
    with ir.Function(get_vars([args[0].shape, attrs])) as func:
        Tasks.Get("atom.hbm.softmax", device)(func, args, outputs, device, **attrs)
    return func


@Op.RegisterAttrs("accel.hbm.activate", "compute")
def Act(args, outputs, **attrs):
    device = args[0].device
    with ir.Function(get_vars([args[0].shape, attrs])) as func:
        Tasks.Get("atom.hbm.act", device)(func, args, outputs, device, **attrs)
    return func


@Op.RegisterAttrs("accel.hbm.dat_hbm", "compute")
def Dat2HBM(args, outputs, **attrs):
    device = args[0].device
    with ir.Function(get_vars([args[0].shape, attrs])) as func:
        Tasks.Get("atom.hbm.dat2hbm", device)(func, args, outputs, device, **attrs)
    return func


@Op.RegisterAttrs("accel.hbm.trp_mvm", "compute")
def TRP_MVM(args, outputs, **attrs):
    device = args[0].device
    PixelBytes, Tout = device.Pixel_Data_Bytes, device.Tout
    s_DAT_BRAM_DEPTH, s_DAT_BRAM_NUM = device.s_DAT_BRAM_DEPTH, device.s_DAT_BRAM_NUM
    s_WT_BRAM_DEPTH, s_WT_BRAM_NUM = device.s_WT_BRAM_DEPTH, device.s_WT_BRAM_NUM
    s_Tin, MAX_DAT_DW, s_MAX_WT_DW = device.s_Tin, device.MAX_DAT_DW, device.MAX_WT_DW
    MAX_TOKEN, MAX_CH_per_HEAD = device.MAX_TOKEN, device.MAX_CH_per_HEAD

    F_Head, Hin, Win = args[0].shape[0], 1, args[0].shape[-2]
    W_Head, CHout, CHin = args[1].shape[0], args[1].shape[-2], args[1].shape[-1]
    F_Head, Hout, Wout = outputs[0].shape[0], 1, outputs[0].shape[-2]
    CHin_div_Tout, CHout_div_Tout = Ceil(CHin, Tout), Ceil(CHout, Tout)
    in_strides = [CHin_div_Tout*Hin*Win*PixelBytes, Hin*Win*PixelBytes, Win*PixelBytes, PixelBytes]
    out_strides = [CHout_div_Tout*Hin*Win*PixelBytes, Hin*Win*PixelBytes, Win*PixelBytes, PixelBytes]
    if hasattr(args[0], "strides"):
        in_strides = args[0].strides
    if hasattr(outputs[0], "strides"):
        out_strides = outputs[0].strides

    Onchip_Dat_BRAM_Bits    =s_DAT_BRAM_NUM*s_DAT_BRAM_DEPTH*s_Tin*MAX_DAT_DW
    Total_Dat_Bits_PerToken =MAX_CH_per_HEAD*(F_Head//W_Head)*MAX_DAT_DW
    Onchip_Token_perWTHead  =Onchip_Dat_BRAM_Bits//Total_Dat_Bits_PerToken

    out_w_per_slice        =Onchip_Token_perWTHead
    Wout_Split_Times_minus1=(Wout+Onchip_Token_perWTHead-1)//Onchip_Token_perWTHead -1
    out_w_in_last_slice    =Wout-(Wout_Split_Times_minus1)*out_w_per_slice

    Onchip_Wt_BRAM_Bits    =s_WT_BRAM_NUM*s_WT_BRAM_DEPTH*s_Tin*s_MAX_WT_DW
    Total_Wt_Bits_perWTHead=MAX_TOKEN*MAX_CH_per_HEAD*s_MAX_WT_DW
    
    if Total_Wt_Bits_perWTHead>Onchip_Wt_BRAM_Bits:
        raise RuntimeError("================ FPGA s_WT BRAM DEPTH not enough for MVM_afterTRP!  ====================");

    with ir.Function(get_vars([args[0].shape, attrs])) as func:
        out_w_slice            = out_w_per_slice
        Wout_Split_Times_minus1= func.assign("wout_split_times_minus1", Wout_Split_Times_minus1, "int")
        Wout_Split_Times       = Wout_Split_Times_minus1 + 1
        out_w_slice_last       = func.assign("out_w_slice_last", out_w_in_last_slice, "int")
        with ir.For("out_w", 0, Wout_Split_Times) as w:
            w_slice = ne.If(w.var < Wout_Split_Times_minus1, out_w_slice, out_w_slice_last)
            dshape = replace(args[0].shape, -2, w_slice)
            data = args[0].gen_tensor(shape=dshape, offset=w.var*out_w_slice*PixelBytes+args[0].offset)
            setattr(data, "strides", in_strides)
            oshape = replace(outputs[0].shape, -2, w_slice)
            output = Tensor(oshape, outputs[0].dtype, device)
            output = outputs[0].gen_tensor(shape=oshape, offset=w.var*out_w_slice*PixelBytes+outputs[0].offset)
            setattr(output, "strides", out_strides)
            Tasks.Get("atom.hbm.trp_mvm", device)(w, replace(args, 0, data), replace(outputs, 0, output), device, **attrs)
        func += w
    return func


@Op.RegisterAttrs("accel.hbm.f2w_mvm", "compute")
def F2W_MVM(args, outputs, **attrs):
    device = args[0].device
    PixelBytes, Tout = device.Pixel_Data_Bytes, device.Tout
    s_DAT_BRAM_DEPTH, s_DAT_BRAM_NUM = device.s_DAT_BRAM_DEPTH, device.s_DAT_BRAM_NUM
    s_WT_BRAM_DEPTH, s_WT_BRAM_NUM = device.s_WT_BRAM_DEPTH, device.s_WT_BRAM_NUM
    s_Tin, MAX_DAT_DW, s_MAX_WT_DW = device.s_Tin, device.MAX_DAT_DW, device.MAX_WT_DW
    MAX_TOKEN, MAX_CH_per_HEAD = device.MAX_TOKEN, device.MAX_CH_per_HEAD

    F_Head, Hin, Win = args[0].shape[0], 1, args[0].shape[-2]
    W_Head, CHin, CHout = args[1].shape[0], args[1].shape[-2], args[1].shape[-1]
    F_Head, Hout, Wout = outputs[0].shape[0], 1, outputs[0].shape[-2]
    CHin_div_Tout, CHout_div_Tout = Ceil(CHin, Tout), Ceil(CHout, Tout)
    CHin_Padding = Ceil(CHin, s_Tin) * s_Tin
    in_strides = [CHin_div_Tout*Hin*Win*PixelBytes, Hin*Win*PixelBytes, Win*PixelBytes, PixelBytes]
    out_strides = [CHout_div_Tout*Hin*Win*PixelBytes, Hin*Win*PixelBytes, Win*PixelBytes, PixelBytes]
    if hasattr(args[0], "strides"):
        in_strides = args[0].strides
    if hasattr(outputs[0], "strides"):
        out_strides = outputs[0].strides

    Onchip_Dat_BRAM_Bits    =s_DAT_BRAM_NUM*s_DAT_BRAM_DEPTH*s_Tin*MAX_DAT_DW
    Total_Dat_Bits_PerToken =CHin_Padding*(F_Head//W_Head)*MAX_DAT_DW
    Onchip_Token_perWTHead  =Onchip_Dat_BRAM_Bits//Total_Dat_Bits_PerToken



    Onchip_Wt_BRAM_Bits    =s_WT_BRAM_NUM*s_WT_BRAM_DEPTH*s_Tin*s_MAX_WT_DW
    Total_Wt_Bits_perWTHead=MAX_TOKEN*MAX_CH_per_HEAD*s_MAX_WT_DW
    
    if Total_Wt_Bits_perWTHead>Onchip_Wt_BRAM_Bits:
        raise RuntimeError("================ FPGA s_WT BRAM DEPTH not enough for MVM_afterF2W!  ====================");

    with ir.Function(get_vars([args[0].shape, attrs])) as func:
        out_w_slice            = func.assign("out_w_per_slice", Onchip_Token_perWTHead, "int")
        Wout_Split_Times_minus1=(Wout+out_w_slice-1)//out_w_slice -1
        out_w_in_last_slice    = Wout-(Wout_Split_Times_minus1)*out_w_slice
        Wout_Split_Times_minus1= func.assign("wout_split_times_minus1", Wout_Split_Times_minus1, "int")
        Wout_Split_Times       = Wout_Split_Times_minus1 + 1
        out_w_slice_last       = func.assign("out_w_slice_last", out_w_in_last_slice, "int")
        with ir.For("out_w", 0, Wout_Split_Times) as w:
            w_slice = ne.If(w.var < Wout_Split_Times_minus1, out_w_slice, out_w_slice_last)
            dshape = replace(args[0].shape, -2, w_slice)
            data = args[0].gen_tensor(shape=dshape, offset=w.var*out_w_slice*PixelBytes+args[0].offset)
            setattr(data, "strides", in_strides)
            oshape = replace(outputs[0].shape, -2, w_slice)
            output = Tensor(oshape, outputs[0].dtype, device)
            output = outputs[0].gen_tensor(shape=oshape, offset=w.var*out_w_slice*PixelBytes+outputs[0].offset)
            setattr(output, "strides", out_strides)
            Tasks.Get("atom.hbm.f2w_mvm", device)(w, replace(args, 0, data), replace(outputs, 0, output), device, **attrs)
        func += w
    return func
