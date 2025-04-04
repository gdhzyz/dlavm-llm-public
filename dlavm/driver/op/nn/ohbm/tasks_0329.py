import math
from dlavm import ne
from dlavm.adr import Op, Attrs
from dlavm.device import ohbm_accel
from dlavm.clib import FP32_to_FP20
from .... import ir
from ....ir import CSB_Write, CSB_Read, While
from ....basic import Tasks, Ceil, Ceil_Padding


class MVMMode:

    norm = 0b001111
    bn = 0b011111
    bn_argmax = 0b111111

    device = ohbm_accel.OHBM
    Py, Px = 0, 0
    Sy, Sx = 1, 1
    Ky, Kx = 1, 1
    reg_19_y = Py + (Sy << (device.log2_P)) + (Ky << (device.log2_P + device.log2_S))
    reg_20_x = Px + (Sx << (device.log2_P)) + (Kx << (device.log2_P + device.log2_S))


#########################################################################################
#                                nn.mvm_f16xf16 compute task                            #
#########################################################################################
@Tasks.Register("ohbm.nn.mvm_f16xf16", ohbm_accel.OHBM0329)
def MVMF16xF16(func, args, outputs, attrs):
    device = args[0].device

    mode = MVMMode.norm
    data, weight = args
    
    w_trp = attrs.get("w_trp")
    # macro define testbench
    WT_DW = device.MAX_WT_DW
    Tin = device.s_Tin
    Sparsity_Factor = 1
    Feature_Head, Height, Width_in = data.shape[-3:]
    Weight_Head = weight.shape[0]
    Padding_Feature_Head = None
    if hasattr(data, "heads"):
        Feature_Head = data.heads[0]
        Padding_Feature_Head = data.heads[2]
    if w_trp:
        Width_out = weight.shape[-2]
        MAX_CH_per_HEAD = Width_in
    else:
        Width_out = weight.shape[-1]
        MAX_CH_per_HEAD = Width_out
    feature_in_addr = data.get_address()
    wt_base_addr = weight.get_address()
    feature_out_addr = outputs[0].get_address()
    # pre process
    Tb = 1
    Hin, Win = 1, Height
    CHin = Ceil_Padding(Width_in, device.Tout)
    Hout, Wout = 1, Height
    CHout = Ceil_Padding(Width_out, device.Tout)
    CHin_div_LTout = Ceil(CHin, device.L_Tout)
    CHin_Padding = CHin_div_LTout * device.L_Tout
    CHout_div_Tout = Ceil(CHout, device.Tout)
    CHout_div_LTout = Ceil(CHout, device.L_Tout)
    CHin_Padding_with_LTout = CHin_Padding

    Head_x_CHin           = Feature_Head//Weight_Head*CHin
    Head_x_CHin_div_LTout = (Head_x_CHin+device.L_Tout-1)//device.L_Tout
    LTout_div_CHin        = device.L_Tout//MAX_CH_per_HEAD # MAX_CH_per_HEAD
    Head_x_CH_div_LTout   = (Feature_Head//Weight_Head*MAX_CH_per_HEAD+device.L_Tout-1)//device.L_Tout
    if Padding_Feature_Head is None:
        Feature_Head_in_Padding = (Head_x_CHin_div_LTout*device.L_Tout//CHin)
        Padding_Feature_Head    = (Feature_Head_in_Padding*Weight_Head)

    WT_CHin                 = device.MAX_TOKEN
    WT_CHout                = MAX_CH_per_HEAD
    WT_CHin_div_Tin = Ceil(CHin, Tin)
    WT_CHin_Padding_with_Tin = WT_CHin_div_Tin*Tin
    WT_CHout_Padding_with_Tout = Ceil(WT_CHout, device.Tout)*device.Tout

    WT_BYTES_PER_HEAD       = (WT_CHout_Padding_with_Tout * WT_CHin_Padding_with_Tin * device.MAX_DAT_DW // 8)
    WT_BYTES_PER_CHOUT      = (WT_BYTES_PER_HEAD//WT_CHout_Padding_with_Tout)

    # stride load or compute
    feature_in_line_stride = device.HBM_1Row_Bytes * Win
    feature_in_head_stride = device.HBM_1Row_Bytes * Win * Hin * CHin_div_LTout
    WT_HEAD_STRIDE          = ((device.MAX_DAT_DW*WT_CHout//8)*device.MAX_TOKEN//device.HBM_Port)
    WT_LINE_STRIDE          = ((device.MAX_DAT_DW*device.Tout//8)*device.MAX_TOKEN//device.HBM_Port)
    feature_out_line_stride = device.HBM_1Row_Bytes * Wout
    feature_out_head_stride = device.HBM_1Row_Bytes * Wout * Hout * CHout_div_LTout
    if hasattr(data, "strides"):
        feature_in_line_stride = data.strides[-1]
        feature_in_head_stride = data.strides[-3]
    if hasattr(weight, "strides"):
        WT_LINE_STRIDE = weight.strides[-1]
        WT_HEAD_STRIDE = weight.strides[-3]
    if hasattr(outputs[0], "strides"):
        feature_out_line_stride = outputs[0].strides[-1]
        feature_out_head_stride = outputs[0].strides[-3]

    # loop define
    dat_bits_per_row=Win*CHin_Padding*device.MAX_DAT_DW
    min_dat_depth   =dat_bits_per_row//(device.HBM_AXI_DATA_WIDTH*device.HBM_Port)
    wt_bits_per_Tout=WT_CHin_Padding_with_Tin*device.Tout*device.MAX_WT_DW
    min_wt_depth    =wt_bits_per_Tout//(device.HBM_AXI_DATA_WIDTH*device.HBM_Port*device.ASYN_FACTOR)

    if w_trp:
        dat_bits_per_row_all_Head=Win*(Head_x_CHin_div_LTout*device.L_Tout*device.MAX_DAT_DW)
    else:
        dat_bits_per_row_all_Head=Win*(CHin_Padding_with_LTout*device.MAX_DAT_DW*Feature_Head)
    min_dat_depth            =dat_bits_per_row_all_Head//(device.HBM_AXI_DATA_WIDTH*device.HBM_Port)
    wt_bits_per_Tout_per_Head=device.Tout*WT_CHin_Padding_with_Tin*device.MAX_DAT_DW
    min_wt_depth             =wt_bits_per_Tout_per_Head//(device.HBM_AXI_DATA_WIDTH*device.HBM_Port*device.ASYN_FACTOR)

    '''
    if min_wt_depth>device.ID1_BRAM_DEPTH:
        print("ohbm.nn.mvm_f16xf16 driver task error:")
        print("=======================================================================")
        print("================ FPGA WT BRAM DEPTH not enough!    ====================")
        print("=======================================================================")
        exit(-1)
    '''

    # ir: create local variable
    min_dat_depth = func.assign("min_dat_depth", min_dat_depth, "int")

    Wout_Split_Times_minus1 = func.assign("Wout_Split_Times_minus1", 0, "int")
    out_w_slice             = func.assign("out_w_slice", Wout, "int")
    out_w_slice_last        = func.assign("out_w_slice_last", Wout, "int")
    t_if = ir.If(min_dat_depth>device.ID0_BRAM_DEPTH)
    with t_if.then_block as _then:
        if w_trp:
            out_w_slice            =_then.assign_var(out_w_slice, device.TOTAL_DAT_BRAM_BITS//(Head_x_CHin_div_LTout*device.L_Tout*device.MAX_DAT_DW))
        else:
            out_w_slice            =_then.assign_var(out_w_slice, device.TOTAL_DAT_BRAM_BITS//(CHin_Padding_with_LTout*device.MAX_DAT_DW*Feature_Head))
        out_w_slice_last       =_then.assign_var(out_w_slice_last, ne.If(Wout%out_w_slice, Wout%out_w_slice, out_w_slice))
        Wout_Split_Times_minus1=_then.assign_var(Wout_Split_Times_minus1, (Wout+out_w_slice-1)//out_w_slice-1)
    func += t_if

    out_ch_slice=(device.TOTAL_WT_BRAM_BITS//wt_bits_per_Tout_per_Head)*device.Tout

    if isinstance(out_ch_slice, ne.Expr):
        tp_out_ch_slice=ne.Numb(1) << (out_ch_slice.log2().cast_int())
        tp_out_ch_slice = func.assign("tp_log_out_ch_slice", tp_out_ch_slice, "int")
        # tp_out_ch_slice=1
        out_ch_slice = ne.If(tp_out_ch_slice>out_ch_slice, tp_out_ch_slice//2, tp_out_ch_slice)
        out_ch_slice = func.assign("tp_out_ch_slice", out_ch_slice, "int")
    else:
        tp_out_ch_slice=1<<(int(math.log2(out_ch_slice)))
        if tp_out_ch_slice>out_ch_slice:
            out_ch_slice=tp_out_ch_slice//2
        else:
            out_ch_slice=tp_out_ch_slice
    
    out_ch_slice = ne.If(out_ch_slice>device.MAX_BN_CH, device.MAX_BN_CH, out_ch_slice)
    out_ch_slice = ne.If(out_ch_slice<CHout, out_ch_slice, CHout)
    out_ch_slice = func.assign("out_ch_slice", out_ch_slice, "int")
    CHout_Split_Times = ne.If(out_ch_slice<CHout, (CHout+out_ch_slice-1)//out_ch_slice, 1)
    CHout_Split_Times = func.assign("CHout_Split_Times", CHout_Split_Times, "int")
    out_ch_slice_last = ne.If(CHout%out_ch_slice, CHout%out_ch_slice, out_ch_slice)
    out_ch_slice_last = func.assign("out_ch_slice_last", out_ch_slice_last, "int")

    '''
    if out_ch_slice>device.MAX_BN_CH:
        out_ch_slice=device.MAX_BN_CH
    if out_ch_slice>=CHout:
        out_ch_slice=CHout
        CHout_Split_Times=1
    else:
        CHout_Split_Times=(CHout+out_ch_slice-1)//out_ch_slice
    if CHout%out_ch_slice:
        out_ch_slice_last=CHout%out_ch_slice
    else:
        out_ch_slice_last=out_ch_slice
    '''
        
    L_Tout_DAT_DW = device.L_Tout*device.MAX_DAT_DW
    CHout_Split_Times_minus1=CHout_Split_Times-1
    total_clks_if_reuse_wt=Tb*(Head_x_CHin_div_LTout*L_Tout_DAT_DW)*Win*(CHout_Split_Times_minus1+1)//(device.HBM_AXI_DATA_WIDTH*device.HBM_Port) \
                          +WT_BYTES_PER_HEAD//(device.HBM_AXI_DATA_WIDTH*device.HBM_Port)+20
    total_clks_if_reuse_dat=Tb*(Head_x_CHin_div_LTout*L_Tout_DAT_DW)*Win//(device.HBM_AXI_DATA_WIDTH*device.HBM_Port) \
                          +WT_BYTES_PER_HEAD*(Wout_Split_Times_minus1+1)//(device.HBM_AXI_DATA_WIDTH*device.HBM_Port)+20
    total_clks_if_reuse_wt =func.assign("total_clks_if_reuse_wt", total_clks_if_reuse_wt, "int")
    total_clks_if_reuse_dat=func.assign("total_clks_if_reuse_dat", total_clks_if_reuse_dat, "int")

    task = Tasks.Get("atom.ohbm.nn.mvm_f16xf16", device)

    reg_17 = 4
    reg_18 = FP32_to_FP20(1.0)
    reg_26 = WT_LINE_STRIDE
    reg_27 = (Head_x_CH_div_LTout << 8) + LTout_div_CHin
    reg_28 = ne.If(LTout_div_CHin > 1, 2, 0)
    H_in_now = Feature_Head // Weight_Head
    H_out_now = H_in_now
    if attrs.get("w_trp"):
        reg_17 = 3
        reg_18 = FP32_to_FP20(1/math.sqrt(Width_in))
        reg_26 = 0
        reg_28 = ne.If(LTout_div_CHin >= 1, 1, 0)
        H_in_now = Padding_Feature_Head // Weight_Head
        H_out_now = Feature_Head // Weight_Head

    t_if = ir.If(total_clks_if_reuse_wt < total_clks_if_reuse_dat)
    with t_if.then_block as _then:
        with ir.For("h", 0, Weight_Head, 1) as h_for:
            h = h_for.var
            with ir.For("ch", 0, CHout_Split_Times_minus1+1, 1) as ch_for:
                ch = ch_for.var
                with ir.For("w", 0, Wout_Split_Times_minus1+1, 1) as w_for:
                    w = w_for.var
                    CH_in_now = CHin
                    CH_out_offset = ch*out_ch_slice
                    W_in_offset = w*out_w_slice
                    W_out_offset = w*out_w_slice
                    CH_out_now = ne.If(ch < CHout_Split_Times_minus1, out_ch_slice, out_ch_slice_last)
                    dma_dat_reuse_now = 0
                    dma_wt_reuse_now = ne.If(w, 1, 0)
                    W_in_now = ne.If(w < Wout_Split_Times_minus1, out_w_slice, out_w_slice_last)
                    W_out_now = ne.If(w < Wout_Split_Times_minus1, out_w_slice, out_w_slice_last)
                    tp_in_addr =feature_in_addr+device.HBM_1Row_Bytes*W_in_offset+h*feature_in_head_stride*(Feature_Head//Weight_Head)
                    tp_wt_addr =wt_base_addr+WT_LINE_STRIDE*ch*out_ch_slice//device.Tout+h*WT_HEAD_STRIDE
                    tp_out_addr=feature_out_addr+device.HBM_1Row_Bytes*W_out_offset+h*feature_out_head_stride*Head_x_CH_div_LTout
                    if w_trp:
                        tp_in_addr =feature_in_addr+device.HBM_1Row_Bytes*W_in_offset+h*feature_in_head_stride*Head_x_CH_div_LTout
                        tp_wt_addr =wt_base_addr+WT_BYTES_PER_CHOUT*ch*out_ch_slice//device.Tout+h*WT_HEAD_STRIDE
                        tp_out_addr=feature_out_addr+device.HBM_1Row_Bytes*W_out_offset+(device.HBM_1Row_Bytes*out_w_slice)*(ch*out_ch_slice//device.L_Tout) \
                                    +h*feature_out_head_stride*(Feature_Head//Weight_Head)
                    task(w_for, CH_in_now, H_in_now, W_in_now, CH_out_offset, CH_out_now, H_out_now, W_out_now,
                         dma_wt_reuse_now, dma_dat_reuse_now, reg_17, reg_18, reg_26, reg_27, reg_28,
                         tp_in_addr, feature_in_head_stride, feature_in_line_stride,
                         tp_wt_addr, WT_BYTES_PER_CHOUT*8,
                         tp_out_addr, feature_out_head_stride, feature_out_line_stride, mode, device)
                ch_for += w_for
            h_for += ch_for
        _then += h_for
    with t_if.else_block as _else:
        with ir.For("h", 0, Weight_Head, 1) as h_for:
            h = h_for.var
            with ir.For("w", 0, Wout_Split_Times_minus1+1, 1) as w_for:
                w = w_for.var
                with ir.For("ch", 0, CHout_Split_Times_minus1+1, 1) as ch_for:
                    ch = ch_for.var
                    CH_in_now = CHin
                    CH_out_offset = ch*out_ch_slice
                    W_in_offset = w*out_w_slice
                    W_out_offset = w*out_w_slice
                    CH_out_now = ne.If(ch < CHout_Split_Times_minus1, out_ch_slice, out_ch_slice_last)
                    dma_wt_reuse_now = 0
                    dma_dat_reuse_now = ne.If(ch, 1, 0)
                    W_in_now = ne.If(w < Wout_Split_Times_minus1, out_w_slice, out_w_slice_last)
                    W_out_now = ne.If(w < Wout_Split_Times_minus1, out_w_slice, out_w_slice_last)
                    tp_in_addr =feature_in_addr+device.HBM_1Row_Bytes*W_in_offset+h*feature_in_head_stride*(Feature_Head//Weight_Head)
                    tp_wt_addr =wt_base_addr+WT_LINE_STRIDE*ch*out_ch_slice//device.Tout+h*WT_HEAD_STRIDE
                    tp_out_addr=feature_out_addr+device.HBM_1Row_Bytes*W_out_offset+h*feature_out_head_stride*Head_x_CH_div_LTout
                    if w_trp:
                        tp_in_addr =feature_in_addr+device.HBM_1Row_Bytes*W_in_offset+h*feature_in_head_stride*Head_x_CH_div_LTout
                        tp_wt_addr =wt_base_addr+WT_BYTES_PER_CHOUT*ch*out_ch_slice//device.Tout+h*WT_HEAD_STRIDE
                        tp_out_addr=feature_out_addr+device.HBM_1Row_Bytes*W_out_offset+(device.HBM_1Row_Bytes*out_w_slice)*(ch*out_ch_slice//device.L_Tout) \
                                    +h*feature_out_head_stride*(Feature_Head//Weight_Head)
                    task(ch_for, CH_in_now, H_in_now, W_in_now, CH_out_offset, CH_out_now, H_out_now, W_out_now,
                         dma_wt_reuse_now, dma_dat_reuse_now, reg_17, reg_18, reg_26, reg_27, reg_28,
                         tp_in_addr, feature_in_head_stride, feature_in_line_stride,
                         tp_wt_addr, WT_BYTES_PER_CHOUT*8,
                         tp_out_addr, feature_out_head_stride, feature_out_line_stride, mode, device)
                w_for += ch_for
            h_for += w_for
        _else += h_for
    func += t_if


#########################################################################################
#                               nn.kvcache2hbm compute task                             #
#########################################################################################
@Tasks.Register("ohbm.nn.kvcache2hbm", ohbm_accel.OHBM0329)
def Kvcache2hbm(func, args, outputs, attrs):
    device = args[0].device
    data = args[0]

    # macro define testbench
    WT_DW = device.MAX_WT_DW
    Sparsity_Factor = 1
    Weight_Head, Height, Width_in = data.shape[-3:]
    Width_out = Width_in
    last_token = attrs.get("cache_len")
    feature_in_base_addr = data.get_address()
    feature_out_base_addr = outputs[0].get_address()
    MAX_TOKEN = attrs.get("cache_size")
    MAX_CH_per_HEAD = Width_in

    # pre process
    WT_CHout = Width_in
    Tb = 1
    Hin, Win = 1, Height
    CHin = Ceil_Padding(Width_in, device.Tout)
    Hout, Wout = 1, Height
    CHout = Ceil_Padding(Width_out, device.Tout)
    CHin_div_LTout = Ceil(CHin, device.L_Tout)
    CHin_Padding = CHin_div_LTout * device.L_Tout
    CHout_div_Tout = Ceil(CHout, device.Tout)
    CHin_Padding_with_LTout = CHin_Padding
    LTout_div_CHin          = device.L_Tout//MAX_CH_per_HEAD
    Head_x_CHin             = (Weight_Head*MAX_CH_per_HEAD)
    Head_x_CHin_div_LTout   = ((Head_x_CHin+device.L_Tout-1)//device.L_Tout)

    # stride load or compute
    feature_in_line_stride = device.HBM_1Row_Bytes * Win
    feature_in_head_stride = device.HBM_1Row_Bytes * Win * Hin * CHin_div_LTout
    feature_out_line_stride = ((device.MAX_DAT_DW*device.Tout//8)*MAX_TOKEN//device.HBM_Port)
    feature_out_head_stride = ((device.MAX_DAT_DW*WT_CHout//8)*MAX_TOKEN//device.HBM_Port)
    if hasattr(data, "strides"):
        feature_in_line_stride = data.strides[-1]
        feature_in_head_stride = data.strides[-3]
    if hasattr(outputs[0], "strides"):
        feature_out_line_stride = outputs[0].strides[-1]
        feature_out_head_stride = outputs[0].strides[-3]

    # task function
    mode = 1 if attrs.get("k_mode") else 2
    kvcache2hbm_reg_bias=128
    reg_15 = (Head_x_CHin_div_LTout << 8) + LTout_div_CHin

    en_log2addr = 0xff00000000
    feature_in_addr = func.assign("feature_in_addr", feature_in_base_addr, "uint64_t")
    feature_out_addr = func.assign("feature_out_addr", feature_out_base_addr, "uint64_t")
    in_addr_low32bits = ir.Cast(feature_in_addr, "uint32_t")
    out_addr_low32bits = ir.Cast(feature_out_addr, "uint32_t")
    addr_high32bits = ((feature_in_addr & en_log2addr) >> 32) + \
                        ((feature_out_addr & en_log2addr) >> 24) 

    func += CSB_Write(kvcache2hbm_reg_bias+2 , addr_high32bits           )
    func += CSB_Write(kvcache2hbm_reg_bias+3 , in_addr_low32bits         )
    func += CSB_Write(kvcache2hbm_reg_bias+4 , feature_in_head_stride    )
    func += CSB_Write(kvcache2hbm_reg_bias+5 , feature_in_line_stride    )
    func += CSB_Write(kvcache2hbm_reg_bias+6 , out_addr_low32bits        )
    func += CSB_Write(kvcache2hbm_reg_bias+7 , feature_out_head_stride   )
    func += CSB_Write(kvcache2hbm_reg_bias+8 , feature_out_line_stride   )
    func += CSB_Write(kvcache2hbm_reg_bias+9 , CHin                      )
    func += CSB_Write(kvcache2hbm_reg_bias+10, Hin                       )
    func += CSB_Write(kvcache2hbm_reg_bias+11, Win                       )
    func += CSB_Write(kvcache2hbm_reg_bias+12, Weight_Head               )
    func += CSB_Write(kvcache2hbm_reg_bias+13, last_token                )
    func += CSB_Write(kvcache2hbm_reg_bias+14, mode                      )
    func += CSB_Write(kvcache2hbm_reg_bias+15, reg_15                    )
    func += CSB_Write(kvcache2hbm_reg_bias+16, 0                         )
    func += CSB_Write(kvcache2hbm_reg_bias+17, 0b1000000                 )

    func += While(CSB_Read(kvcache2hbm_reg_bias+1)!=1) 


