import math
import numpy as np
from dlavm import ne
from ...clib import FP32_to_FP20
from ...device import hbm_accel
from ..basic import Tasks
from .. import ir
from ..ir import CSB_Write, CSB_Read, While

MVM_MODE = 0b00100011111
MVMBN_MODE = 0b01100011111
MVMBNARG_MODE = 0b101100011111
MVMBNRES_MODE = 0b11100011111
MVMBNRESARG_MODE = 0b111100011111


@Tasks.Register("atom.hbm.trp_mvm", hbm_accel.HBM1128)
def MVM_afterTRP_task(block, args, outputs, device, kvcache=0, last_token=0, **attrs):
    data, weight = args
    Token = data.shape[-2]
    Feature_Head = data.shape[0]
    Weight_Head = weight.shape[0]
    DAT_IN_BASE_ADDR = data.get_address()
    WT_BASE_ADDR = weight.get_address()
    DAT_OUT_BASE_ADDR = outputs[0].get_address()
    DAT_IN_ONCHIP = attrs.get("DAT_IN_ONCHIP")
    DAT_OUT_ONCHIP = attrs.get("DAT_OUT_ONCHIP")
    log2_WT_base_addr_Bank_Step = attrs.get("log2_WT_base_addr_Bank_Step", 28)

    Tin = device.base_Tin
    Tout = device.Tout
    Pixel_Data_Bytes = device.Pixel_Data_Bytes
    MAX_TOKEN = device.MAX_TOKEN
    MAX_CH_per_HEAD = device.MAX_CH_per_HEAD
    sCONV_OUT_DW = device.sCONV_OUT_DW

    Dynamic_Token = Token
    Total_Token = weight.shape[-2]
    Win = Dynamic_Token
    Hin = 1
    CHin = MAX_CH_per_HEAD
    CHout = Token + last_token
    Wout = Win
    Hout = Hin
    CHout_div_Tout = ((CHout + Tout - 1) // Tout)
    CHin_div_Tout = ((CHin + Tout - 1) // Tout)

    DAT_IN_LINE_STRIDE = Pixel_Data_Bytes * Win
    DAT_IN_HEAD_STRIDE = Pixel_Data_Bytes * Win * Hin * CHin_div_Tout
    WET_IN_LINE_STRIDE = Pixel_Data_Bytes * MAX_TOKEN
    WET_IN_HEAD_STRIDE = Pixel_Data_Bytes * MAX_TOKEN * ((MAX_CH_per_HEAD + Tout - 1) // Tout) 
    DAT_OUT_LINE_STRIDE = Pixel_Data_Bytes * Wout
    DAT_OUT_HEAD_STRIDE = Pixel_Data_Bytes * Wout * Hout * CHout_div_Tout
    if hasattr(args[0], "strides"):
        DAT_IN_LINE_STRIDE = args[0].strides[-2]
        DAT_IN_HEAD_STRIDE = args[0].strides[-4]
    if hasattr(args[1], "strides"):
        WET_IN_LINE_STRIDE = args[1].strides[-2]
        WET_IN_HEAD_STRIDE = args[1].strides[-4]
    if hasattr(outputs[0], "strides"):
        DAT_OUT_LINE_STRIDE = outputs[0].strides[-2]
        DAT_OUT_HEAD_STRIDE = outputs[0].strides[-4]

    # Hardware Testbench
    feature_in_base=DAT_IN_BASE_ADDR
    feature_out_base=DAT_OUT_BASE_ADDR
    Head_Cfg = (Feature_Head // Weight_Head - 1) * 256 * 256 + Feature_Head * 256 + Weight_Head
    reg_14 = FP32_to_FP20(1/math.sqrt(CHin)) + (log2_WT_base_addr_Bank_Step << sCONV_OUT_DW)
    reg_bias = 192

    onchip = 0
    if DAT_IN_ONCHIP is not None:
        onchip += 0b1
        feature_in_base = ne.If(kvcache, DAT_IN_ONCHIP, feature_in_base)
    if DAT_OUT_ONCHIP is not None:
        onchip += 0b10
        feature_out_base = ne.If(kvcache, DAT_OUT_ONCHIP, feature_out_base)

    if onchip:
        onchip = ne.If(kvcache, onchip, 0)
    block += CSB_Write(reg_bias+2 , WT_BASE_ADDR               )
    block += CSB_Write(reg_bias+3 , feature_in_base            )
    block += CSB_Write(reg_bias+4 , DAT_IN_HEAD_STRIDE         )
    block += CSB_Write(reg_bias+5 , DAT_IN_LINE_STRIDE         )
    block += CSB_Write(reg_bias+6 , feature_out_base           )
    block += CSB_Write(reg_bias+7 , DAT_OUT_HEAD_STRIDE        )
    block += CSB_Write(reg_bias+8 , DAT_OUT_LINE_STRIDE        )
    block += CSB_Write(reg_bias+9 , WET_IN_HEAD_STRIDE         )
    block += CSB_Write(reg_bias+10, WET_IN_LINE_STRIDE         )
    block += CSB_Write(reg_bias+11, CHin                       )
    block += CSB_Write(reg_bias+12, Total_Token                )
    block += CSB_Write(reg_bias+13, Dynamic_Token              )
    block += CSB_Write(reg_bias+14, reg_14                     )
    block += CSB_Write(reg_bias+15, Head_Cfg                   )
    block += CSB_Write(reg_bias+16, onchip                     )
    block += CSB_Write(reg_bias+17, 0b00_0010                  )
    block += While(CSB_Read(reg_bias+1) != 1)


@Tasks.Register("atom.hbm.f2w_mvm", hbm_accel.HBM1128)
def MVM_afterF2W_Task_0912(block, args, outputs, device, kvcache=0, last_token=0, **attrs):
    data, weight = args
    Token = data.shape[-2]
    Feature_Head = data.shape[0]
    Weight_Head = weight.shape[0]
    DAT_IN_BASE_ADDR = data.get_address()
    WT_BASE_ADDR = weight.get_address()
    DAT_OUT_BASE_ADDR = outputs[0].get_address()
    DAT_IN_ONCHIP = attrs.get("DAT_IN_ONCHIP")
    DAT_OUT_ONCHIP = attrs.get("DAT_OUT_ONCHIP")
    log2_WT_base_addr_Bank_Step = attrs.get("log2_WT_base_addr_Bank_Step", 28)

    Tout = device.Tout
    Pixel_Data_Bytes = device.Pixel_Data_Bytes
    MAX_TOKEN = device.MAX_TOKEN
    MAX_CH_per_HEAD = device.MAX_CH_per_HEAD
    sCONV_OUT_DW = device.sCONV_OUT_DW

    Win = Token
    Total_Token = weight.shape[-2]
    Hin = 1
    CHin = Token + last_token
    CHout = MAX_CH_per_HEAD
    Wout = Win
    Hout = Hin
    CHout_div_Tout = ((CHout + Tout - 1) // Tout)
    CHin_div_Tout = ((CHin + Tout - 1) // Tout)
    DAT_IN_LINE_STRIDE = Pixel_Data_Bytes * Win
    DAT_IN_HEAD_STRIDE = Pixel_Data_Bytes * Win * Hin * CHin_div_Tout
    WET_IN_LINE_STRIDE = Pixel_Data_Bytes * MAX_TOKEN
    WET_IN_HEAD_STRIDE = Pixel_Data_Bytes * MAX_TOKEN * CHout_div_Tout
    DAT_OUT_LINE_STRIDE = Pixel_Data_Bytes * Wout
    DAT_OUT_HEAD_STRIDE = Pixel_Data_Bytes * Wout * Hout * CHout_div_Tout
    if hasattr(args[0], "strides"):
        DAT_IN_LINE_STRIDE = args[0].strides[-2]
        DAT_IN_HEAD_STRIDE = args[0].strides[-4]
    if hasattr(args[1], "strides"):
        WET_IN_LINE_STRIDE = args[1].strides[-2]
        WET_IN_HEAD_STRIDE = args[1].strides[-4]
    if hasattr(outputs[0], "strides"):
        DAT_OUT_LINE_STRIDE = outputs[0].strides[-2]
        DAT_OUT_HEAD_STRIDE = outputs[0].strides[-4]

    # Hardware Testbench
    feature_in_base=DAT_IN_BASE_ADDR
    feature_out_base=DAT_OUT_BASE_ADDR
    Dynamic_Token = Win
    Head_Cfg = (Feature_Head // Weight_Head - 1) * 256 * 256 + Feature_Head * 256 + Weight_Head
    reg_14 = FP32_to_FP20(1.0) + (log2_WT_base_addr_Bank_Step << sCONV_OUT_DW)
    reg_bias = 192

    onchip = 0
    if DAT_IN_ONCHIP is not None:
        onchip += 0b1
        feature_in_base = ne.If(kvcache, DAT_IN_ONCHIP, feature_in_base)
    if DAT_OUT_ONCHIP is not None:
        onchip += 0b10
        feature_out_base = ne.If(kvcache, DAT_OUT_ONCHIP, feature_out_base)

    if onchip:
        onchip = ne.If(kvcache, onchip, 0)

    block += CSB_Write(reg_bias+2 , WT_BASE_ADDR               )
    block += CSB_Write(reg_bias+3 , feature_in_base            )
    block += CSB_Write(reg_bias+4 , DAT_IN_HEAD_STRIDE         )
    block += CSB_Write(reg_bias+5 , DAT_IN_LINE_STRIDE         )
    block += CSB_Write(reg_bias+6 , feature_out_base           )
    block += CSB_Write(reg_bias+7 , DAT_OUT_HEAD_STRIDE        )
    block += CSB_Write(reg_bias+8 , DAT_OUT_LINE_STRIDE        )
    block += CSB_Write(reg_bias+9 , WET_IN_HEAD_STRIDE         )
    block += CSB_Write(reg_bias+10, WET_IN_LINE_STRIDE         )
    block += CSB_Write(reg_bias+11, CHout                      )
    block += CSB_Write(reg_bias+12, Total_Token                )
    block += CSB_Write(reg_bias+13, Dynamic_Token              )
    block += CSB_Write(reg_bias+14, reg_14                     )
    block += CSB_Write(reg_bias+15, Head_Cfg                   )
    block += CSB_Write(reg_bias+16, 0                          )
    block += CSB_Write(reg_bias+17, 0b00_0001                  )
    block += While(CSB_Read(reg_bias+1) != 1)


@Tasks.Register("atom.hbm.dat2hbm", hbm_accel.HBM1128)
def Dat2HBM_Task(block, args, outputs, device, last_token=0, trp=0, **attrs):
    Token = args[0].shape[-2]
    Feature_Head = args[0].shape[0]
    DAT_IN_BASE_ADDR = args[0].get_address()
    DAT_OUT_BASE_ADDR = outputs[0].get_address()
    log2_WT_base_addr_Bank_Step = attrs.get("log2_WT_base_addr_Bank_Step", 28)

    Tout = device.Tout
    Pixel_Data_Bytes = device.Pixel_Data_Bytes
    MAX_TOKEN, HBM_Port = device.MAX_TOKEN, device.HBM_Port
    MAX_CH_per_HEAD = device.MAX_CH_per_HEAD

    Win = Token
    CHin = MAX_CH_per_HEAD
    CHout = MAX_CH_per_HEAD
    CHout_div_Tout = ((CHout + Tout - 1) // Tout)
    CHin_div_Tout = ((CHin + Tout - 1) // Tout)
    DAT_IN_LINE_STRIDE = Pixel_Data_Bytes * Win
    DAT_IN_HEAD_STRIDE = Pixel_Data_Bytes * Win * CHin_div_Tout
    WT_OUT_LINE_STRIDE = Pixel_Data_Bytes * MAX_TOKEN
    WT_OUT_HEAD_STRIDE = Pixel_Data_Bytes * MAX_TOKEN * CHout_div_Tout
    if hasattr(args[0], "strides"):
        DAT_IN_LINE_STRIDE = args[0].strides[-2]
        DAT_IN_HEAD_STRIDE = args[0].strides[-4]
    if hasattr(outputs[0], "strides"):
        WT_OUT_LINE_STRIDE = outputs[0].strides[-2]
        WT_OUT_HEAD_STRIDE = outputs[0].strides[-4]

    # Hardware Testbench
    feature_in_base=DAT_IN_BASE_ADDR
    feature_out_base=DAT_OUT_BASE_ADDR
    reg_bias = 192

    block += CSB_Write(reg_bias+3 , feature_in_base            )
    block += CSB_Write(reg_bias+4 , DAT_IN_HEAD_STRIDE         )
    block += CSB_Write(reg_bias+5 , DAT_IN_LINE_STRIDE         )
    block += CSB_Write(reg_bias+6 , feature_out_base           )
    block += CSB_Write(reg_bias+7 , WT_OUT_HEAD_STRIDE         )
    block += CSB_Write(reg_bias+8 , WT_OUT_LINE_STRIDE         )
    block += CSB_Write(reg_bias+9 , log2_WT_base_addr_Bank_Step)
    block += CSB_Write(reg_bias+10, last_token                 )
    block += CSB_Write(reg_bias+11, Token                      )
    block += CSB_Write(reg_bias+12, Feature_Head               )
    block += CSB_Write(reg_bias+13, CHin_div_Tout              )
    block += CSB_Write(reg_bias+14, trp                        )
    block += CSB_Write(reg_bias+15, 0                          )
    block += CSB_Write(reg_bias+16, 0                          )
    block += CSB_Write(reg_bias+17, 0b100_0000                 )
    block += While(CSB_Read(reg_bias+1) != 1)


@Tasks.Register("atom.hbm.mvm", hbm_accel.HBM1128)
def MVMBasic(block, args, outputs, device, onchip={}, kvcache=0, **attrs):
    RELU_EN = attrs.get("RELU_EN", 0)
    Skip_Factor = attrs.get("Skip_Factor", 1)
    EW_MODE = attrs.get("res_mode", 0)
    log2_WT_base_addr_Bank_Step = attrs.get("log2_WT_base_addr_Bank_Step", 28)
    DAT_OUT_LINE_STRIDE = None
    DAT_OUT_SURFACE_STRIDE = None

    DAT_IN_BASE_ADDR = args[0].get_address()
    HBM00_WT_BASE_ADDR = args[1].get_address()
    BN_BASE_ADDR = None
    Res_Add_BASE_ADDR = None
    if len(args) > 2:
        BN_BASE_ADDR = args[2].get_address()
        if len(args) > 3:
            Res_Add_BASE_ADDR = args[3].get_address()

    DAT_OUT_BASE_ADDR = outputs[0].get_address()
    AUGMAX_OUT_ADDR = None
    if len(outputs) > 1:
        AUGMAX_OUT_ADDR = outputs[1].get_address()

    DAT_IN_ONCHIP = onchip.get("DAT_IN_ONCHIP")
    RES_IN_ONCHIP = onchip.get("RES_IN_ONCHIP")
    DAT_OUT_ONCHIP = onchip.get("DAT_OUT_ONCHIP")
    reg_22 = (device.log2_WT_CH_Tgroup << device.log2_CH) + device.WT_CH_Tgroup
    reg_23 = (device.log2_T_quant_block << device.log2_CH) + device.T_quant_block

    Tin = device.base_Tin
    Tout = device.Tout
    Pixel_Data_Bytes = device.Pixel_Data_Bytes
    WT_DW = device.MAX_WT_DW
    HBM_AXI_DATA_WIDTH = device.HBM_AXI_DATA_WIDTH
    WT_CH_Tgroup = device.WT_CH_Tgroup
    MAX_WT_DW = device.MAX_WT_DW
    MAX_BN_DW = device.MAX_BN_DW
    HBM_Port = device.HBM_Port
    WT_BRAM_DEPTH = device.WT_BRAM_DEPTH
    AXI_BN_WIDTH = device.AXI_BN_WIDTH
    BN_FIFO_DEP = device.BN_FIFO_DEP
    BN_FIFO_NUM = device.BN_FIFO_NUM
    ASYN_FACTOR = device.ASYN_FACTOR

    Token = args[0].shape[-2]
    Win = Token
    Hin = args[0].shape[0]
    CHin = args[1].shape[0]
    CHout = args[1].shape[1]
    Wout = Win
    Hout = Hin
    CHout_div_Tout = ((CHout + Tout - 1) // Tout)
    CHin_div_Tout = ((CHin + Tout - 1) // Tout)
    CHin_Padding_with_Tout = CHin_div_Tout * Tout
    Tin_div_Tout = (Tin + Tout - 1) // Tout
    CHout_Padding = CHout_div_Tout * Tout
    DAT_IN_LINE_STRIDE = Pixel_Data_Bytes * Win
    DAT_IN_SURFACE_STRIDE = Pixel_Data_Bytes * Win * Hin
    DAT_OUT_LINE_STRIDE = Pixel_Data_Bytes * Wout
    DAT_OUT_SURFACE_STRIDE = Pixel_Data_Bytes * Wout * Hout
    if hasattr(args[0], "strides"):
        DAT_IN_LINE_STRIDE = args[0].strides[-2]
        DAT_IN_SURFACE_STRIDE = args[0].strides[-3]
    if hasattr(outputs[0], "strides"):
        DAT_OUT_LINE_STRIDE = outputs[0].strides[-2]
        DAT_OUT_SURFACE_STRIDE = outputs[0].strides[-3]

    WT_CHin_div_Tin = (CHin + Tin - 1) // Tin
    WT_CHin_Padding_with_Tin = WT_CHin_div_Tin*Tin
    WT_scale_group_nums = (WT_CHin_Padding_with_Tin + WT_CH_Tgroup-1) // WT_CH_Tgroup
    WT_scale_bits = CHout_Padding*HBM_AXI_DATA_WIDTH*WT_scale_group_nums
    WT_SIZE_IN_BYTES = (((CHout_Padding*WT_CHin_Padding_with_Tin*WT_DW)>>3)+((WT_scale_bits)>>3))

    if AUGMAX_OUT_ADDR is not None:
        if Res_Add_BASE_ADDR is not None:
            mode = MVMBNRESARG_MODE
        elif BN_BASE_ADDR is not None:
            mode = MVMBNARG_MODE
            Res_Add_BASE_ADDR = 0
        else:
            raise RuntimeError("MVM does NOT support ArgMax")
    elif Res_Add_BASE_ADDR is not None:
        mode = MVMBNRES_MODE
    elif BN_BASE_ADDR is not None:
        mode = MVMBN_MODE
        Res_Add_BASE_ADDR = 0
    else:
        mode = MVM_MODE
        BN_BASE_ADDR = 0
        Res_Add_BASE_ADDR = 0
    
    ## Hardware Testbench
    CHin = CHin_Padding_with_Tout
    CHout = CHout_Padding
    min_wt_depth=WT_CHin_div_Tin*((Tin*MAX_WT_DW)//HBM_AXI_DATA_WIDTH)*(Tout//HBM_Port)
    out_ch_slice=((WT_BRAM_DEPTH*ASYN_FACTOR)//min_wt_depth)*Tout
    # make sure reg 7 mast be 2**x
    tp_out_ch_slice=1<<(int(math.ceil(math.log2(out_ch_slice))))
    if tp_out_ch_slice>out_ch_slice:
        out_ch_slice=tp_out_ch_slice//2
    else:
        out_ch_slice=tp_out_ch_slice

    BN_FIFO_bits=AXI_BN_WIDTH*BN_FIFO_DEP*BN_FIFO_NUM
    BN_FIFO_chout_num=BN_FIFO_bits//(MAX_BN_DW*2)

    if mode != MVM_MODE:
        out_ch_slice = ne.If(out_ch_slice>BN_FIFO_chout_num, BN_FIFO_chout_num, out_ch_slice)
    out_ch_slice = ne.If(out_ch_slice >= CHout_Padding, CHout_Padding, out_ch_slice)
    CHout_Split_Times = ne.If(out_ch_slice >= CHout_Padding, 1, (CHout_Padding+out_ch_slice-1)//out_ch_slice)
    out_ch_slice_last = ne.If(CHout%out_ch_slice, CHout_Padding%out_ch_slice, out_ch_slice)
    CHout_Split_Times_minus1=CHout_Split_Times-1
    wt_size_in_bits = WT_SIZE_IN_BYTES // CHout_Padding * 8
    CHout = out_ch_slice
    CHout_last = out_ch_slice_last
    Last_Group_CHin = ne.If(WT_CHin_Padding_with_Tin%WT_CH_Tgroup, WT_CHin_Padding_with_Tin%WT_CH_Tgroup, WT_CH_Tgroup)

    feature_out_base = DAT_OUT_BASE_ADDR

    onchip = 0
    if DAT_IN_ONCHIP is not None:
        onchip += 0b1
        DAT_IN_BASE_ADDR = ne.If(kvcache, DAT_IN_ONCHIP, DAT_IN_BASE_ADDR)
    if RES_IN_ONCHIP is not None:
        Res_Add_BASE_ADDR = ne.If(kvcache, RES_IN_ONCHIP, Res_Add_BASE_ADDR)
    if DAT_OUT_ONCHIP is not None:
        onchip += 0b10
        feature_out_base = ne.If(kvcache, DAT_OUT_ONCHIP, feature_out_base)
    if onchip:
        onchip = ne.If(kvcache, onchip, 0)
    block += CSB_Write(2,CHin)
    block += CSB_Write(3,Win)
    block += CSB_Write(4,Hin)
    block += CSB_Write(5,Wout)
    block += CSB_Write(6,Hout)
    block += CSB_Write(7,CHout)
    block += CSB_Write(8,CHout_last)
    block += CSB_Write(9,Win)
    
    block += CSB_Write(10,DAT_IN_BASE_ADDR)
    block += CSB_Write(11,HBM00_WT_BASE_ADDR)
    block += CSB_Write(12,wt_size_in_bits)
    block += CSB_Write(13,feature_out_base)
    block += CSB_Write(14,CHout_Split_Times_minus1)
    block += CSB_Write(15,log2_WT_base_addr_Bank_Step)
    block += CSB_Write(16,EW_MODE)
    if AUGMAX_OUT_ADDR is not None:
        block += CSB_Write(17,AUGMAX_OUT_ADDR)
    else:
        block += CSB_Write(17,Skip_Factor-1)
    block += CSB_Write(18,onchip)
    
    block += CSB_Write(19,0)
    block += CSB_Write(20,0)
    block += CSB_Write(21,0)
    block += CSB_Write(22,reg_22)
    block += CSB_Write(23,reg_23)
    block += CSB_Write(24,Last_Group_CHin)
    block += CSB_Write(25,0)
    block += CSB_Write(26,BN_BASE_ADDR)
    block += CSB_Write(27,Res_Add_BASE_ADDR)
    block += CSB_Write(28,0)
    block += CSB_Write(29,DAT_IN_SURFACE_STRIDE)
    block += CSB_Write(30,DAT_IN_LINE_STRIDE)
    block += CSB_Write(31,DAT_OUT_SURFACE_STRIDE)
    block += CSB_Write(32,DAT_OUT_LINE_STRIDE)
    block += CSB_Write(33,mode)
    
    block += While(CSB_Read(1) != 1)


