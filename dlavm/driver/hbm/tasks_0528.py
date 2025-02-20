import numpy as np
import math
from ..basic import CSB_Read, CSB_Write
from ...ne import Var, If, For, Numb, expr_for_hook
from ...clib import FP32_to_FP20

__version__ = "HBM0528"

MVM_MODE = 0b00100011111
MVMBN_MODE = 0b01100011111
MVMBNARG_MODE = 0b101100011111
MVMBNRES_MODE = 0b11100011111
MVMBNRESARG_MODE = 0b111100011111


def MVMBasic(**kwargs):
    device = kwargs["device"]
    Token = kwargs["Token"]
    kvcache = kwargs.get("kvcache", 0)
    kvcache_offset = kwargs.get("kvcache_offset", 0)
    Head = kwargs.get("Head", 1)
    Padding = kwargs.get("padding", 0)
    Width_in = kwargs["Width_in"]
    Width_out = kwargs["Width_out"]
    EW_MODE = kwargs.get("EW_MODE", 0)
    RELU_EN = kwargs.get("RELU_EN", 0)
    DAT_OUT_LINE_STRIDE = kwargs.get("DAT_OUT_LINE_STRIDE")
    DAT_OUT_SURFACE_STRIDE = kwargs.get("DAT_OUT_SURFACE_STRIDE")
    DAT_IN_BASE_ADDR = kwargs.get("DAT_IN_BASE_ADDR")
    HBM00_WT_BASE_ADDR = kwargs.get("HBM00_WT_BASE_ADDR")
    BN_BASE_ADDR = kwargs.get("BN_BASE_ADDR")
    Res_Add_BASE_ADDR = kwargs.get("Res_Add_BASE_ADDR")
    DAT_OUT_BASE_ADDR = kwargs.get("DAT_OUT_BASE_ADDR")
    AUGMAX_OUT_ADDR = kwargs.get("AUGMAX_OUT_ADDR")
    log2_WT_base_addr_Bank_Step = kwargs.get("log2_WT_base_addr_Bank_Step", 28)
    Skip_Factor = kwargs.get("Skip_Factor", 1)
    last_token = kwargs.get("last_token", 0)

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
    MAX_TOKEN = device.MAX_TOKEN

    Win = If(kvcache, 1, Token - last_token)
    Hin = Head
    CHin = Width_in
    CHout = Width_out
    Wout = Win
    Hout = Hin
    CHout_div_Tout = ((CHout + Tout - 1) // Tout)
    CHin_div_Tout = ((CHin + Tout - 1) // Tout)
    CHin_Padding_with_Tout = CHin_div_Tout * Tout
    Tin_div_Tout = (Tin + Tout - 1) // Tout
    CHout_Padding = CHout_div_Tout * Tout
    DAT_IN_LINE_STRIDE = Pixel_Data_Bytes * Win
    DAT_IN_SURFACE_STRIDE = Pixel_Data_Bytes * Win * Hin
    if DAT_OUT_LINE_STRIDE is None or DAT_OUT_SURFACE_STRIDE is None:
        DAT_OUT_LINE_STRIDE = Pixel_Data_Bytes * Wout
        DAT_OUT_SURFACE_STRIDE = Pixel_Data_Bytes * Wout * Hout

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
    dat_num_per_row=Win*WT_CHin_div_Tin
    min_dat_depth=dat_num_per_row
    min_wt_depth=WT_CHin_div_Tin*((Tin*MAX_WT_DW)//HBM_AXI_DATA_WIDTH)*(Tout//HBM_Port)

    out_ch_slice=((WT_BRAM_DEPTH*2)//min_wt_depth)*Tout
    BN_FIFO_bits=AXI_BN_WIDTH*BN_FIFO_DEP*BN_FIFO_NUM
    BN_FIFO_chout_num=BN_FIFO_bits//(MAX_BN_DW*2)
    if out_ch_slice>BN_FIFO_chout_num and mode != MVM_MODE:
        out_ch_slice=BN_FIFO_chout_num

    if out_ch_slice>=CHout_Padding:
        out_ch_slice=CHout_Padding
        CHout_Split_Times=1
    else:
        CHout_Split_Times=(CHout_Padding+out_ch_slice-1)//out_ch_slice

    if CHout%out_ch_slice==0:
        out_ch_slice_last=out_ch_slice
    else:
        out_ch_slice_last=CHout_Padding%out_ch_slice

    CHout_Split_Times_minus1=CHout_Split_Times-1
    wt_size_in_bits = WT_SIZE_IN_BYTES // CHout_Padding * 8
    CHout = out_ch_slice
    CHout_last = out_ch_slice_last

    if Padding:
        feature_out_base = If(kvcache, DAT_OUT_BASE_ADDR + (Token-1)*Pixel_Data_Bytes, DAT_OUT_BASE_ADDR + last_token*Pixel_Data_Bytes)
        DAT_OUT_LINE_STRIDE = Pixel_Data_Bytes*MAX_TOKEN
        DAT_OUT_SURFACE_STRIDE = Pixel_Data_Bytes*MAX_TOKEN*Hout
    else:
        DAT_IN_BASE_ADDR = If(kvcache_offset, DAT_IN_BASE_ADDR + (Token-1)*Pixel_Data_Bytes, DAT_IN_BASE_ADDR)
        DAT_IN_LINE_STRIDE = If(kvcache_offset, Pixel_Data_Bytes * Win, DAT_IN_LINE_STRIDE)
        DAT_IN_SURFACE_STRIDE = If(kvcache_offset, Pixel_Data_Bytes * Win * Hin, DAT_IN_SURFACE_STRIDE)
        feature_out_base = DAT_OUT_BASE_ADDR

    regs = []
    CSB_Write(regs, 2,CHin)
    CSB_Write(regs, 3,Win)
    CSB_Write(regs, 4,Hin)
    CSB_Write(regs, 5,Wout)
    CSB_Write(regs, 6,Hout)
    CSB_Write(regs, 7,CHout)
    CSB_Write(regs, 8,CHout_last)
    CSB_Write(regs, 9,If(kvcache, 1, (Token - last_token)))
    
    CSB_Write(regs, 10,DAT_IN_BASE_ADDR)
    CSB_Write(regs, 11,HBM00_WT_BASE_ADDR)
    CSB_Write(regs, 12,wt_size_in_bits)
    CSB_Write(regs, 13,feature_out_base)
    CSB_Write(regs, 14,CHout_Split_Times_minus1)
    CSB_Write(regs, 15,log2_WT_base_addr_Bank_Step)
    CSB_Write(regs, 16,(EW_MODE << 1) + RELU_EN)
    if AUGMAX_OUT_ADDR is not None:
        CSB_Write(regs, 17,AUGMAX_OUT_ADDR)
    else:
        CSB_Write(regs, 17,Skip_Factor-1)
    CSB_Write(regs, 18,0)
    
    CSB_Write(regs, 19,0)
    CSB_Write(regs, 20,0)
    CSB_Write(regs, 21,1)
    CSB_Write(regs, 22,1)
    CSB_Write(regs, 23,1)
    CSB_Write(regs, 24,1)
    CSB_Write(regs, 25,0)
    CSB_Write(regs, 26,BN_BASE_ADDR)
    CSB_Write(regs, 27,Res_Add_BASE_ADDR)
    CSB_Write(regs, 28,0)
    CSB_Write(regs, 29,DAT_IN_SURFACE_STRIDE)
    CSB_Write(regs, 30,DAT_IN_LINE_STRIDE)
    CSB_Write(regs, 31,DAT_OUT_SURFACE_STRIDE)
    CSB_Write(regs, 32,DAT_OUT_LINE_STRIDE)
    CSB_Write(regs, 33,mode)
    
    if AUGMAX_OUT_ADDR is not None:
        CSB_Read(regs, 40, 1)
    else:
        CSB_Read(regs, 1, 1)
    return regs


def LayerNorm(**kwargs):
    device = kwargs["device"]
    Token = kwargs["Token"]
    kvcache = kwargs.get("kvcache", 0)
    Head = kwargs.get("Head", 1)
    Width_in = kwargs["Width_in"]
    RMS_Norm = kwargs.get("RMS_Norm", 0)
    kvcache_offset = kwargs.get("kvcache_offset", 0)
    DAT_OUT_LINE_STRIDE = kwargs.get("DAT_OUT_LINE_STRIDE")
    DAT_OUT_SURFACE_STRIDE = kwargs.get("DAT_OUT_SURFACE_STRIDE")
    DAT_IN_BASE_ADDR = kwargs.get("DAT_IN_BASE_ADDR")
    LN_WT_BASE_ADDR = kwargs.get("LN_WT_BASE_ADDR")
    DAT_OUT_BASE_ADDR = kwargs.get("DAT_OUT_BASE_ADDR")
    last_token = kwargs.get("last_token", 0)

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
    AXI_DAT_WIDTH = device.AXI_DAT_WIDTH
    log2_AXI_BURST_LEN = device.log2_AXI_BURST_LEN

    Win = If(kvcache, 1, Token - last_token)
    Hin = Head
    CHin = Width_in
    CHout = CHin
    Wout = Win
    Hout = Hin
    Layer_Norm = 1 - RMS_Norm
    CHout_div_Tout = ((CHout + Tout - 1) // Tout)
    CHin_div_Tout = ((CHin + Tout - 1) // Tout)
    CHin_Padding_with_Tout = CHin_div_Tout * Tout
    LN_num_per_AXI_DW = AXI_DAT_WIDTH // (2*MAX_BN_DW)
    Tin_div_Tout = (Tin + Tout - 1) // Tout
    CHout_Padding = CHout_div_Tout * Tout
    DAT_IN_LINE_STRIDE = Pixel_Data_Bytes * Win
    DAT_IN_SURFACE_STRIDE = Pixel_Data_Bytes * Win * Hin
    if DAT_OUT_LINE_STRIDE is None or DAT_OUT_SURFACE_STRIDE is None:
        DAT_OUT_LINE_STRIDE = Pixel_Data_Bytes * Wout
        DAT_OUT_SURFACE_STRIDE = Pixel_Data_Bytes * Wout * Hout
    DAT_IN_BASE_ADDR = If(kvcache_offset, DAT_IN_BASE_ADDR + ((Token - last_token)-1)*Pixel_Data_Bytes, DAT_IN_BASE_ADDR)
    DAT_IN_LINE_STRIDE = If(kvcache_offset, Pixel_Data_Bytes * (Token - last_token), DAT_IN_LINE_STRIDE)
    DAT_IN_SURFACE_STRIDE = If(kvcache_offset, Pixel_Data_Bytes * (Token - last_token) * Hin, DAT_IN_SURFACE_STRIDE)

    ## Hardware Testbench
    CHin = CHin_Padding_with_Tout
    recip_ch = 1 / CHin
    FP20_recip_CH_r = FP32_to_FP20(recip_ch)
    LN_CH_burst_times_minus1=(CHin//LN_num_per_AXI_DW)>>log2_AXI_BURST_LEN
    pixel_in = Win
    ch_out = Width_in
    Ln_reg_bias = 192
    
    regs = []
    CSB_Write(regs, Ln_reg_bias+2 , LN_WT_BASE_ADDR         )
    CSB_Write(regs, Ln_reg_bias+3 , DAT_IN_BASE_ADDR        )
    CSB_Write(regs, Ln_reg_bias+4 , DAT_IN_SURFACE_STRIDE   )
    CSB_Write(regs, Ln_reg_bias+5 , DAT_IN_LINE_STRIDE      )
    CSB_Write(regs, Ln_reg_bias+6 , DAT_OUT_BASE_ADDR       )
    CSB_Write(regs, Ln_reg_bias+7 , DAT_OUT_SURFACE_STRIDE  )
    CSB_Write(regs, Ln_reg_bias+8 , DAT_OUT_LINE_STRIDE     )
    CSB_Write(regs, Ln_reg_bias+9 , (CHin+Tout-1)//Tout     )
    CSB_Write(regs, Ln_reg_bias+10, Hin                     )
    CSB_Write(regs, Ln_reg_bias+11, Win                     )
    CSB_Write(regs, Ln_reg_bias+12, pixel_in                )
    CSB_Write(regs, Ln_reg_bias+13, FP20_recip_CH_r         )
    CSB_Write(regs, Ln_reg_bias+14, LN_CH_burst_times_minus1)
    CSB_Write(regs, Ln_reg_bias+15,               Layer_Norm)
    CSB_Write(regs, Ln_reg_bias+16,                        0)
    CSB_Write(regs, Ln_reg_bias+17,                0b10_0000)
    CSB_Read(regs, Ln_reg_bias+1, 1)
    return regs


def EleminateWise(**kwargs):
    device = kwargs["device"]
    Token = kwargs["Token"]
    kvcache = kwargs.get("kvcache", 0)
    Mode = kwargs["Mode"]
    Width_in = kwargs["Width_in"]
    A_DAT_IN_BASE_ADDR = kwargs.get("A_DAT_IN_BASE_ADDR")
    B_DAT_IN_BASE_ADDR = kwargs.get("B_DAT_IN_BASE_ADDR")
    DAT_OUT_BASE_ADDR = kwargs.get("DAT_OUT_BASE_ADDR")
    last_token = kwargs.get("last_token", 0)

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
    AXI_DAT_WIDTH = device.AXI_DAT_WIDTH
    log2_AXI_BURST_LEN = device.log2_AXI_BURST_LEN

    Win = If(kvcache, 1, Token - last_token)
    Hin = 1
    CHin = Width_in
    CHout = CHin
    Wout = kwargs.get("Wout", Win)
    Hout = Hin
    CHout_div_Tout = ((CHout + Tout - 1) // Tout)
    CHin_div_Tout = ((CHin + Tout - 1) // Tout)
    CHin_Padding_with_Tout = CHin_div_Tout * Tout
    LN_num_per_AXI_DW = AXI_DAT_WIDTH // (2*MAX_BN_DW)
    Tin_div_Tout = (Tin + Tout - 1) // Tout
    CHout_Padding = CHout_div_Tout * Tout
    DAT_IN_LINE_STRIDE = Pixel_Data_Bytes * Win
    DAT_IN_SURFACE_STRIDE = Pixel_Data_Bytes * Win * Hin
    DAT_OUT_LINE_STRIDE = Pixel_Data_Bytes * Wout
    DAT_OUT_SURFACE_STRIDE = Pixel_Data_Bytes * Wout * Hout

    ## Hardware Testbench
    Elementwise_reg_bias = 128
    
    regs = []
    CSB_Write(regs, Elementwise_reg_bias+2 , Mode         )
    CSB_Write(regs, Elementwise_reg_bias+3 , A_DAT_IN_BASE_ADDR      )
    CSB_Write(regs, Elementwise_reg_bias+4 , DAT_IN_SURFACE_STRIDE   )
    CSB_Write(regs, Elementwise_reg_bias+5 , DAT_IN_LINE_STRIDE      )
    CSB_Write(regs, Elementwise_reg_bias+6 , DAT_OUT_BASE_ADDR       )
    CSB_Write(regs, Elementwise_reg_bias+7 , DAT_OUT_SURFACE_STRIDE  )
    CSB_Write(regs, Elementwise_reg_bias+8 , DAT_OUT_LINE_STRIDE     )
    CSB_Write(regs, Elementwise_reg_bias+9 , (CHin+Tout-1)//Tout     )
    CSB_Write(regs, Elementwise_reg_bias+10, Hin                     )
    CSB_Write(regs, Elementwise_reg_bias+11, Win                     )
    CSB_Write(regs, Elementwise_reg_bias+12, B_DAT_IN_BASE_ADDR      )
    CSB_Write(regs, Elementwise_reg_bias+13,                        0)
    CSB_Write(regs, Elementwise_reg_bias+14,                0b00_0001)
    CSB_Read(regs, Elementwise_reg_bias+1, 1)
    return regs


def PosEmb(**kwargs):
    device = kwargs["device"]
    Token = kwargs["Token"]
    kvcache = kwargs.get("kvcache", 0)
    Head = kwargs.get("Head", 1)
    Feature_Head = kwargs["Feature_Head"]
    DAT_IN_BASE_ADDR = kwargs.get("DAT_IN_BASE_ADDR")
    POS_IN_BASE_ADDR = kwargs.get("POS_IN_BASE_ADDR")
    DAT_OUT_BASE_ADDR = kwargs.get("DAT_OUT_BASE_ADDR")
    last_token = kwargs.get("last_token", 0)

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
    AXI_DAT_WIDTH = device.AXI_DAT_WIDTH
    log2_AXI_BURST_LEN = device.log2_AXI_BURST_LEN
    MAX_TOKEN = device.MAX_TOKEN
    MAX_CH_per_HEAD = device.MAX_CH_per_HEAD

    Win = MAX_TOKEN
    Hin = Head
    CHin = MAX_CH_per_HEAD
    CHout = CHin
    Wout = Win
    Hout = Hin
    CHout_div_Tout = ((CHout + Tout - 1) // Tout)
    CHin_div_Tout = ((CHin + Tout - 1) // Tout)
    CHin_Padding_with_Tout = CHin_div_Tout * Tout
    LN_num_per_AXI_DW = AXI_DAT_WIDTH // (2*MAX_BN_DW)
    Tin_div_Tout = (Tin + Tout - 1) // Tout
    CHout_Padding = CHout_div_Tout * Tout
    DAT_IN_LINE_STRIDE = Pixel_Data_Bytes * Win
    DAT_IN_HEAD_STRIDE = Pixel_Data_Bytes * Win * Hin * CHin_div_Tout
    DAT_OUT_LINE_STRIDE = Pixel_Data_Bytes * Wout
    DAT_OUT_HEAD_STRIDE = Pixel_Data_Bytes * Wout * Hout * CHout_div_Tout
    POS_LINE_STRIDE = Pixel_Data_Bytes * Win

    ## Hardware Testbench
    PosEmb_reg_bias=192
    feature_in_base=If(kvcache, DAT_IN_BASE_ADDR+(Token-1)*Pixel_Data_Bytes, DAT_IN_BASE_ADDR+last_token*Pixel_Data_Bytes)
    feature_out_base=If(kvcache, DAT_OUT_BASE_ADDR+(Token-1)*Pixel_Data_Bytes, DAT_OUT_BASE_ADDR+last_token*Pixel_Data_Bytes)
    PosEmb_in_base=If(kvcache, POS_IN_BASE_ADDR+(Token-1)*Pixel_Data_Bytes, POS_IN_BASE_ADDR+last_token*Pixel_Data_Bytes)
    Dynamic_Token = If(kvcache, 1, Token - last_token)

    regs = [] 
    CSB_Write(regs, PosEmb_reg_bias+2 ,PosEmb_in_base         )
    CSB_Write(regs, PosEmb_reg_bias+3 ,feature_in_base        )
    CSB_Write(regs, PosEmb_reg_bias+4 ,DAT_IN_HEAD_STRIDE     )
    CSB_Write(regs, PosEmb_reg_bias+5 ,DAT_IN_LINE_STRIDE     )
    CSB_Write(regs, PosEmb_reg_bias+6 ,feature_out_base       )
    CSB_Write(regs, PosEmb_reg_bias+7 ,DAT_OUT_HEAD_STRIDE    )
    CSB_Write(regs, PosEmb_reg_bias+8 ,DAT_OUT_LINE_STRIDE    )
    CSB_Write(regs, PosEmb_reg_bias+9 ,CHin_div_Tout          )
    CSB_Write(regs, PosEmb_reg_bias+10,Dynamic_Token          )
    CSB_Write(regs, PosEmb_reg_bias+11,Feature_Head           )
    CSB_Write(regs, PosEmb_reg_bias+12,POS_LINE_STRIDE        )
    CSB_Write(regs, PosEmb_reg_bias+13,0                      )
    CSB_Write(regs, PosEmb_reg_bias+14,0                      )
    CSB_Write(regs, PosEmb_reg_bias+15,0                      )
    CSB_Write(regs, PosEmb_reg_bias+16,0                      )
    CSB_Write(regs, PosEmb_reg_bias+17,0b00_0100              )
    CSB_Read(regs, PosEmb_reg_bias+1,1)
    return regs


def MVM_afterTRP(**kwargs):
    device = kwargs["device"]
    Token = kwargs["Token"]
    kvcache = kwargs.get("kvcache", 0)
    Head = kwargs.get("Head", 1)
    Feature_Head = kwargs["Feature_Head"]
    Weight_Head = kwargs["Weight_Head"]
    DAT_IN_BASE_ADDR = kwargs.get("DAT_IN_BASE_ADDR")
    WT_BASE_ADDR = kwargs.get("WT_BASE_ADDR")
    DAT_OUT_BASE_ADDR = kwargs.get("DAT_OUT_BASE_ADDR")
    last_token = kwargs.get("last_token", 0)

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
    AXI_DAT_WIDTH = device.AXI_DAT_WIDTH
    log2_AXI_BURST_LEN = device.log2_AXI_BURST_LEN
    MAX_TOKEN = device.MAX_TOKEN
    MAX_CH_per_HEAD = device.MAX_CH_per_HEAD
    MIN_WT_HEAD = device.MIN_WT_HEAD

    Win = MAX_TOKEN
    Hin = Head
    CHin = MAX_CH_per_HEAD
    CHout = MAX_TOKEN
    Wout = Win
    Hout = Hin
    CHout_div_Tout = ((CHout + Tout - 1) // Tout)
    CHin_div_Tout = ((CHin + Tout - 1) // Tout)
    CHin_Padding_with_Tout = CHin_div_Tout * Tout
    Tin_div_Tout = (Tin + Tout - 1) // Tout
    CHout_Padding = CHout_div_Tout * Tout
    DAT_IN_LINE_STRIDE = Pixel_Data_Bytes * Win
    DAT_IN_HEAD_STRIDE = Pixel_Data_Bytes * Win * Hin * CHin_div_Tout
    DAT_OUT_LINE_STRIDE = Pixel_Data_Bytes * Wout
    DAT_OUT_HEAD_STRIDE = Pixel_Data_Bytes * Wout * Hout * CHout_div_Tout

    # Hardware Testbench
    feature_in_base=If(kvcache, DAT_IN_BASE_ADDR+(Token-1)*Pixel_Data_Bytes, DAT_IN_BASE_ADDR+last_token*Pixel_Data_Bytes)
    feature_out_base=If(kvcache, DAT_OUT_BASE_ADDR+(Token-1)*Pixel_Data_Bytes, DAT_OUT_BASE_ADDR+last_token*Pixel_Data_Bytes)
    Dynamic_Token = If(kvcache, 1, Token - last_token)
    recip_ch = np.array([1/math.sqrt(CHin),], dtype="float16")
    FP16_rsqrt = np.frombuffer(recip_ch.tobytes(), dtype="uint16")[0]
    Feature_Repeat_times_minus1=Feature_Head//MIN_WT_HEAD-1
    reg_bias = 192

    regs = []
    CSB_Write(regs, reg_bias+2 , WT_BASE_ADDR               )
    CSB_Write(regs, reg_bias+3 , feature_in_base            )
    CSB_Write(regs, reg_bias+4 , DAT_IN_HEAD_STRIDE         )
    CSB_Write(regs, reg_bias+5 , DAT_IN_LINE_STRIDE         )
    CSB_Write(regs, reg_bias+6 , feature_out_base           )
    CSB_Write(regs, reg_bias+7 , DAT_OUT_HEAD_STRIDE        )
    CSB_Write(regs, reg_bias+8 , DAT_OUT_LINE_STRIDE        )
    CSB_Write(regs, reg_bias+9 , (CHin+Tout-1)//Tout        )
    CSB_Write(regs, reg_bias+10, Token                      )
    CSB_Write(regs, reg_bias+11, Weight_Head                )
    CSB_Write(regs, reg_bias+12, Dynamic_Token              )
    CSB_Write(regs, reg_bias+13, Feature_Head               )
    CSB_Write(regs, reg_bias+14, Feature_Repeat_times_minus1)
    CSB_Write(regs, reg_bias+15, FP16_rsqrt                 )
    CSB_Write(regs, reg_bias+16, 0                          )
    CSB_Write(regs, reg_bias+17, 0b00_0010                  )
    CSB_Read(regs, reg_bias+1,1)
    return regs


def MVM_afterF2W(**kwargs):
    device = kwargs["device"]
    Token = kwargs["Token"]
    kvcache = kwargs.get("kvcache", 0)
    Head = kwargs.get("Head", 1)
    Feature_Head = kwargs["Feature_Head"]
    Weight_Head = kwargs["Weight_Head"]
    DAT_IN_BASE_ADDR = kwargs.get("DAT_IN_BASE_ADDR")
    WT_BASE_ADDR = kwargs.get("WT_BASE_ADDR")
    DAT_OUT_BASE_ADDR = kwargs.get("DAT_OUT_BASE_ADDR")
    last_token = kwargs.get("last_token", 0)

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
    AXI_DAT_WIDTH = device.AXI_DAT_WIDTH
    log2_AXI_BURST_LEN = device.log2_AXI_BURST_LEN
    MAX_TOKEN = device.MAX_TOKEN
    MAX_CH_per_HEAD = device.MAX_CH_per_HEAD
    MIN_WT_HEAD = device.MIN_WT_HEAD

    Win = MAX_TOKEN
    Hin = Head
    CHin = MAX_TOKEN
    CHout = MAX_CH_per_HEAD
    Wout = If(kvcache, 1, Token - last_token)
    Hout = Hin
    CHout_div_Tout = ((CHout + Tout - 1) // Tout)
    CHin_div_Tout = ((CHin + Tout - 1) // Tout)
    CHin_Padding_with_Tout = CHin_div_Tout * Tout
    Tin_div_Tout = (Tin + Tout - 1) // Tout
    CHout_Padding = CHout_div_Tout * Tout
    DAT_IN_LINE_STRIDE = Pixel_Data_Bytes * Win
    DAT_IN_HEAD_STRIDE = Pixel_Data_Bytes * Win * Hin * CHin_div_Tout
    DAT_OUT_LINE_STRIDE = Pixel_Data_Bytes * Wout
    DAT_OUT_HEAD_STRIDE = Pixel_Data_Bytes * Wout * Hout * CHout_div_Tout

    # Hardware Testbench
    feature_in_base=If(kvcache, DAT_IN_BASE_ADDR+(Token-1)*Pixel_Data_Bytes, DAT_IN_BASE_ADDR+last_token*Pixel_Data_Bytes)
    feature_out_base=DAT_OUT_BASE_ADDR
    Dynamic_Token = If(kvcache, 1, Token - last_token)
    Feature_Repeat_times_minus1=Feature_Head//MIN_WT_HEAD-1
    reg_bias = 192

    regs = []
    CSB_Write(regs, reg_bias+2 , WT_BASE_ADDR               )
    CSB_Write(regs, reg_bias+3 , feature_in_base            )
    CSB_Write(regs, reg_bias+4 , DAT_IN_HEAD_STRIDE         )
    CSB_Write(regs, reg_bias+5 , DAT_IN_LINE_STRIDE         )
    CSB_Write(regs, reg_bias+6 , feature_out_base           )
    CSB_Write(regs, reg_bias+7 , DAT_OUT_HEAD_STRIDE        )
    CSB_Write(regs, reg_bias+8 , DAT_OUT_LINE_STRIDE        )
    CSB_Write(regs, reg_bias+9 , (Token+Tout-1)//Tout       )
    CSB_Write(regs, reg_bias+10, Token                      )
    CSB_Write(regs, reg_bias+11, Weight_Head                )
    CSB_Write(regs, reg_bias+12, Dynamic_Token              )
    CSB_Write(regs, reg_bias+13, Feature_Head               )
    CSB_Write(regs, reg_bias+14, Feature_Repeat_times_minus1)
    CSB_Write(regs, reg_bias+15, (CHout+Tout-1)//Tout       ) # CHout_div_Tout
    CSB_Write(regs, reg_bias+16, 0                          )
    CSB_Write(regs, reg_bias+17, 0b00_0001                  )
    CSB_Read(regs, reg_bias+1,1)
    return regs


def Softmax(**kwargs):
    device = kwargs["device"]
    Token = kwargs["Token"]
    kvcache = kwargs.get("kvcache", 0)
    Head = kwargs.get("Head", 1)
    Feature_Head = kwargs["Feature_Head"]
    DAT_IN_BASE_ADDR = kwargs.get("DAT_IN_BASE_ADDR")
    DAT_OUT_BASE_ADDR = kwargs.get("DAT_OUT_BASE_ADDR")
    last_token = kwargs.get("last_token", 0)

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
    AXI_DAT_WIDTH = device.AXI_DAT_WIDTH
    log2_AXI_BURST_LEN = device.log2_AXI_BURST_LEN
    MAX_TOKEN = device.MAX_TOKEN
    MAX_CH_per_HEAD = device.MAX_CH_per_HEAD
    MIN_WT_HEAD = device.MIN_WT_HEAD
    AXI_BURST_LEN_SOFTMAX = device.AXI_BURST_LEN_SOFTMAX

    Need_Mask = 1 - kvcache
    Win = MAX_TOKEN
    Hin = Head
    CHin = MAX_TOKEN
    CHout = MAX_TOKEN
    Wout = Win
    Hout = Hin
    CHout_div_Tout = ((CHout + Tout - 1) // Tout)
    CHin_div_Tout = ((CHin + Tout - 1) // Tout)
    CHin_Padding_with_Tout = CHin_div_Tout * Tout
    Tin_div_Tout = (Tin + Tout - 1) // Tout
    CHout_Padding = CHout_div_Tout * Tout
    DAT_IN_LINE_STRIDE = Pixel_Data_Bytes * Win
    DAT_IN_HEAD_STRIDE = Pixel_Data_Bytes * Win * Hin * CHin_div_Tout
    DAT_OUT_LINE_STRIDE = Pixel_Data_Bytes * Wout
    DAT_OUT_HEAD_STRIDE = Pixel_Data_Bytes * Wout * Hout * CHout_div_Tout

    # Hardware Testbench
    feature_in_base=If(kvcache, DAT_IN_BASE_ADDR+(Token-1)*Pixel_Data_Bytes, DAT_IN_BASE_ADDR+last_token*Pixel_Data_Bytes)
    feature_out_base=If(kvcache, DAT_OUT_BASE_ADDR+(Token-1)*Pixel_Data_Bytes, DAT_OUT_BASE_ADDR+last_token*Pixel_Data_Bytes)
    Dynamic_Token = If(kvcache, 1, Token - last_token)
    Feature_Repeat_times_minus1=Feature_Head//MIN_WT_HEAD-1
    w_in_tp = (Token-last_token+AXI_BURST_LEN_SOFTMAX-1)//AXI_BURST_LEN_SOFTMAX*AXI_BURST_LEN_SOFTMAX
    w_in=If(kvcache, AXI_BURST_LEN_SOFTMAX, w_in_tp)
    reg_bias = 192
    regs = []
    CSB_Write(regs, reg_bias+2 , Need_Mask                  )
    CSB_Write(regs, reg_bias+3 , feature_in_base            )
    CSB_Write(regs, reg_bias+4 , DAT_IN_HEAD_STRIDE         )
    CSB_Write(regs, reg_bias+5 , DAT_IN_LINE_STRIDE         )
    CSB_Write(regs, reg_bias+6 , feature_out_base           )
    CSB_Write(regs, reg_bias+7 , DAT_OUT_HEAD_STRIDE        )
    CSB_Write(regs, reg_bias+8 , DAT_OUT_LINE_STRIDE        )
    CSB_Write(regs, reg_bias+9 , (Token+Tout-1)//Tout       )
    CSB_Write(regs, reg_bias+10, Feature_Head               )
    CSB_Write(regs, reg_bias+11, Token                      )
    CSB_Write(regs, reg_bias+12, w_in                       )
    CSB_Write(regs, reg_bias+13, Dynamic_Token              )
    CSB_Write(regs, reg_bias+14, last_token                 )
    CSB_Write(regs, reg_bias+15, 0                          )
    CSB_Write(regs, reg_bias+16, 0                          )
    CSB_Write(regs, reg_bias+17, 0b00_1000                  )
    CSB_Read(regs, reg_bias+1,1)
    return regs


def ACT(**kwargs):
    device = kwargs["device"]
    Token = kwargs["Height"]
    kvcache = kwargs.get("kvcache", 0)
    Head = kwargs.get("Head", 1)
    Width_in = kwargs["Width_in"]
    DAT_OUT_LINE_STRIDE = kwargs.get("DAT_OUT_LINE_STRIDE")
    DAT_OUT_SURFACE_STRIDE = kwargs.get("DAT_OUT_SURFACE_STRIDE")
    DAT_IN_BASE_ADDR = kwargs.get("DAT_IN_BASE_ADDR")
    WT_BASE_ADDR = kwargs.get("WT_BASE_ADDR")
    DAT_OUT_BASE_ADDR = kwargs.get("DAT_OUT_BASE_ADDR")
    last_token = kwargs.get("last_token", 0)

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
    AXI_DAT_WIDTH = device.AXI_DAT_WIDTH
    log2_AXI_BURST_LEN = device.log2_AXI_BURST_LEN

    Win = If(kvcache, 1, Token - last_token)
    Hin = Head
    CHin = Width_in
    CHout = CHin
    Wout = kwargs.get("Wout", Win)
    Hout = Hin
    CHout_div_Tout = ((CHout + Tout - 1) // Tout)
    CHin_div_Tout = ((CHin + Tout - 1) // Tout)
    CHin_Padding_with_Tout = CHin_div_Tout * Tout
    LN_num_per_AXI_DW = AXI_DAT_WIDTH // (2*MAX_BN_DW)
    Tin_div_Tout = (Tin + Tout - 1) // Tout
    CHout_Padding = CHout_div_Tout * Tout
    DAT_IN_LINE_STRIDE = Pixel_Data_Bytes * Win
    DAT_IN_SURFACE_STRIDE = Pixel_Data_Bytes * Win * Hin
    if DAT_OUT_LINE_STRIDE is None or DAT_OUT_SURFACE_STRIDE is None:
        DAT_OUT_LINE_STRIDE = Pixel_Data_Bytes * Wout
        DAT_OUT_SURFACE_STRIDE = Pixel_Data_Bytes * Wout * Hout

    ## Hardware Testbench
    CHin = CHin_Padding_with_Tout
    parameters_base = WT_BASE_ADDR
    pixel_in = Win
    act_reg_bias = 192
    
    regs = []
    CSB_Write(regs, act_reg_bias+2 , parameters_base         )
    CSB_Write(regs, act_reg_bias+3 , DAT_IN_BASE_ADDR        )
    CSB_Write(regs, act_reg_bias+4 , DAT_IN_SURFACE_STRIDE   )
    CSB_Write(regs, act_reg_bias+5 , DAT_IN_LINE_STRIDE      )
    CSB_Write(regs, act_reg_bias+6 , DAT_OUT_BASE_ADDR       )
    CSB_Write(regs, act_reg_bias+7 , DAT_OUT_SURFACE_STRIDE  )
    CSB_Write(regs, act_reg_bias+8 , DAT_OUT_LINE_STRIDE     )
    CSB_Write(regs, act_reg_bias+9 , CHout_div_Tout          )
    CSB_Write(regs, act_reg_bias+10, Hin                     )
    CSB_Write(regs, act_reg_bias+11, Win                     )
    CSB_Write(regs, act_reg_bias+12, pixel_in                )
    CSB_Write(regs, act_reg_bias+13, CHout_div_Tout          )
    CSB_Write(regs, act_reg_bias+14, CHout                   )
    CSB_Write(regs, act_reg_bias+15, 0                       )
    CSB_Write(regs, act_reg_bias+16, 0                       )
    CSB_Write(regs, act_reg_bias+17,                0b01_0000)
    CSB_Read(regs, act_reg_bias+1, 1)
    return regs
