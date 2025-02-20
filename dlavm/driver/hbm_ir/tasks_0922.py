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

@Tasks.Register("atom.hbm.aux", hbm_accel.HBM)
def AUX(func, inst_addr, aux_numb, task_numb, upt_calls=[]):
    func += CSB_Write(64+1, inst_addr)
    func += CSB_Write(64+2, aux_numb)
    func += CSB_Write(64+3, task_numb)
    func += CSB_Write(64+9, 1)
    for call in upt_calls:
        func += call
        func.update_args(call.func.args)
    func += While(CSB_Read(64) != 1)

@Tasks.Register("atom.hbm.wt2hbm", hbm_accel.HBM)
def Wt2HBM(rt_addr, hbm_addr, fname, total_bytes, ddr_base, device):
    with ir.Function([ne.Var("prefix", -1, "char*")]) as func:
        path = func.args[0]
        with ir.For("port", 0, device.HBM_Port, 1, "uint64_t") as port:
            total_bits = total_bytes * 8
            split_times=(total_bits+device.AUX_WT_BUF_DEPTH*device.AXI_DAT_WIDTH-1)//(device.AUX_WT_BUF_DEPTH*device.AXI_DAT_WIDTH)
            if split_times == 1:
                last_split_bits = total_bits
            else:
                last_split_bits = total_bits % (device.AUX_WT_BUF_DEPTH*device.AXI_DAT_WIDTH)
            if split_times==1:
                total_bits_in_each_slice=total_bits
            else:
                total_bits_in_each_slice=device.AUX_WT_BUF_DEPTH*device.AXI_DAT_WIDTH
            real_path = port[ir.StrFormat("real_path", "%s/"+fname, path, port.var)]
            rt_addr = ne.Var(rt_addr, -1)
            ddr_addr = ir.Cast(rt_addr + ddr_base, "uint64_t")
            port += ir.MemWriteFile(ddr_addr, real_path.var, total_bytes)
            hbm_out_addr = port.assign("hbm_out_addr", ne.Var(hbm_addr, -1) + (1 << device.log2_Bank_Step) * port.var, "uint64_t")
            if split_times == 1:
                port += CSB_Write(64+4, rt_addr)
                port += CSB_Write(64+5, ir.Cast(hbm_out_addr, "int"))
                port += CSB_Write(64+6, ir.Cast(hbm_out_addr >> 32, "int"))
                port += CSB_Write(64+7, last_split_bits)
                port += CSB_Write(64+8, port.var)
                port += CSB_Write(64+9, 2)
                port += While(CSB_Read(64) != 1)
            else:
                addr_step = device.AUX_WT_BUF_DEPTH * device.AXI_DAT_WIDTH // 8
                with ir.For("i", 0, split_times, 1) as i:
                    i += CSB_Write(64+4, rt_addr + i.var*addr_step)
                    i += CSB_Write(64+5, ir.Cast(hbm_out_addr, "int"))
                    i += CSB_Write(64+6, ir.Cast(hbm_out_addr >> 32, "int"))
                    i += CSB_Write(64+7, ne.If(i.var < split_times-1, total_bits_in_each_slice, last_split_bits))
                    i += CSB_Write(64+8, port.var)
                    i += CSB_Write(64+9, 2)
                    i += While(CSB_Read(64) != 1)
                    i += ir.Inplace(hbm_out_addr, ne.Op.add, total_bytes)
                port += i
            port += ir.MemInit(ddr_addr, total_bytes)
        func += port
    return func


@Tasks.Register("atom.hbm.pcie2mem", hbm_accel.HBM)
def PCIe2MEM(addr, fname, total_bytes, addr_base, device, is_hbm):
    with ir.Function([ne.Var("prefix", -1, "char*")]) as func:
        path = func.args[0]
        addr = ne.Var(addr, -1)
        addr = ir.Cast(addr + addr_base, "uint64_t")
        if is_hbm:
            with ir.For("port", 0, device.HBM_Port, 1, "int") as port:
                real_path = port[ir.StrFormat("real_path", "%s/"+fname, path, port.var)]
                port += ir.MemWriteFile(addr, real_path.var, total_bytes)
            func += port
        else:
            real_path = func[ir.StrFormat("real_path", "%s/"+fname, path)]
            func += ir.MemWriteFile(addr, real_path.var, total_bytes)
    return func


@Tasks.Register("atom.hbm.mvm", hbm_accel.HBM0912)
def MVMBasic_v2(block, args, outputs, device, onchip={}, kvcache=0, EW_MODE=0, **attrs):
    RELU_EN = attrs.get("RELU_EN", 0)
    Skip_Factor = attrs.get("Skip_Factor", 1)
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
    block += CSB_Write(16,(EW_MODE << 1) + RELU_EN)
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


@Tasks.Register("atom.hbm.trp_mvm", hbm_accel.HBM0912)
def MVM_afterTRP_task_0912(block, args, outputs, device, kvcache=0, last_token=0, **attrs):
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

    Dynamic_Token = Token
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
    recip_ch = np.array([1/math.sqrt(CHin),], dtype="float16")
    FP16_rsqrt = np.frombuffer(recip_ch.tobytes(), dtype="uint16")[0]
    Head_Cfg = (Feature_Head // Weight_Head - 1) * 256 * 256 + Feature_Head * 256 + Weight_Head
    reg_14 = int(FP16_rsqrt) + (log2_WT_base_addr_Bank_Step << 16)
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
    block += CSB_Write(reg_bias+12, CHout                      )
    block += CSB_Write(reg_bias+13, Dynamic_Token              )
    block += CSB_Write(reg_bias+14, reg_14                     )
    block += CSB_Write(reg_bias+15, Head_Cfg                   )
    block += CSB_Write(reg_bias+16, onchip                     )
    block += CSB_Write(reg_bias+17, 0b00_0010                  )
    block += While(CSB_Read(reg_bias+1) != 1)


@Tasks.Register("atom.hbm.f2w_mvm", hbm_accel.HBM0912)
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

    Win = Token
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
    reg_14 = 0x3c00 + (log2_WT_base_addr_Bank_Step << 16)
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
    block += CSB_Write(reg_bias+12, CHin                       )
    block += CSB_Write(reg_bias+13, Dynamic_Token              )
    block += CSB_Write(reg_bias+14, reg_14                     )
    block += CSB_Write(reg_bias+15, Head_Cfg                   )
    block += CSB_Write(reg_bias+16, 0                          )
    block += CSB_Write(reg_bias+17, 0b00_0001                  )
    block += While(CSB_Read(reg_bias+1) != 1)


@Tasks.Register("atom.hbm.dat2hbm", hbm_accel.HBM0912)
def Dat2HBM_Task_0912(block, args, outputs, device, last_token=0, trp=0, **attrs):
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
    feature_out_base=DAT_OUT_BASE_ADDR + last_token / HBM_Port * Pixel_Data_Bytes
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


@Tasks.Register("atom.hbm.layer_norm", hbm_accel.EdgeLLMv2)
def LayerNorm_v2(block, args, outputs, device, kvcache=0, rms=0, **attrs):
    Token = args[0].shape[-2]
    DAT_IN_BASE_ADDR = args[0].get_address()
    DAT_OUT_BASE_ADDR = outputs[0].get_address()

    DAT_IN_BASE_ADDR = args[0].get_address()
    LN_WT_BASE_ADDR = args[1].get_address()
    DAT_OUT_BASE_ADDR = outputs[0].get_address()
    DAT_IN_ONCHIP = attrs.get("DAT_IN_ONCHIP")
    DAT_OUT_ONCHIP = attrs.get("DAT_OUT_ONCHIP")

    Tin = device.base_Tin
    Tout = device.Tout
    Pixel_Data_Bytes = device.Pixel_Data_Bytes
    MAX_BN_DW = device.MAX_BN_DW
    AXI_DAT_WIDTH = device.AXI_DAT_WIDTH
    log2_AXI_BURST_LEN = device.log2_AXI_BURST_LEN

    Win = Token
    Hin = 1
    CHin = args[0].shape[-1]
    CHout = CHin
    Wout = Win
    Hout = Hin
    Layer_Norm = 1 - rms
    CHin_div_Tout = ((CHin + Tout - 1) // Tout)
    CHin_Padding_with_Tout = CHin_div_Tout * Tout
    LN_num_per_AXI_DW = AXI_DAT_WIDTH // (2*MAX_BN_DW)
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

    ## Hardware Testbench
    CHin = CHin_Padding_with_Tout
    recip_ch = 1 / CHin
    FP20_recip_CH_r = FP32_to_FP20(recip_ch)
    LN_CH_burst_times_minus1=(CHin//LN_num_per_AXI_DW)>>log2_AXI_BURST_LEN
    pixel_in = Win
    Ln_reg_bias = 192
    
    onchip = 0
    if DAT_IN_ONCHIP is not None:
        onchip += 0b1
        DAT_IN_BASE_ADDR = ne.If(kvcache, DAT_IN_ONCHIP, DAT_IN_BASE_ADDR)
    if DAT_OUT_ONCHIP is not None:
        onchip += 0b10
        DAT_OUT_BASE_ADDR = ne.If(kvcache, DAT_OUT_ONCHIP, DAT_OUT_BASE_ADDR)

    if onchip:
        onchip = ne.If(kvcache, onchip, 0)
    block += CSB_Write(Ln_reg_bias+2 , LN_WT_BASE_ADDR         )
    block += CSB_Write(Ln_reg_bias+3 , DAT_IN_BASE_ADDR        )
    block += CSB_Write(Ln_reg_bias+4 , DAT_IN_SURFACE_STRIDE   )
    block += CSB_Write(Ln_reg_bias+5 , DAT_IN_LINE_STRIDE      )
    block += CSB_Write(Ln_reg_bias+6 , DAT_OUT_BASE_ADDR       )
    block += CSB_Write(Ln_reg_bias+7 , DAT_OUT_SURFACE_STRIDE  )
    block += CSB_Write(Ln_reg_bias+8 , DAT_OUT_LINE_STRIDE     )
    block += CSB_Write(Ln_reg_bias+9 , (CHin+Tout-1)//Tout     )
    block += CSB_Write(Ln_reg_bias+10, Hin                     )
    block += CSB_Write(Ln_reg_bias+11, Win                     )
    block += CSB_Write(Ln_reg_bias+12, pixel_in                )
    block += CSB_Write(Ln_reg_bias+13, FP20_recip_CH_r         )
    block += CSB_Write(Ln_reg_bias+14, LN_CH_burst_times_minus1)
    block += CSB_Write(Ln_reg_bias+15,               Layer_Norm)
    block += CSB_Write(Ln_reg_bias+16,                   onchip)
    block += CSB_Write(Ln_reg_bias+17,                0b10_0000)
    block += While(CSB_Read(Ln_reg_bias+1) != 1)


@Tasks.Register("atom.hbm.pos_emb", hbm_accel.EdgeLLMv2)
def PosEmb_v2(block, args, outputs, device, kvcache=0, last_token=0, **attrs):
    Token = args[0].shape[-2]
    Feature_Head = args[0].shape[0]
    DAT_IN_BASE_ADDR = args[0].get_address()
    POS_IN_BASE_ADDR = args[1].get_address()
    DAT_OUT_BASE_ADDR = outputs[0].get_address()
    DAT_IN_ONCHIP = attrs.get("DAT_IN_ONCHIP")
    DAT_OUT_ONCHIP = attrs.get("DAT_OUT_ONCHIP")

    Tout = device.Tout
    Pixel_Data_Bytes = device.Pixel_Data_Bytes
    MAX_TOKEN = device.MAX_TOKEN
    MAX_CH_per_HEAD = device.MAX_CH_per_HEAD

    Dynamic_Token = Token
    Win = Dynamic_Token
    Hin = 1
    CHin = MAX_CH_per_HEAD
    CHout = CHin
    Wout = Win
    Hout = Hin
    CHout_div_Tout = ((CHout + Tout - 1) // Tout)
    CHin_div_Tout = ((CHin + Tout - 1) // Tout)
    DAT_IN_LINE_STRIDE = Pixel_Data_Bytes * Win
    DAT_IN_HEAD_STRIDE = Pixel_Data_Bytes * Win * Hin * CHin_div_Tout
    DAT_OUT_LINE_STRIDE = Pixel_Data_Bytes * Wout
    DAT_OUT_HEAD_STRIDE = Pixel_Data_Bytes * Wout * Hout * CHout_div_Tout
    POS_HEAD_STRIDE = Pixel_Data_Bytes*MAX_TOKEN*CHin_div_Tout//2
    POS_LINE_STRIDE = Pixel_Data_Bytes*MAX_TOKEN
    if hasattr(args[0], "strides"):
        DAT_IN_LINE_STRIDE = args[0].strides[-2]
        DAT_IN_HEAD_STRIDE = args[0].strides[-4]
    if hasattr(outputs[0], "strides"):
        DAT_OUT_LINE_STRIDE = outputs[0].strides[-2]
        DAT_OUT_HEAD_STRIDE = outputs[0].strides[-4]

    ## Hardware Testbench
    PosEmb_reg_bias=192
    feature_in_base=DAT_IN_BASE_ADDR
    feature_out_base = DAT_OUT_BASE_ADDR
    PosEmb_in_base=POS_IN_BASE_ADDR+last_token*Pixel_Data_Bytes

    onchip = 0
    if DAT_IN_ONCHIP is not None:
        onchip += 0b1
        feature_in_base = ne.If(kvcache, DAT_IN_ONCHIP, feature_in_base)
    if DAT_OUT_ONCHIP is not None:
        onchip += 0b10
        feature_out_base = ne.If(kvcache, DAT_OUT_ONCHIP, feature_out_base)

    if onchip:
        onchip = ne.If(kvcache, onchip, 0)

    block += CSB_Write(PosEmb_reg_bias+2 ,PosEmb_in_base         )
    block += CSB_Write(PosEmb_reg_bias+3 ,feature_in_base        )
    block += CSB_Write(PosEmb_reg_bias+4 ,DAT_IN_HEAD_STRIDE     )
    block += CSB_Write(PosEmb_reg_bias+5 ,DAT_IN_LINE_STRIDE     )
    block += CSB_Write(PosEmb_reg_bias+6 ,feature_out_base       )
    block += CSB_Write(PosEmb_reg_bias+7 ,DAT_OUT_HEAD_STRIDE    )
    block += CSB_Write(PosEmb_reg_bias+8 ,DAT_OUT_LINE_STRIDE    )
    block += CSB_Write(PosEmb_reg_bias+9 ,CHin_div_Tout          )
    block += CSB_Write(PosEmb_reg_bias+10,Dynamic_Token          )
    block += CSB_Write(PosEmb_reg_bias+11,last_token             )
    block += CSB_Write(PosEmb_reg_bias+12,Feature_Head           )
    block += CSB_Write(PosEmb_reg_bias+13,POS_HEAD_STRIDE        )
    block += CSB_Write(PosEmb_reg_bias+14,POS_LINE_STRIDE        )
    block += CSB_Write(PosEmb_reg_bias+15,0                      )
    block += CSB_Write(PosEmb_reg_bias+16,onchip                 )
    block += CSB_Write(PosEmb_reg_bias+17,0b00_0100              )
    block += While(CSB_Read(PosEmb_reg_bias+1) != 1)


@Tasks.Register("atom.hbm.softmax", hbm_accel.EdgeLLMv2)
def Softmax_v2(block, args, outputs, device, kvcache=0, last_token=0, **attrs):
    Token = args[0].shape[-2]
    Feature_Head = args[0].shape[0]
    DAT_IN_BASE_ADDR = args[0].get_address()
    DAT_OUT_BASE_ADDR = outputs[0].get_address()
    DAT_IN_ONCHIP = attrs.get("DAT_IN_ONCHIP")
    DAT_OUT_ONCHIP = attrs.get("DAT_OUT_ONCHIP")

    Tout = device.Tout
    Pixel_Data_Bytes = device.Pixel_Data_Bytes

    Dynamic_Token = Token
    Need_Mask = ne.If(Token - 1, 1, 0)
    Win = Dynamic_Token
    Hin = 1
    CHin = Token + last_token
    CHout = CHin
    Wout = Win
    Hout = Hin
    CHout_div_Tout = ((CHout + Tout - 1) // Tout)
    CHin_div_Tout = ((CHin + Tout - 1) // Tout)
    DAT_IN_LINE_STRIDE = Pixel_Data_Bytes * Win
    DAT_IN_HEAD_STRIDE = Pixel_Data_Bytes * Win * Hin * CHin_div_Tout
    DAT_OUT_LINE_STRIDE = Pixel_Data_Bytes * Wout
    DAT_OUT_HEAD_STRIDE = Pixel_Data_Bytes * Wout * Hout * CHout_div_Tout
    if hasattr(args[0], "strides"):
        DAT_IN_LINE_STRIDE = args[0].strides[-2]
        DAT_IN_HEAD_STRIDE = args[0].strides[-4]
    if hasattr(outputs[0], "strides"):
        DAT_OUT_LINE_STRIDE = outputs[0].strides[-2]
        DAT_OUT_HEAD_STRIDE = outputs[0].strides[-4]

    # Hardware Testbench
    feature_in_base = DAT_IN_BASE_ADDR
    feature_out_base = DAT_OUT_BASE_ADDR
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

    block += CSB_Write(reg_bias+2 , Need_Mask                  )
    block += CSB_Write(reg_bias+3 , feature_in_base            )
    block += CSB_Write(reg_bias+4 , DAT_IN_HEAD_STRIDE         )
    block += CSB_Write(reg_bias+5 , DAT_IN_LINE_STRIDE         )
    block += CSB_Write(reg_bias+6 , feature_out_base           )
    block += CSB_Write(reg_bias+7 , DAT_OUT_HEAD_STRIDE        )
    block += CSB_Write(reg_bias+8 , DAT_OUT_LINE_STRIDE        )
    block += CSB_Write(reg_bias+9 , (CHin+Tout-1)//Tout        )
    block += CSB_Write(reg_bias+10, Feature_Head               )
    block += CSB_Write(reg_bias+11, CHin                       )
    block += CSB_Write(reg_bias+12, Dynamic_Token              )
    block += CSB_Write(reg_bias+13, Win                        )
    block += CSB_Write(reg_bias+14, last_token                 )
    block += CSB_Write(reg_bias+15, 0                          )
    block += CSB_Write(reg_bias+16, onchip                     )
    block += CSB_Write(reg_bias+17, 0b00_1000                  )
    block += While(CSB_Read(reg_bias+1) != 1)


@Tasks.Register("atom.hbm.act", hbm_accel.EdgeLLMv2)
def ACT_v2(block, args, outputs, device, kvcache=0, last_token=0, **attrs):
    Token = args[0].shape[-2]
    DAT_IN_BASE_ADDR = args[0].get_address()
    WT_BASE_ADDR = args[1].get_address()
    DAT_OUT_BASE_ADDR = outputs[0].get_address()
    DAT_IN_ONCHIP = attrs.get("DAT_IN_ONCHIP")
    DAT_OUT_ONCHIP = attrs.get("DAT_OUT_ONCHIP")

    Tout = device.Tout
    Pixel_Data_Bytes = device.Pixel_Data_Bytes

    Win = Token
    Hin = 1
    CHin = args[0].shape[-1]
    CHout = CHin
    Wout = Win
    Hout = Hin
    CHout_div_Tout = ((CHout + Tout - 1) // Tout)
    CHin_div_Tout = ((CHin + Tout - 1) // Tout)
    CHin_Padding_with_Tout = CHin_div_Tout * Tout
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

    ## Hardware Testbench
    CHin = CHin_Padding_with_Tout
    parameters_base = WT_BASE_ADDR
    pixel_in = Win
    act_reg_bias = 192

    onchip = 0
    if DAT_IN_ONCHIP is not None:
        onchip += 0b1
        DAT_IN_BASE_ADDR = ne.If(kvcache, DAT_IN_ONCHIP, DAT_IN_BASE_ADDR)
    if DAT_OUT_ONCHIP is not None:
        onchip += 0b10
        DAT_OUT_BASE_ADDR = ne.If(kvcache, DAT_OUT_ONCHIP, DAT_OUT_BASE_ADDR)

    if onchip:
        onchip = ne.If(kvcache, onchip, 0)
    
    block += CSB_Write(act_reg_bias+2 , parameters_base         )
    block += CSB_Write(act_reg_bias+3 , DAT_IN_BASE_ADDR        )
    block += CSB_Write(act_reg_bias+4 , DAT_IN_SURFACE_STRIDE   )
    block += CSB_Write(act_reg_bias+5 , DAT_IN_LINE_STRIDE      )
    block += CSB_Write(act_reg_bias+6 , DAT_OUT_BASE_ADDR       )
    block += CSB_Write(act_reg_bias+7 , DAT_OUT_SURFACE_STRIDE  )
    block += CSB_Write(act_reg_bias+8 , DAT_OUT_LINE_STRIDE     )
    block += CSB_Write(act_reg_bias+9 , CHout_div_Tout          )
    block += CSB_Write(act_reg_bias+10, Hin                     )
    block += CSB_Write(act_reg_bias+11, Win                     )
    block += CSB_Write(act_reg_bias+12, pixel_in                )
    block += CSB_Write(act_reg_bias+13, CHout_div_Tout          )
    block += CSB_Write(act_reg_bias+14, CHout                   )
    block += CSB_Write(act_reg_bias+15, 0                       )
    block += CSB_Write(act_reg_bias+16, onchip                  )
    block += CSB_Write(act_reg_bias+17,                0b01_0000)
    block += While(CSB_Read(act_reg_bias+1) != 1)

