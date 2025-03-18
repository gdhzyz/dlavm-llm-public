import math
from dlavm import ne
from dlavm.adr import Op, Attrs
from dlavm.device import ohbm_accel
from dlavm.clib import FP32_to_FP20
from .... import ir
from ....ir import CSB_Write, CSB_Read, While
from ....basic import Tasks, Ceil, Ceil_Padding


#########################################################################################
#                                nn.norm compute task                                   #
#########################################################################################
@Tasks.Register("ohbm.nn.norm", ohbm_accel.OHBM0316)
def Norm(func, args, outputs, attrs):
    device = args[0].device
    data, weight = args

    # macro define testbench
    WT_DW = device.MAX_WT_DW
    Sparsity_Factor = 1
    Head, Height, Width_in = data.shape[-3:]
    Width_out = Width_in
    feature_in_base_addr = data.get_address()
    wt_base_addr = weight.get_address()
    feature_out_base_addr = outputs[0].get_address()
    Layer_Norm = 0 if attrs.get("rms") else 1

    # pre process
    Tb = 1
    Hin, Win = 1, Height
    CHin = Ceil_Padding(Width_in, device.Tout)
    Hout, Wout = 1, Height
    CHout = Ceil_Padding(Width_out, device.Tout)
    CHin_div_LTout = Ceil(CHin, device.L_Tout)
    CHin_Padding = CHin_div_LTout * device.L_Tout
    CHout_div_Tout = Ceil(CHout, device.Tout)
    CHin_Padding_with_LTout = CHin_Padding

    # stride load or compute
    feature_in_line_stride = device.HBM_1Row_Bytes * Win
    feature_in_surface_stride = device.HBM_1Row_Bytes * Win * Hin
    feature_out_line_stride = device.HBM_1Row_Bytes * Wout
    feature_out_surface_stride = device.HBM_1Row_Bytes * Wout * Hout
    if hasattr(data, "strides"):
        feature_in_line_stride = data.strides[-1]
        feature_in_surface_stride = data.strides[-2]
    if hasattr(outputs[0], "strides"):
        feature_out_line_stride = outputs[0].strides[-1]
        feature_out_surface_stride = outputs[0].strides[-2]

    # task function
    FP20_recip_CH_r = FP32_to_FP20(1/(CHin))
    pixel_in=Height
    ch_out=Width_in
    Ln_reg_bias=128

    Total_Dat_Bits_PerToken =CHin_Padding_with_LTout*device.MAX_DAT_DW
    Onchip_Token_perWTHead  =device.TOTAL_DAT_BRAM_BITS//Total_Dat_Bits_PerToken
    Required_Dat_BRAM_Bits  =device.AXI_BURST_LEN*device.MAX_DAT_DW*CHin_Padding_with_LTout
    
    Wout_Split_Times_minus1 = func.assign("Wout_Split_Times_minus1", 0, "int")
    out_w_per_slice         = func.assign("out_w_per_slice", Wout, "int")
    out_w_in_last_slice     = func.assign("out_w_in_last_slice", Wout, "int")
    t_if = ir.If(Required_Dat_BRAM_Bits>device.TOTAL_DAT_BRAM_BITS)
    with t_if.then_block as _then:
        Wout_Split_Times_minus1=_then.assign_var(Wout_Split_Times_minus1, (Wout+Onchip_Token_perWTHead-1)//Onchip_Token_perWTHead-1)
        out_w_per_slice        =_then.assign_var(out_w_per_slice, Onchip_Token_perWTHead)
        out_w_in_last_slice    =_then.assign_var(out_w_in_last_slice, Wout-(Wout_Split_Times_minus1)*out_w_per_slice)
    func += t_if
    with ir.For("w", 0, Wout_Split_Times_minus1+1, 1) as w_for:
        w = w_for.var
        Win = ne.If(w < Wout_Split_Times_minus1, out_w_per_slice, out_w_in_last_slice)
        tp_feature_in_base_addr =feature_in_base_addr  +w*out_w_per_slice*device.HBM_1Row_Bytes
        tp_feature_out_base_addr=feature_out_base_addr +w*out_w_per_slice*device.HBM_1Row_Bytes
       
        w_for += CSB_Write(Ln_reg_bias+2 , wt_base_addr              )
        w_for += CSB_Write(Ln_reg_bias+3 , tp_feature_in_base_addr   )
        w_for += CSB_Write(Ln_reg_bias+4 , feature_in_surface_stride )
        w_for += CSB_Write(Ln_reg_bias+5 , feature_in_line_stride    )
        w_for += CSB_Write(Ln_reg_bias+6 , tp_feature_out_base_addr  )
        w_for += CSB_Write(Ln_reg_bias+7 , feature_out_surface_stride)
        w_for += CSB_Write(Ln_reg_bias+8 , feature_out_line_stride   )
        w_for += CSB_Write(Ln_reg_bias+9 , CHin                      )
        w_for += CSB_Write(Ln_reg_bias+10, Hin                       )
        w_for += CSB_Write(Ln_reg_bias+11, Win                       )
        w_for += CSB_Write(Ln_reg_bias+12, pixel_in                  )
        w_for += CSB_Write(Ln_reg_bias+13, FP20_recip_CH_r           )
        w_for += CSB_Write(Ln_reg_bias+14, 0                         )
        w_for += CSB_Write(Ln_reg_bias+15, Layer_Norm                )
        w_for += CSB_Write(Ln_reg_bias+16, 0                         )
        w_for += CSB_Write(Ln_reg_bias+17, 0b100000                  )
   
        w_for += While(CSB_Read(Ln_reg_bias+1)!=1) 
    func += w_for


