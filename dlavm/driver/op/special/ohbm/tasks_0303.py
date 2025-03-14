import math
from dlavm import ne
from dlavm.adr import Op, Attrs
from dlavm.device import ohbm_accel
from .... import ir
from ....ir import CSB_Write, CSB_Read, While
from ....basic import Tasks, Ceil, Ceil_Padding


#########################################################################################
#                                glm.pos_emb compute task                               #
#########################################################################################
@Tasks.Register("ohbm.glm.pos_emb", ohbm_accel.OHBM)
def PosEmb(func, args, outputs, attrs):
    device = args[0].device
    data, weight = args

    # macro define testbench
    WT_DW = device.MAX_WT_DW
    Sparsity_Factor = 1
    Feature_Head, Height, Width_in = data.shape[-3:]
    Width_out = Width_in
    feature_in_base_addr  = data.get_address()
    PosEmb_in_base_addr   = weight.get_address()
    feature_out_base_addr = outputs[0].get_address()
    last_token = attrs.get("last_token", 0)

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

    # TODO: Width_in must be MAX_CH_per_HEAD, and it should be verified before this module
    LTout_div_CHout = (device.L_Tout//Width_in)
    Head_x_CH_div_LTout = (Feature_Head+LTout_div_CHout-1)//LTout_div_CHout

    # stride load or compute
    Pos_Num                = device.MAX_TOKEN
    feature_in_line_stride = device.HBM_1Row_Bytes * Win
    feature_in_head_stride = device.HBM_1Row_Bytes * Win * Hin * CHin_div_LTout
    feature_out_line_stride = device.HBM_1Row_Bytes * Wout
    feature_out_head_stride = device.HBM_1Row_Bytes * Wout * Hout * CHout_div_LTout
    PosEmb_in_line_stride  = (device.HBM_1Row_Bytes*Pos_Num)
    PosEmb_in_head_stride  = (device.HBM_1Row_Bytes*Pos_Num*CHout_div_LTout)
    if hasattr(data, "strides"):
        feature_in_line_stride = data.strides[-1]
        feature_in_head_stride = data.strides[-3]
    if hasattr(outputs[0], "strides"):
        feature_out_line_stride = outputs[0].strides[-1]
        feature_out_head_stride = outputs[0].strides[-3]

    # task function
    PosEmb_reg_bias=128
    
    Onchip_Dat_BRAM_Bits    =device.TOTAL_DAT_BRAM_BITS
    Total_Dat_Bits_perWTHead=Height*CHin_Padding_with_LTout*device.MAX_DAT_DW
    Total_Dat_Bits_PerToken =CHin_Padding_with_LTout*device.MAX_DAT_DW
    Onchip_Token_perWTHead  =Onchip_Dat_BRAM_Bits//Total_Dat_Bits_PerToken
    Total_Wt_Bits_perWTHead =(Height)*Width_in*device.MAX_DAT_DW
    
    """
    TODO: limit the Height var if it is dynamic symbol
    if(Total_Wt_Bits_perWTHead>(`TOTAL_WT_BRAM_BITS/`LTout_div_CHout))
    begin
        $display("=======================================================================");
        $display("================ FPGA WT BRAM DEPTH not enough!    ====================");
        $display("=======================================================================");
        $finish;
    end
    """
    
    Wout_Split_Times_minus1 = func.assign("Wout_Split_Times_minus1", 0, "int")
    out_w_per_slice         = func.assign("out_w_per_slice", Wout, "int")
    out_w_in_last_slice     = func.assign("out_w_in_last_slice", Wout, "int")
    t_if = ir.If(Total_Dat_Bits_perWTHead>Onchip_Dat_BRAM_Bits)
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
        tp_PosEmb_addr          =PosEmb_in_base_addr+(last_token+w*out_w_per_slice)*device.HBM_1Row_Bytes
      
        reg_15 = (Head_x_CH_div_LTout << 8) + LTout_div_CHout
        w_for += CSB_Write(PosEmb_reg_bias+2 , tp_PosEmb_addr            )
        w_for += CSB_Write(PosEmb_reg_bias+3 , tp_feature_in_base_addr   )
        w_for += CSB_Write(PosEmb_reg_bias+4 , feature_in_head_stride    )
        w_for += CSB_Write(PosEmb_reg_bias+5 , feature_in_line_stride    )
        w_for += CSB_Write(PosEmb_reg_bias+6 , tp_feature_out_base_addr  )
        w_for += CSB_Write(PosEmb_reg_bias+7 , feature_out_head_stride   )
        w_for += CSB_Write(PosEmb_reg_bias+8 , feature_out_line_stride   )
        w_for += CSB_Write(PosEmb_reg_bias+9 , CHin                      )
        w_for += CSB_Write(PosEmb_reg_bias+10, Win                       )
        w_for += CSB_Write(PosEmb_reg_bias+11, 1                         ) # Qwen=0, GLM=1
        w_for += CSB_Write(PosEmb_reg_bias+12, Feature_Head              )
        w_for += CSB_Write(PosEmb_reg_bias+13, PosEmb_in_head_stride     )
        w_for += CSB_Write(PosEmb_reg_bias+14, PosEmb_in_line_stride     )
        w_for += CSB_Write(PosEmb_reg_bias+15, reg_15                    )
        w_for += CSB_Write(PosEmb_reg_bias+16, 0                         )
        w_for += CSB_Write(PosEmb_reg_bias+17, 0b000100                  )
   
        w_for += While(CSB_Read(PosEmb_reg_bias+1)!=1) 
    func += w_for


