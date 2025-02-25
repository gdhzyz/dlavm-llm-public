import math
from dlavm import ne
from dlavm.adr import Op, Attrs
from dlavm.device import ohbm_accel
from ... import ir
from ...ir import CSB_Write, CSB_Read, While
from ...basic import Tasks, Ceil, Ceil_Padding

@Tasks.Register("atom.hbm.pcie2mem", ohbm_accel.OHBM)
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


@Tasks.Register("atom.ohbm.nn.mvm", ohbm_accel.OHBM)
def AtomMVMSingleTime(block, CHin, Hin, Win, CHout_offset, CHout, Hout, Wout, relu_en,
                      wt_reuse, dat_reuse, wt_ch_group_reg, t_quant_block_reg, Last_Group_CHin,
                      feature_in_addr, feature_in_surface_stride, feature_in_line_stride,
                      wt_base_addr, wt_bits_in_one_CHout,
                      feature_out_addr, feature_out_surface_stride, feature_out_line_stride):
    onchip_mode = 0
    block += CSB_Write (2 , CHin                      )
    block += CSB_Write (3 , Win                       )
    block += CSB_Write (4 , Hin                       )
    block += CSB_Write (5 , Wout                      )
    block += CSB_Write (6 , Hout                      )
    block += CSB_Write (7 , CHout                     )
    block += CSB_Write (8 , CHout_offset              )
    block += CSB_Write (9 , 0                         )
    block += CSB_Write (10, feature_in_addr           )
    block += CSB_Write (11, wt_base_addr              )
    block += CSB_Write (12, wt_bits_in_one_CHout      )
    block += CSB_Write (13, feature_out_addr          )
    block += CSB_Write (14, relu_en                   )
    block += CSB_Write (15, onchip_mode               )
    block += CSB_Write (16, dat_reuse*2+wt_reuse      )
    block += CSB_Write (17, 0                         )
    block += CSB_Write (18, 0                         )

    block += CSB_Write (19, 0                         )
    block += CSB_Write (20, 0                         )
    block += CSB_Write (21, 0                         )
    block += CSB_Write (22, wt_ch_group_reg           )
    block += CSB_Write (23, t_quant_block_reg         )
    block += CSB_Write (24, Last_Group_CHin           )

    block += CSB_Write (25, 0                         )
    block += CSB_Write (26, 0                         )
    block += CSB_Write (27, 0                         )
    block += CSB_Write (28, 0                         )
    block += CSB_Write (29, feature_in_surface_stride )
    block += CSB_Write (30, feature_in_line_stride    )
    block += CSB_Write (31, feature_out_surface_stride)
    block += CSB_Write (32, feature_out_line_stride   )

    block += CSB_Write (33, 0b001111                  )
    block += While(CSB_Read(1)!=1) 


@Tasks.Register("ohbm.nn.mvm", ohbm_accel.OHBM)
def MVM(func, args, outputs, attrs):
    if len(args) > 2:
        raise RuntimeError("not support bn and res in mvm")
    device = args[0].device
    data, weight = args

    # macro define testbench
    WT_DW = device.MAX_WT_DW
    Sparsity_Factor = 1
    Height, Width_in = data.shape[-2:]
    Width_out = weight.shape[0]
    feature_in_addr = data.get_address()
    wt_base_addr = weight.get_address()
    feature_out_addr = outputs[0].get_address()
    relu_en = attrs.get("relu", 0)

    # pre process
    Tb = 1
    Hin, Win = 1, Height
    CHin = Ceil_Padding(Width_in, device.Tout)
    Hout, Wout = 1, Height
    CHout = Ceil_Padding(Width_out, device.Tout)
    CHin_div_LTout = Ceil(CHin, device.L_Tout)
    CHin_Padding = CHin_div_LTout * device.L_Tout
    CHout_div_Tout = Ceil(CHout, device.Tout)
    WT_CHin_div_Tin = Ceil(CHin, device.Tin)
    WT_CHin_Padding_with_Tin = WT_CHin_div_Tin*device.Tin
    WT_CHout_Padding_with_Tout = CHout_div_Tout*device.Tout

    WT_CH_Tgroup = (device.T_quant_block*Sparsity_Factor*device.HBM_AXI_DATA_WIDTH//device.WT_quant_scale_DW)
    WT_scale_group_nums = ((WT_CHin_Padding_with_Tin+WT_CH_Tgroup-1)//WT_CH_Tgroup)
    WT_scale_bits = (WT_CHout_Padding_with_Tout*device.HBM_AXI_DATA_WIDTH*WT_scale_group_nums)
    WT_SIZE_IN_BYTES = (((WT_CHout_Padding_with_Tout*WT_CHin_Padding_with_Tin*WT_DW)>>3)+((WT_scale_bits)>>3))
    WT_BYTES_per_CH = WT_SIZE_IN_BYTES//WT_CHout_Padding_with_Tout
    log2_WT_CH_Tgroup = int(math.log2(WT_CH_Tgroup))
    Last_Group_CHin = ne.If(WT_CHin_Padding_with_Tin%WT_CH_Tgroup, WT_CHin_Padding_with_Tin%WT_CH_Tgroup, WT_CH_Tgroup)

    # stride load or compute
    feature_in_line_stride = device.Pixel_Data_Bytes * Win
    feature_in_surface_stride = device.Pixel_Data_Bytes * Win * Hin
    feature_out_line_stride = device.Pixel_Data_Bytes * Wout
    feature_out_surface_stride = device.Pixel_Data_Bytes * Wout * Hout
    if hasattr(data, "strides"):
        feature_in_line_stride = data.strides[-1]
        feature_in_surface_stride = data.strides[-2]
    if hasattr(outputs[0], "strides"):
        feature_out_line_stride = outputs[0].strides[-1]
        feature_out_surface_stride = outputs[0].strides[-2]

    # loop define
    dat_bits_per_row=Win*CHin_Padding*device.MAX_DAT_DW
    min_dat_depth   =dat_bits_per_row//(device.HBM_AXI_DATA_WIDTH*device.HBM_Port)
    wt_bits_per_Tout=WT_CHin_Padding_with_Tin*device.Tout*device.MAX_WT_DW
    min_wt_depth    =wt_bits_per_Tout//(device.HBM_AXI_DATA_WIDTH*device.HBM_Port*device.ASYN_FACTOR)

    if min_wt_depth>device.ID1_BRAM_DEPTH:
        print("ohbm.nn.mvm driver task error:")
        print("=======================================================================")
        print("================ FPGA WT BRAM DEPTH not enough!    ====================")
        print("=======================================================================")
        exit(-1)

    # ir: create local variable
    min_dat_depth = func.assign("min_dat_depth", min_dat_depth, "int")

    Wout_Split_Times_minus1 = func.assign("Wout_Split_Times_minus1", 0, "int")
    out_w_slice             = func.assign("out_w_slice", Wout, "int")
    out_w_slice_last        = func.assign("out_w_slice_last", Wout, "int")
    t_if = ir.If(min_dat_depth>device.ID0_BRAM_DEPTH)
    with t_if.then_block as _then:
        out_w_slice            =_then.assign_var(out_w_slice, device.TOTAL_DAT_BRAM_BITS//(CHin_Padding*device.MAX_DAT_DW))
        out_w_slice_last       =_then.assign_var(out_w_slice_last, ne.If(Wout%out_w_slice, Wout%out_w_slice, out_w_slice))
        Wout_Split_Times_minus1=_then.assign_var(Wout_Split_Times_minus1, (Wout+out_w_slice-1)//out_w_slice-1)
    func += t_if

    out_ch_slice=(device.TOTAL_WT_BRAM_BITS//wt_bits_per_Tout)*device.Tout
    if out_ch_slice>=CHout:
        out_ch_slice=CHout
        CHout_Split_Times=1
    else:
        CHout_Split_Times=(CHout+out_ch_slice-1)//out_ch_slice

    if CHout%out_ch_slice==0:
        out_ch_slice_last=out_ch_slice
    else:
        out_ch_slice_last=CHout%out_ch_slice
        
    CHout_Split_Times_minus1=CHout_Split_Times-1
    total_clks_if_reuse_wt=Tb*CHin_Padding*Win*Hin*(CHout_Split_Times_minus1+1)*device.MAX_DAT_DW//(device.HBM_AXI_DATA_WIDTH*device.HBM_Port) \
                          +WT_SIZE_IN_BYTES//(device.HBM_AXI_DATA_WIDTH*device.HBM_Port)+20
    total_clks_if_reuse_dat=Tb*CHin_Padding*Win*Hin*device.MAX_DAT_DW//(device.HBM_AXI_DATA_WIDTH*device.HBM_Port) \
                          +WT_SIZE_IN_BYTES*(Wout_Split_Times_minus1+1)//(device.HBM_AXI_DATA_WIDTH*device.HBM_Port)+20
    total_clks_if_reuse_wt =func.assign("total_clks_if_reuse_wt", total_clks_if_reuse_wt, "int")
    total_clks_if_reuse_dat=func.assign("total_clks_if_reuse_dat", total_clks_if_reuse_dat, "int")

    wt_ch_group_reg = (log2_WT_CH_Tgroup << device.log2_CH) + WT_CH_Tgroup
    t_quant_block_reg = (device.log2_T_quant_block << device.log2_CH) + device.T_quant_block
    t_if = ir.If(total_clks_if_reuse_wt < total_clks_if_reuse_dat)
    with t_if.then_block as _then:
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
                task = Tasks.Get("atom.ohbm.nn.mvm", device)
                task(w_for, CH_in_now, 1, W_in_now, CH_out_offset, CH_out_now, 1, W_out_now, relu_en,
                     dma_wt_reuse_now, dma_dat_reuse_now, wt_ch_group_reg, t_quant_block_reg, Last_Group_CHin,
                     feature_in_addr+device.Pixel_Data_Bytes*W_in_offset, feature_in_surface_stride, feature_in_line_stride,
                     wt_base_addr+WT_BYTES_per_CH//device.HBM_Port*out_ch_slice*ch, WT_BYTES_per_CH*8,
                     feature_out_addr+device.Pixel_Data_Bytes*W_out_offset+(device.Pixel_Data_Bytes*out_w_slice)*(ch*out_ch_slice//device.L_Tout),
                     feature_out_surface_stride, feature_out_line_stride)
            ch_for += w_for
        _then += ch_for
    with t_if.else_block as _else:
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
                task = Tasks.Get("atom.ohbm.nn.mvm", device)
                task(ch_for, CH_in_now, 1, W_in_now, CH_out_offset, CH_out_now, 1, W_out_now, relu_en,
                     dma_wt_reuse_now, dma_dat_reuse_now, wt_ch_group_reg, t_quant_block_reg, Last_Group_CHin,
                     feature_in_addr+device.Pixel_Data_Bytes*W_in_offset, feature_in_surface_stride, feature_in_line_stride,
                     wt_base_addr+WT_BYTES_per_CH//device.HBM_Port*out_ch_slice*ch, WT_BYTES_per_CH*8,
                     feature_out_addr+device.Pixel_Data_Bytes*W_out_offset+(device.Pixel_Data_Bytes*out_w_slice)*(ch*out_ch_slice//device.L_Tout),
                     feature_out_surface_stride, feature_out_line_stride)
            w_for += ch_for
        _else += w_for
    func += t_if
