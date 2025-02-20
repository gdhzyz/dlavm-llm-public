from ..basic import CSB_Write, CSB_Read


def RunConvSingleTime(CHin, Hin, Win, CHout,
                        Kx, Ky, Sx, Sy,
                        pad_x, pad_y, relu_en, L0_DW, L1_DW,
                        feature_in_base, feature_in_surface_stride, feature_in_line_stride, feature_in_scale,
                        wt_base_addr, wt_num_div_Tout, wt_num_div_Tin, wt_scale, conv_out_scale,
                        feature_out_base, feature_out_surface_stride, feature_out_line_stride, feature_out_scale,
                        out_width, out_height, dat_buf_num, dma_dat_reuse, dma_wt_reuse, device, Sparsity, Sparsity_num):
    
    shift_value = 0
    shift_sign = 0
    Tin_L0 = device.MAX_DAT_DW // L0_DW
    
    if CHin % (Tin_L0 * device.Max_Sparsity) == 0:
        CH_in_res_Tin_div_Tin16_minus1 = device.Max_Sparsity - 1
    else:
        CH_in_res_Tin_div_Tin16_minus1 = ((CHin % (Tin_L0 * device.Max_Sparsity)) + Tin_L0 - 1) // Tin_L0 - 1
    
    if CHin % (Tin_L0 * device.Max_Sparsity) == 0:
        CH_in_res_Tin_div_SparsityNum_Tin16_minus1 = Sparsity // (16 / device.Max_Sparsity) - 1
    else:
        CH_in_res_Tin_div_SparsityNum_Tin16_minus1 = ((CHin % (Tin_L0 * device.Max_Sparsity)) + (Sparsity_num * Tin_L0) - 1) // (Sparsity_num * Tin_L0) - 1
    
    if CHin % (Tin_L0 * device.Max_Sparsity) == 0:
        CH_in_res_Tin_div_SparsityNum_mul2_minus1 = Sparsity - 1
    else:
        CH_in_res_Tin_div_SparsityNum_mul2_minus1 = (CH_in_res_Tin_div_Tin16_minus1 + Sparsity_num - 1) // Sparsity_num * 2 - 1
    
    if CHin % Tin_L0 == 0:
        CH_in_res_Tin_res_16_div_Tout_minus1 = Tin_L0 // device.Tout - 1
    else:
        CH_in_res_Tin_res_16_div_Tout_minus1 = (((CHin % (Tin_L0 * device.Max_Sparsity)) % Tin_L0) + device.Tout - 1) // device.Tout - 1

    Tin_factor = device.MAX_DAT_DW // L0_DW
    sparsity = Sparsity

    shift_sign = 0  # right shift
    shift_value = (feature_in_scale + wt_scale) - conv_out_scale
    if shift_value < 0:
        shift_value = conv_out_scale - (feature_in_scale + wt_scale)
        shift_sign = 1  # left shift

    regs = []
    CSB_Write(regs, 2, Tin_factor)
    CSB_Write(regs, 3, dat_buf_num)
    CSB_Write(regs, 4, Win)
    CSB_Write(regs, 5, Hin)
    CSB_Write(regs, 6, Win * Hin)
    CSB_Write(regs, 7, (CHin + device.Tout - 1) // device.Tout)
    CSB_Write(regs, 8, (CHin + (Tin_L0 * device.Max_Sparsity) - 1) // (Tin_L0 * device.Max_Sparsity))
    CSB_Write(regs, 31, CH_in_res_Tin_res_16_div_Tout_minus1)
    CSB_Write(regs, 41, CH_in_res_Tin_div_Tin16_minus1)
    CSB_Write(regs, 42, CH_in_res_Tin_div_SparsityNum_Tin16_minus1)
    CSB_Write(regs, 43, CH_in_res_Tin_div_SparsityNum_mul2_minus1)
    CSB_Write(regs, 9, pad_y)
    CSB_Write(regs, 10, pad_x)
    CSB_Write(regs, 11, Sx)
    CSB_Write(regs, 12, Sy)
    CSB_Write(regs, 13, Kx)
    CSB_Write(regs, 14, Ky)
    CSB_Write(regs, 15, out_width)
    CSB_Write(regs, 16, out_width * out_height)
    CSB_Write(regs, 17, out_height)
    CSB_Write(regs, 18, CHout)
    CSB_Write(regs, 19, (CHout + device.Tout - 1) // device.Tout)
    CSB_Write(regs, 33, ((out_width * out_height + device.Tout - 1) // device.Tout))
    CSB_Write(regs, 34, feature_out_base)
    CSB_Write(regs, 35, feature_out_surface_stride)
    CSB_Write(regs, 36, feature_out_line_stride)
    CSB_Write(regs, 38, relu_en)
    CSB_Write(regs, 39, (L1_DW - 1))
    CSB_Write(regs, 40, sparsity)
    CSB_Write(regs, 20, feature_in_base)
    CSB_Write(regs, 21, feature_in_surface_stride)
    CSB_Write(regs, 22, feature_in_line_stride)
    CSB_Write(regs, 23, (wt_num_div_Tout // Tin_factor))
    CSB_Write(regs, 32, wt_num_div_Tin)
    CSB_Write(regs, 24, wt_base_addr)
    CSB_Write(regs, 25, (dma_wt_reuse << 1) | dma_dat_reuse)
    CSB_Write(regs, 26, (shift_sign << 4) | shift_value)
    CSB_Write(regs, 0, 0b011111)

    CSB_Read(regs, 37, 1)
    return regs


def RunConv(fin, wt, fout, attrs):
    device = fin[0].device
    dshape, wshape, oshape = fin[0].shape, wt[0].shape, fout[0].shape
    daddrs, waddrs, oaddrs = fin[1] & 0xfffffff, wt[1] & 0xfffffff, fout[1] & 0xfffffff
    pad_y, pad_x, Sy, Sx = attrs["padding"][0], attrs["padding"][1], attrs["stride"][0], attrs["stride"][1]
    Hin, Win, _ = dshape
    Ky, Kx, CHin, CHout = wshape
    out_width = ((Win + 2 * pad_x - Kx) // Sx + 1)
    out_height = ((Hin + 2 * pad_y - Ky) // Sy + 1)
    overlap = Ky - Sy
    DAT_DW_L0, DAT_DW_L1 = attrs["widths"]
    feature_in_scale, wt_scale, feature_out_scale = attrs["scales"]
    relu_en = attrs["relu_en"]
    conv_out_scale = feature_out_scale
    Sparsity, Sparsity_num = attrs["sparsity"]
    Max_Sparsity, base_Tin, Tout, MAX_DAT_DW = device.Max_Sparsity, device.base_Tin, device.Tout, device.MAX_DAT_DW
    BRAM_DEPTH, BRAM_NUM = device.BRAM_DEPTH, device.BRAM_NUM
    Tin_L0 = base_Tin*(MAX_DAT_DW//DAT_DW_L0)
    slice_of_CHin_L0 = ((CHin+Tin_L0*Max_Sparsity-1)//(Tin_L0*Max_Sparsity))
    dat_num_per_row = Win * slice_of_CHin_L0
    dat_banks_min = (dat_num_per_row * Ky + BRAM_DEPTH - 1) // BRAM_DEPTH
    wt_banks_min = (Kx * Ky * Tout * slice_of_CHin_L0 + BRAM_DEPTH - 1) // BRAM_DEPTH
    WT_DW_L0 = DAT_DW_L0
    slice_of_CHout_L0 = ((CHout+Tout-1)//Tout)
    slice_of_CHin_L0_no_minus1 = ((CHin+Tin_L0*Max_Sparsity)//(Tin_L0*Max_Sparsity))
    CH_in_res_Tin_div_Tin16_L0 = (((CHin%(Tin_L0*Max_Sparsity))+Tin_L0-1)//Tin_L0)
    print(slice_of_CHin_L0_no_minus1, CH_in_res_Tin_div_Tin16_L0)
    wt_size_in_bytes = ((Tin_L0*WT_DW_L0)>>3)*Kx*Ky*CHout*((CHin+Tin_L0-1)//Tin_L0)
    if Sparsity in [2, 4, 8]:
        wt_size_in_bytes = (((Tin_L0*WT_DW_L0)>>3)*slice_of_CHout_L0*((slice_of_CHin_L0_no_minus1-1)*Sparsity+(CH_in_res_Tin_div_Tin16_L0+Sparsity_num-1)//Sparsity_num*2)*Ky*Kx*Tout)
    wt_num_div_Tin = (Kx*Ky*CHout*((CHin+Tin_L0*Max_Sparsity-1)//(Tin_L0*Max_Sparsity)))

    if (dat_banks_min + wt_banks_min) > BRAM_NUM:
        print("Resource limitation exceeded")
        return

    mininum_bw = 0
    best_dat_banks = 0
    best_method = 0

    for dat_buf_num in range(dat_banks_min, BRAM_NUM - wt_banks_min + 1):
        wt_banks = BRAM_NUM - dat_buf_num
        out_ch_slice = (BRAM_DEPTH * wt_banks // (Kx * Ky * Tout * slice_of_CHin_L0)) * Tout

        if out_ch_slice >= CHout:
            out_ch_slice = CHout
            CHout_Split_Times = 1
        else:
            CHout_Split_Times = (CHout + out_ch_slice - 1) // out_ch_slice

        out_ch_slice_last = CHout % out_ch_slice if CHout % out_ch_slice != 0 else out_ch_slice

        out_height_first = (((BRAM_DEPTH * dat_buf_num) // dat_num_per_row + pad_y - Ky) // Sy + 1)
        in_height_first = (out_height_first - 1) * Sy + Ky - pad_y

        out_height_middle = (((BRAM_DEPTH * dat_buf_num) // dat_num_per_row - Ky) // Sy + 1)
        in_height_middle = (out_height_middle - 1) * Sy + Ky

        if out_height_first >= out_height:
            out_height_first = out_height
            in_height_first = Hin

        Hout_Split_Times = (out_height - out_height_first) // out_height_middle + 1 if (out_height - out_height_first) % out_height_middle == 0 else (out_height - out_height_first) // out_height_middle + 2
        out_height_last = (out_height - out_height_first) % out_height_middle if (out_height - out_height_first) % out_height_middle != 0 else out_height_middle

        in_height_last = Hin - in_height_first + overlap - (Hout_Split_Times - 2) * (in_height_middle - overlap)
        total_bw_if_reuse_wt = (dat_num_per_row * Hin + dat_num_per_row * overlap * (Hout_Split_Times - 1)) * CHout_Split_Times + Kx * Ky * CHout * slice_of_CHin_L0
        total_bw_if_reuse_dat = Hout_Split_Times * Kx * Ky * CHout * slice_of_CHin_L0 + dat_num_per_row * Hin + dat_num_per_row * overlap * (Hout_Split_Times - 1)

        if mininum_bw == 0 or total_bw_if_reuse_wt < mininum_bw:
            best_dat_banks = dat_buf_num
            mininum_bw = total_bw_if_reuse_wt
            best_method = 0

        if mininum_bw == 0 or total_bw_if_reuse_dat < mininum_bw:
            best_dat_banks = dat_buf_num
            mininum_bw = total_bw_if_reuse_dat
            best_method = 1

    dat_buf_num = best_dat_banks

    wt_banks = BRAM_NUM - dat_buf_num
    out_ch_slice = (BRAM_DEPTH * wt_banks // (Kx * Ky * Tout * slice_of_CHin_L0)) * Tout

    if out_ch_slice >= CHout:
        out_ch_slice = CHout
        CHout_Split_Times = 1
    else:
        CHout_Split_Times = (CHout + out_ch_slice - 1) // out_ch_slice

    out_ch_slice_last = CHout % out_ch_slice if CHout % out_ch_slice != 0 else out_ch_slice

    out_height_first = (((BRAM_DEPTH * dat_buf_num) // dat_num_per_row + pad_y - Ky) // Sy + 1)
    in_height_first = (out_height_first - 1) * Sy + Ky - pad_y

    out_height_middle = (((BRAM_DEPTH * dat_buf_num) // dat_num_per_row - Ky) // Sy + 1)
    in_height_middle = (out_height_middle - 1) * Sy + Ky

    if out_height_first >= out_height:
        out_height_first = out_height
        in_height_first = Hin

    Hout_Split_Times = (out_height - out_height_first) // out_height_middle + 1 if (out_height - out_height_first) % out_height_middle == 0 else (out_height - out_height_first) // out_height_middle + 2
    out_height_last = (out_height - out_height_first) % out_height_middle if (out_height - out_height_first) % out_height_middle != 0 else out_height_middle

    in_height_last = Hin - in_height_first + overlap - (Hout_Split_Times - 2) * (in_height_middle - overlap)

    L0_DW, L1_DW = DAT_DW_L0, DAT_DW_L1
    regs = []
    feature_in_base, wt_base_addr, feature_out_base = daddrs, waddrs, oaddrs
    feature_in_surface_stride = device.Pixel_Data_Bytes * dshape[-1] * dshape[-2]
    feature_in_line_stride = device.Pixel_Data_Bytes * dshape[-1]
    feature_out_surface_stride = device.Pixel_Data_Bytes * oshape[-1] * oshape[-2]
    feature_out_line_stride = device.Pixel_Data_Bytes * oshape[-1]
    if best_method == 0:
        for n in range(CHout_Split_Times):
            for k in range(Hout_Split_Times):
                CH_in_single = CHin
                CH_out_single = out_ch_slice if n != CHout_Split_Times - 1 else out_ch_slice_last

                if k == 0:
                    line_offset_in = 0
                    line_offset_out = 0
                    pad_y_single = pad_y
                    dma_wt_reuse_single = 0
                else:
                    line_offset_in = (in_height_first - overlap) + (k - 1) * (in_height_middle - overlap)
                    line_offset_out = out_height_first + (k - 1) * out_height_middle
                    pad_y_single = 0
                    dma_wt_reuse_single = 1

                if k == 0:
                    in_height_single = in_height_first
                    out_height_single = out_height_first
                elif k == Hout_Split_Times - 1:
                    in_height_single = in_height_last
                    out_height_single = out_height_last
                else:
                    in_height_single = in_height_middle
                    out_height_single = out_height_middle

                regs += RunConvSingleTime(CH_in_single, in_height_single, Win, CH_out_single,
                                Kx, Ky, Sx, Sy, pad_x, pad_y_single, relu_en, L0_DW, L1_DW,
                                feature_in_base + feature_in_line_stride * line_offset_in, feature_in_surface_stride, feature_in_line_stride, feature_in_scale,
                                wt_base_addr + wt_size_in_bytes // CHout * out_ch_slice * n, wt_size_in_bytes // CHout * CH_out_single * 8 // (WT_DW_L0 * Tout), wt_num_div_Tin // CHout * CH_out_single, wt_scale, conv_out_scale,
                                feature_out_base + feature_out_line_stride * line_offset_out + feature_out_surface_stride * n * (out_ch_slice // Tout), feature_out_surface_stride, feature_out_line_stride, feature_out_scale,
                                out_width, out_height_single, best_dat_banks, 0, dma_wt_reuse_single, device, Sparsity, Sparsity_num)

    else:
        for k in range(Hout_Split_Times):
            for n in range(CHout_Split_Times):
                CH_in_single = CHin
                CH_out_single = out_ch_slice if n != CHout_Split_Times - 1 else out_ch_slice_last
                dma_dat_reuse_single = 0 if n == 0 else 1

                if k == 0:
                    line_offset_in, line_offset_out, pad_y_single = 0, 0, pad_y
                    in_height_single, out_height_single = in_height_first, out_height_first
                else:
                    line_offset_in = (in_height_first - overlap) + (k - 1) * (in_height_middle - overlap)
                    line_offset_out = out_height_first + (k - 1) * out_height_middle
                    pad_y_single = 0
                    if k == Hout_Split_Times - 1:
                        in_height_single, out_height_single = in_height_last, out_height_last
                    else:
                        in_height_single, out_height_single = in_height_middle, out_height_middle

                regs += RunConvSingleTime(CH_in_single, in_height_single, Win, CH_out_single,
                                    Kx, Ky, Sx, Sy, pad_x, pad_y_single, relu_en, L0_DW, L1_DW,
                                    feature_in_base + feature_in_line_stride * line_offset_in, feature_in_surface_stride, feature_in_line_stride, feature_in_scale,
                                    wt_base_addr + wt_size_in_bytes // CHout * out_ch_slice * n, wt_size_in_bytes // CHout * CH_out_single * 8 // (WT_DW_L0 * Tout), wt_num_div_Tin // CHout * CH_out_single, wt_scale, conv_out_scale,
                                    feature_out_base + feature_out_line_stride * line_offset_out + feature_out_surface_stride * n * (out_ch_slice // Tout), feature_out_surface_stride, feature_out_line_stride, feature_out_scale,
                                    out_width, out_height_single, best_dat_banks, dma_dat_reuse_single, 0, device, Sparsity, Sparsity_num)
       
    return regs
