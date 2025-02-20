from .basic import *


def RunConv(CHin, Hin, Win, CHout, Kx, Ky, Sx, Sy, pad_x, pad_y, relu_en, L0_DW, L1_DW, 
            feature_in_base, wt_base_addr, feature_out_base, feature_in_scale, wt_scale, conv_out_scale, feature_out_scale):
    out_width = (Win+2*pad_x-Kx)//Sx+1
    out_height = (Hin+2*pad_y-Ky)//Sy+1
    Tin_L0 = base_Tin*(MAX_DAT_DW // L0_DW)
    feature_in_surface_stride, feature_in_line_stride = Pixel_Data_Bytes*Hin*Win, Pixel_Data_Bytes*Win
    wt_size_in_bytes = ((Tin_L0*L0_DW)>>3)*Kx*Ky*CHout*((CHin+Tin_L0-1)//Tin_L0)
    wt_num_div_Tin = Kx*Ky*CHout*((CHin+Tin_L0-1)//Tin_L0)
    feature_out_surface_stride, feature_out_line_stride = Pixel_Data_Bytes*out_width*out_height, Pixel_Data_Bytes*out_width
    slice_of_CHin_L0  = (CHin+Tin_L0-1)//Tin_L0

    mininum_bw = 0

    overlap = Ky-Sy
    dat_num_per_row = Win * slice_of_CHin_L0
    dat_banks_min = (dat_num_per_row*Ky+BRAM_DEPTH-1)//BRAM_DEPTH
    wt_banks_min = (Kx*Ky*Tout*slice_of_CHin_L0+BRAM_DEPTH-1)//BRAM_DEPTH

    if(dat_banks_min+wt_banks_min) > BRAM_NUM:
        print("=======================================================================")
        print("===================   FPGA BRAM not enough!    ========================")
        print("=======================================================================")
        print("dat_banks_min=%0d,wt_banks_min=%0d"%(dat_banks_min,wt_banks_min))

    for dat_buf_num in range(dat_banks_min, BRAM_NUM-wt_banks_min+1):
        wt_banks = BRAM_NUM-dat_buf_num
        out_ch_slice = ((BRAM_DEPTH*wt_banks)//(Kx*Ky*Tout*slice_of_CHin_L0)) * Tout

        if out_ch_slice>=CHout:
            out_ch_slice=CHout
            CHout_Split_Times=1
        else:
            CHout_Split_Times=(CHout+out_ch_slice-1)//out_ch_slice

        if CHout%out_ch_slice==0:
            out_ch_slice_last=out_ch_slice 
        else:
            out_ch_slice_last=CHout%out_ch_slice

        out_height_first = ((BRAM_DEPTH*dat_buf_num)//dat_num_per_row+pad_y-Ky)//Sy+1
        in_height_first=(out_height_first-1)*Sy+Ky-pad_y

        out_height_middle=((BRAM_DEPTH*dat_buf_num)//dat_num_per_row-Ky)//Sy+1
        in_height_middle=(out_height_middle-1)*Sy+Ky

        if out_height_first>=out_height:
            out_height_first=out_height
            in_height_first=Hin

        if (out_height-out_height_first)%out_height_middle == 0:
            Hout_Split_Times=(out_height-out_height_first)//out_height_middle+1
            out_height_last=out_height_middle
        else:
            Hout_Split_Times=(out_height-out_height_first)//out_height_middle+2
            out_height_last=(out_height-out_height_first)%out_height_middle
        in_height_last=Hin-in_height_first+overlap-(Hout_Split_Times-2)*(in_height_first-overlap)
        total_bw_if_reuse_wt=(dat_num_per_row*Hin+dat_num_per_row*overlap*(Hout_Split_Times-1))*CHout_Split_Times+Kx*Ky*CHout*slice_of_CHin_L0
        total_bw_if_reuse_dat=Hout_Split_Times*Kx*Ky*CHout*slice_of_CHin_L0+dat_num_per_row*Hin+dat_num_per_row*overlap*(Hout_Split_Times-1)

        if (mininum_bw==0) or (total_bw_if_reuse_wt<mininum_bw):
            best_dat_banks=dat_buf_num
            mininum_bw=total_bw_if_reuse_wt
            best_method=0
        if (mininum_bw==0) or (total_bw_if_reuse_dat<mininum_bw):
            best_dat_banks=dat_buf_num
            mininum_bw=total_bw_if_reuse_dat
            best_method=1

    dat_buf_num = best_dat_banks

    wt_banks=BRAM_NUM-dat_buf_num
    out_ch_slice=((BRAM_DEPTH*wt_banks)//(Kx*Ky*Tout*slice_of_CHin_L0) ) * Tout

    if out_ch_slice>=CHout:
        out_ch_slice=CHout
        CHout_Split_Times=1
    else:
        CHout_Split_Times=(CHout+out_ch_slice-1)//out_ch_slice

    if CHout%out_ch_slice==0:
        out_ch_slice_last=out_ch_slice
    else:
        out_ch_slice_last=CHout%out_ch_slice

    out_height_first=((BRAM_DEPTH*dat_buf_num)//dat_num_per_row+pad_y-Ky)//Sy+1
    in_height_first=(out_height_first-1)*Sy+Ky-pad_y

    out_height_middle=((BRAM_DEPTH*dat_buf_num)//dat_num_per_row-Ky)//Sy+1
    in_height_middle=(out_height_middle-1)*Sy+Ky

    if out_height_first>=out_height:
        out_height_first=out_height
        in_height_first=Hin

    if((out_height-out_height_first)%out_height_middle == 0):
        Hout_Split_Times=(out_height-out_height_first)//out_height_middle+1
        out_height_last=out_height_middle
    else:
        Hout_Split_Times=(out_height-out_height_first)//out_height_middle+2
        out_height_last=(out_height-out_height_first)%out_height_middle

    in_height_last=Hin-in_height_first+overlap-(Hout_Split_Times-2)*(in_height_middle-overlap)

    regs = []
    if best_method==0:
        for n in range(0, CHout_Split_Times):
            for k in range(0, Hout_Split_Times):
                CH_in_single=CHin

                if n!=CHout_Split_Times-1:
                    CH_out_single=out_ch_slice
                else:
                    CH_out_single=out_ch_slice_last

                if k==0:
                    line_offset_in=0
                    line_offset_out=0
                    pad_y_single=pad_y
                    dma_wt_reuse_single=0
                else:
                    line_offset_in=(in_height_first-overlap)+(k-1)*(in_height_middle-overlap)
                    line_offset_out=out_height_first+(k-1)*out_height_middle
                    pad_y_single=0
                    dma_wt_reuse_single=1

                if k==0:
                    in_height_single=in_height_first
                    out_height_single=out_height_first
                else:
                    if k==Hout_Split_Times-1:
                        in_height_single=in_height_last
                        out_height_single=out_height_last
                    else:
                        in_height_single=in_height_middle
                        out_height_single=out_height_middle

                regs += RunConv_single_time(CH_in_single,in_height_single,Win,CH_out_single,
                        Kx,Ky,Sx,Sy,pad_x,pad_y_single,relu_en,L0_DW,L1_DW,
                        feature_in_base+feature_in_line_stride*line_offset_in,feature_in_surface_stride,feature_in_line_stride,feature_in_scale,
                        wt_base_addr+wt_size_in_bytes//CHout*out_ch_slice*n,wt_size_in_bytes//CHout*CH_out_single*8//(L0_DW*Tout),wt_num_div_Tin//CHout*CH_out_single,wt_scale,conv_out_scale, 
                        feature_out_base+feature_out_line_stride*line_offset_out+feature_out_surface_stride*n*(out_ch_slice//Tout),feature_out_surface_stride,feature_out_line_stride,feature_out_scale,
                        out_width,out_height_single,best_dat_banks,0,dma_wt_reuse_single, Tin_L0)
    else:
        for k in range(0, Hout_Split_Times):
            for n in range(0, CHout_Split_Times):
                CH_in_single=CHin

                if n!=CHout_Split_Times-1:
                    CH_out_single=out_ch_slice
                else:
                    CH_out_single=out_ch_slice_last

                if n==0:
                    dma_dat_reuse_single=0
                else:
                    dma_dat_reuse_single=1

                if k==0:
                    line_offset_in=0
                    line_offset_out=0
                    pad_y_single=pad_y
                else:
                    line_offset_in=(in_height_first-overlap)+(k-1)*(in_height_middle-overlap)
                    line_offset_out=out_height_first+(k-1)*out_height_middle
                    pad_y_single=0

                if k==0:
                    in_height_single=in_height_first
                    out_height_single=out_height_first
                else:
                    if k==Hout_Split_Times-1:
                        in_height_single=in_height_last
                        out_height_single=out_height_last
                    else:
                        in_height_single=in_height_middle
                        out_height_single=out_height_middle

                regs += RunConv_single_time(CH_in_single,in_height_single,Win,CH_out_single,
                        Kx,Ky,Sx,Sy,pad_x,pad_y_single,relu_en,L0_DW,L1_DW,
                        feature_in_base+feature_in_line_stride*line_offset_in,feature_in_surface_stride,feature_in_line_stride,feature_in_scale,
                        wt_base_addr+wt_size_in_bytes//CHout*out_ch_slice*n,wt_size_in_bytes//CHout*CH_out_single*8//(L0_DW*Tout),wt_num_div_Tin//CHout*CH_out_single,wt_scale,conv_out_scale,
                        feature_out_base+feature_out_line_stride*line_offset_out+feature_out_surface_stride*n*(out_ch_slice//Tout),feature_out_surface_stride,feature_out_line_stride,feature_out_scale,
                        out_width,out_height_single,best_dat_banks,dma_dat_reuse_single,0, Tin_L0)
    return regs


def RunConv_single_time(CHin, Hin, Win, CHout, Kx, Ky, Sx, Sy,
		 pad_x, pad_y, relu_en, L0_DW, L1_DW,
		 feature_in_base, feature_in_surface_stride, feature_in_line_stride, feature_in_scale,
		 wt_base_addr, wt_num_div_Tout, wt_num_div_Tin, wt_scale, conv_out_scale,										  
		 feature_out_base, feature_out_surface_stride, feature_out_line_stride,feature_out_scale,
		 out_width, out_height, dat_buf_num, dma_dat_reuse, dma_wt_reuse, Tin_L0):
    
    if (CHin%Tin_L0)==0:
        CH_in_res_Tin_div_Tout_minus1=(Tin_L0//Tout-1)
    else:
        CH_in_res_Tin_div_Tout_minus1=(((CHin%Tin_L0)+Tout-1)//Tout-1)
    
    Tin_factor=(MAX_DAT_DW//L0_DW)

    shift_sign=0
    shift_value=(feature_in_scale+wt_scale)-conv_out_scale
    if shift_value<0:
        shift_value=conv_out_scale-(feature_in_scale+wt_scale)
        shift_sign=1

    regs = []
    CSB_Write(regs, 2,Tin_factor)
    CSB_Write(regs, 3,dat_buf_num)
    CSB_Write(regs, 4,Win)
    CSB_Write(regs, 5,Hin)
    CSB_Write(regs, 6,Win*Hin)
    CSB_Write(regs, 7,(CHin+Tout-1)//Tout)
    CSB_Write(regs, 8,(CHin+Tin_L0-1)//Tin_L0)
    CSB_Write(regs, 31,CH_in_res_Tin_div_Tout_minus1)
    CSB_Write(regs, 9,pad_y)
    CSB_Write(regs, 10,pad_x)
    CSB_Write(regs, 11,Sx)
    CSB_Write(regs, 12,Sy)
    CSB_Write(regs, 13,Kx)
    CSB_Write(regs, 14,Ky)
    CSB_Write(regs, 15,out_width)
    CSB_Write(regs, 16,out_width*out_height)
    CSB_Write(regs, 17,out_height)
    CSB_Write(regs, 18,CHout)
    CSB_Write(regs, 19,((CHout+Tout-1)//Tout))
    CSB_Write(regs, 33,((out_width*out_height+Tout-1)//Tout))
    CSB_Write(regs, 34,feature_out_base)
    CSB_Write(regs, 35,feature_out_surface_stride)
    CSB_Write(regs, 36,feature_out_line_stride)		
    CSB_Write(regs, 38,relu_en)	
    CSB_Write(regs, 39,(L1_DW-1))	

    CSB_Write(regs, 20,feature_in_base)
    CSB_Write(regs, 21,feature_in_surface_stride)
    CSB_Write(regs, 22,feature_in_line_stride)
    CSB_Write(regs, 23,(wt_num_div_Tout//Tin_factor))
    CSB_Write(regs, 32,wt_num_div_Tin)
    CSB_Write(regs, 24,wt_base_addr)
    CSB_Write(regs, 25,(dma_wt_reuse<<1) + dma_dat_reuse)

    CSB_Write(regs, 26,(shift_sign << (log2_scale - 1)) + shift_value)

    CSB_Write(regs, 0, 0b011111)
    CSB_Read(regs, 37, 1)
    return regs