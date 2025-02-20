#include "process.h"

HBM_DLL int** WT_TRANS(int chin, int chout, int ch_size, int *wt, uint16_t *wt_FP_scale)
{
    int** HBM_DDR = (int**)malloc(sizeof(int*)*32);
    for (int i = 0; i < 32; i++) {
        HBM_DDR[i] = (int*)malloc(ch_size);
    }
    int CHout                         = chout;
    int CHin                          = chin;
    int CHout_div_Tout                = ((chout+Tout-1)/Tout);
    int WT_CHin_div_Tin               = ((chin+Tin-1)/Tin);
    int WT_CHin_Padding_with_Tin      = WT_CHin_div_Tin*Tin;
    int WT_scale_group_nums           = ((WT_CHin_Padding_with_Tin+WT_CH_Tgroup-1)/WT_CH_Tgroup);

    int WT_CH_Tgroup_div_Tblock       = (WT_CH_Tgroup/T_quant_block);
    int WT_CHin_div_Tblock            = ((WT_CHin_Padding_with_Tin+T_quant_block-1)/T_quant_block);
    int CHin_WT_Bytes                 = (WT_CHin_Padding_with_Tin*WT_DW/8);
    int CHin_Scale_Bytes              = (HBM_AXI_DATA_WIDTH*WT_scale_group_nums/8);
    int CHin_WT_and_Scale_Bytes       = CHin_WT_Bytes+CHin_Scale_Bytes;

    int Group_WT_Bytes                = (WT_CH_Tgroup*WT_DW/8);
    int Group_Scale_Bytes             = (HBM_AXI_DATA_WIDTH/8);
    int Group_WT_and_Scale_Bytes      = (Group_WT_Bytes+Group_Scale_Bytes);
    int Last_Group_Scale_Bytes        = (HBM_AXI_DATA_WIDTH/8);
    int Last_Group_CHin               = (WT_CHin_Padding_with_Tin%WT_CH_Tgroup);
    int Last_Group_WT_Bytes           = (Last_Group_CHin*WT_DW/8);
    int Last_Group_WT_and_Scale_Bytes = Last_Group_WT_Bytes+Last_Group_Scale_Bytes;

    int *HBM_wt_FP_scale = (int*)malloc(sizeof(int)*CHout_div_Tout*WT_scale_group_nums*Tout/HBM_Port*HBM_Port*(HBM_AXI_DATA_WIDTH/32));
    if (HBM_wt_FP_scale == NULL){printf("fail to malloc HBM_wt_FP_scale \n");}

    int  *HBM_wt_mem = (int*)malloc(sizeof(int)*CHout_div_Tout*Tout/HBM_Port*HBM_Port*(WT_DW*WT_CHin_Padding_with_Tin/32)); 
    if (HBM_wt_mem == NULL){printf("fail to malloc HBM_wt_mem \n");}

    int wt_start_ch_in;
    int wt_end_ch_in;
    int wt_addr_bias;
    int tmp_wt;

    for(int i=0;i<CHout_div_Tout;i++) {
        for(int j=0;j<Tout/HBM_Port;j++) {
            for(int k=0;k<HBM_Port;k++) {
                for(int m=0;m<WT_CHin_Padding_with_Tin*WT_DW/32;m++) {
                    int tmp = 0;
                    for(int p=0; p<8; p++) {
                        if((i*Tout+j*HBM_Port+k<CHout) && (m*8+p < CHin)) {
                            if(wt[(CHin*(i*Tout+j*HBM_Port+k))+m*8+p]<0) {
                                   tmp_wt=8-wt[(CHin*(i*Tout+j*HBM_Port+k))+m*8+p];
                            } else {
                                  tmp_wt=wt[(CHin*(i*Tout+j*HBM_Port+k))+m*8+p];
                            }
                            tmp = tmp + ( (tmp_wt&(0x0000000f)) << WT_DW*p ); 
                        }
                    }
                    HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + j*HBM_Port + k)*(WT_DW*WT_CHin_Padding_with_Tin/32)+m] = tmp;
                }    
            }
        }
    }


    for(int i=0;i<CHout_div_Tout;i++) {
        for(int j=0;j<WT_scale_group_nums;j++) {
            for(int k=0;k<Tout/HBM_Port;k++) {
                for(int m=0;m<HBM_Port;m++) {
                    for(int n=0;n<WT_CH_Tgroup_div_Tblock*16/32;n++) {
                        int tmp = 0;
                        for(int p=0; p<2; p++) {
                            if((i*Tout+k*HBM_Port+m<chout) && (j*WT_CH_Tgroup_div_Tblock+n*2+p<WT_CHin_div_Tblock)) {
                               tmp = tmp +  (int(wt_FP_scale[i*WT_CHin_div_Tblock*Tout + (j*WT_CH_Tgroup_div_Tblock+n*2+p) + (k*HBM_Port+m)*WT_CHin_div_Tblock]) << p*WT_quant_scale_DW);                            
                            }
                        }
                        HBM_wt_FP_scale[(i*(WT_scale_group_nums*Tout/HBM_Port*HBM_Port) + j*(Tout/HBM_Port*HBM_Port) + k*HBM_Port + m)*WT_CH_Tgroup_div_Tblock*16/32+n]=tmp;
                    }
                }
            }
        }
    }

    for(int i=0;i<CHout_div_Tout;i++) {
        for(int j=0;j<WT_scale_group_nums;j++) {
            for(int k=0;k<Tout/HBM_Port;k++) {
				for(int m=0;m<HBM_Port;m++) {
					int scale_addr_bias=static_cast<int>((i*CHin_WT_and_Scale_Bytes*8/32+j*Group_WT_and_Scale_Bytes*8/32)*(Tout/HBM_Port)
					                + ((j==WT_scale_group_nums-1)? (k*Last_Group_WT_and_Scale_Bytes*8/32) : (k*Group_WT_and_Scale_Bytes*8/32)));
					for(int n=0;n<HBM_AXI_DATA_WIDTH/32;n++) {    
						if(m==  0){HBM_DDR[ 0][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m==  1){HBM_DDR[ 1][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m==  2){HBM_DDR[ 2][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m==  3){HBM_DDR[ 3][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m==  4){HBM_DDR[ 4][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m==  5){HBM_DDR[ 5][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m==  6){HBM_DDR[ 6][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m==  7){HBM_DDR[ 7][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m==  8){HBM_DDR[ 8][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m==  9){HBM_DDR[ 9][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m== 10){HBM_DDR[10][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m== 11){HBM_DDR[11][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m== 12){HBM_DDR[12][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m== 13){HBM_DDR[13][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m== 14){HBM_DDR[14][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m== 15){HBM_DDR[15][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m== 16){HBM_DDR[16][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m== 17){HBM_DDR[17][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m== 18){HBM_DDR[18][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m== 19){HBM_DDR[19][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m== 20){HBM_DDR[20][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m== 21){HBM_DDR[21][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m== 22){HBM_DDR[22][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m== 23){HBM_DDR[23][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m== 24){HBM_DDR[24][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m== 25){HBM_DDR[25][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m== 26){HBM_DDR[26][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m== 27){HBM_DDR[27][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m== 28){HBM_DDR[28][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m== 29){HBM_DDR[29][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m== 30){HBM_DDR[30][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m== 31){HBM_DDR[31][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                    }
                }
            }
       }
    }

    for(int i=0;i<CHout_div_Tout;i++) {
    	for(int j=0;j<WT_scale_group_nums;j++) {
    		for(int k=0;k<Tout/HBM_Port;k++) {
    			for(int m=0;m<HBM_Port;m++) {
                    wt_start_ch_in=j*WT_CH_Tgroup;
                    wt_end_ch_in=static_cast<int>(j==WT_scale_group_nums-1)?WT_CHin_Padding_with_Tin:(j+1)*WT_CH_Tgroup;
                    wt_addr_bias=static_cast<int>((i*CHin_WT_and_Scale_Bytes+j*Group_WT_and_Scale_Bytes)*8/32*(Tout/HBM_Port)+Group_Scale_Bytes*8/32
                                + ((j==WT_scale_group_nums-1)? (k*Last_Group_WT_and_Scale_Bytes*8/32) : (k*Group_WT_and_Scale_Bytes*8/32)));
                                //+cfg.HBM00_WT_BASE_ADDR/4+cfg.WT_base_addr_Bank_Step/4*m);
                    for(int n = WT_DW*wt_start_ch_in/32;n<WT_DW*wt_end_ch_in/32;n++) {
                        if( m == 0){ HBM_DDR[ 0][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];} //cfg.WT_DW*cfg.WT_CHin_Padding_with_Tin/32
                        if( m == 1){ HBM_DDR[ 1][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if( m == 2){ HBM_DDR[ 2][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if( m == 3){ HBM_DDR[ 3][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if( m == 4){ HBM_DDR[ 4][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if( m == 5){ HBM_DDR[ 5][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if( m == 6){ HBM_DDR[ 6][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if( m == 7){ HBM_DDR[ 7][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if( m == 8){ HBM_DDR[ 8][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if( m == 9){ HBM_DDR[ 9][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if(m == 10){ HBM_DDR[10][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if(m == 11){ HBM_DDR[11][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if(m == 12){ HBM_DDR[12][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if(m == 13){ HBM_DDR[13][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if(m == 14){ HBM_DDR[14][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if(m == 15){ HBM_DDR[15][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if(m == 16){ HBM_DDR[16][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if(m == 17){ HBM_DDR[17][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if(m == 18){ HBM_DDR[18][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if(m == 19){ HBM_DDR[19][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if(m == 20){ HBM_DDR[20][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if(m == 21){ HBM_DDR[21][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if(m == 22){ HBM_DDR[22][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if(m == 23){ HBM_DDR[23][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if(m == 24){ HBM_DDR[24][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if(m == 25){ HBM_DDR[25][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if(m == 26){ HBM_DDR[26][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if(m == 27){ HBM_DDR[27][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if(m == 28){ HBM_DDR[28][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if(m == 29){ HBM_DDR[29][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if(m == 30){ HBM_DDR[30][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if(m == 31){ HBM_DDR[31][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                    }
                }
            }
        }
    }

    return HBM_DDR;
}


HBM_DLL int** WT_TRANS_INT4(int chin, int chout, int ch_size, int *wt, uint16_t *wt_FP_scale)
{
    int** HBM_DDR = (int**)malloc(sizeof(int*)*32);
    for (int i = 0; i < 32; i++) {
        HBM_DDR[i] = (int*)malloc(ch_size);
    }
    int CHout                         = chout;
    int CHin                          = chin;
    int CHout_div_Tout                = ((chout+Tout-1)/Tout);
    int WT_CHin_div_Tin               = ((chin+Tin-1)/Tin);
    int WT_CHin_Padding_with_Tin      = WT_CHin_div_Tin*Tin;
    int WT_scale_group_nums           = ((WT_CHin_Padding_with_Tin+WT_CH_Tgroup-1)/WT_CH_Tgroup);

    int WT_CH_Tgroup_div_Tblock       = (WT_CH_Tgroup/T_quant_block);
    int WT_CHin_div_Tblock            = ((WT_CHin_Padding_with_Tin+T_quant_block-1)/T_quant_block);
    int CHin_WT_Bytes                 = (WT_CHin_Padding_with_Tin*WT_DW/8);
    int CHin_Scale_Bytes              = (HBM_AXI_DATA_WIDTH*WT_scale_group_nums/8);
    int CHin_WT_and_Scale_Bytes       = CHin_WT_Bytes+CHin_Scale_Bytes;

    int Group_WT_Bytes                = (WT_CH_Tgroup*WT_DW/8);
    int Group_Scale_Bytes             = (HBM_AXI_DATA_WIDTH/8);
    int Group_WT_and_Scale_Bytes      = (Group_WT_Bytes+Group_Scale_Bytes);
    int Last_Group_Scale_Bytes        = (HBM_AXI_DATA_WIDTH/8);
    int Last_Group_CHin               = (WT_CHin_Padding_with_Tin%WT_CH_Tgroup);
    int Last_Group_WT_Bytes           = (Last_Group_CHin*WT_DW/8);
    int Last_Group_WT_and_Scale_Bytes = Last_Group_WT_Bytes+Last_Group_Scale_Bytes;

    int *HBM_wt_FP_scale = (int*)malloc(sizeof(int)*CHout_div_Tout*WT_scale_group_nums*Tout/HBM_Port*HBM_Port*(HBM_AXI_DATA_WIDTH/32));
    if (HBM_wt_FP_scale == NULL){printf("fail to malloc HBM_wt_FP_scale \n");}

    int  *HBM_wt_mem = (int*)malloc(sizeof(int)*CHout_div_Tout*Tout/HBM_Port*HBM_Port*(WT_DW*WT_CHin_Padding_with_Tin/32)); 
    if (HBM_wt_mem == NULL){printf("fail to malloc HBM_wt_mem \n");}

    int wt_start_ch_in;
    int wt_end_ch_in;
    int wt_addr_bias;
    int tmp_wt;
    int tmp_wt_op;

    for(int i=0;i<CHout_div_Tout;i++) {
        for(int j=0;j<Tout/HBM_Port;j++) {
            for(int k=0;k<HBM_Port;k++) {
                for(int m=0;m<WT_CHin_Padding_with_Tin*WT_DW/32;m++) {
                    int tmp = 0;
                    for(int p=0; p<8; p++) {
                        if((i*Tout+j*HBM_Port+k<CHout) && (m*8+p < CHin)) {
                            tmp_wt_op = (wt[(CHin*(i*Tout+j*HBM_Port+k))/8+m] >> (WT_DW * p)) & 0xf;
                            if (tmp_wt_op > 0x8) {
                                tmp_wt = (8 - int(tmp_wt_op | 0xfffffff0)) & 0xf;
                            } else {
                                tmp_wt = tmp_wt_op;
                            }
                            tmp = tmp + ( (tmp_wt&(0x0000000f)) << WT_DW*p ); 
                        }
                    }
                    HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + j*HBM_Port + k)*(WT_DW*WT_CHin_Padding_with_Tin/32)+m] = tmp;
                }    
            }
        }
    }


    for(int i=0;i<CHout_div_Tout;i++) {
        for(int j=0;j<WT_scale_group_nums;j++) {
            for(int k=0;k<Tout/HBM_Port;k++) {
                for(int m=0;m<HBM_Port;m++) {
                    for(int n=0;n<WT_CH_Tgroup_div_Tblock*16/32;n++) {
                        int tmp = 0;
                        for(int p=0; p<2; p++) {
                            if((i*Tout+k*HBM_Port+m<chout) && (j*WT_CH_Tgroup_div_Tblock+n*2+p<WT_CHin_div_Tblock)) {
                               tmp = tmp +  (int(wt_FP_scale[i*WT_CHin_div_Tblock*Tout + (j*WT_CH_Tgroup_div_Tblock+n*2+p) + (k*HBM_Port+m)*WT_CHin_div_Tblock]) << p*WT_quant_scale_DW);                            
                            }
                        }
                        HBM_wt_FP_scale[(i*(WT_scale_group_nums*Tout/HBM_Port*HBM_Port) + j*(Tout/HBM_Port*HBM_Port) + k*HBM_Port + m)*WT_CH_Tgroup_div_Tblock*16/32+n]=tmp;
                    }
                }
            }
        }
    }

    for(int i=0;i<CHout_div_Tout;i++) {
        for(int j=0;j<WT_scale_group_nums;j++) {
            for(int k=0;k<Tout/HBM_Port;k++) {
				for(int m=0;m<HBM_Port;m++) {
					int scale_addr_bias=static_cast<int>((i*CHin_WT_and_Scale_Bytes*8/32+j*Group_WT_and_Scale_Bytes*8/32)*(Tout/HBM_Port)
					                + ((j==WT_scale_group_nums-1)? (k*Last_Group_WT_and_Scale_Bytes*8/32) : (k*Group_WT_and_Scale_Bytes*8/32)));
					for(int n=0;n<HBM_AXI_DATA_WIDTH/32;n++) {    
						if(m==  0){HBM_DDR[ 0][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m==  1){HBM_DDR[ 1][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m==  2){HBM_DDR[ 2][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m==  3){HBM_DDR[ 3][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m==  4){HBM_DDR[ 4][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m==  5){HBM_DDR[ 5][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m==  6){HBM_DDR[ 6][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m==  7){HBM_DDR[ 7][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m==  8){HBM_DDR[ 8][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m==  9){HBM_DDR[ 9][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m== 10){HBM_DDR[10][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m== 11){HBM_DDR[11][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m== 12){HBM_DDR[12][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m== 13){HBM_DDR[13][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m== 14){HBM_DDR[14][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m== 15){HBM_DDR[15][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m== 16){HBM_DDR[16][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m== 17){HBM_DDR[17][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m== 18){HBM_DDR[18][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m== 19){HBM_DDR[19][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m== 20){HBM_DDR[20][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m== 21){HBM_DDR[21][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m== 22){HBM_DDR[22][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m== 23){HBM_DDR[23][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m== 24){HBM_DDR[24][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m== 25){HBM_DDR[25][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m== 26){HBM_DDR[26][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m== 27){HBM_DDR[27][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m== 28){HBM_DDR[28][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m== 29){HBM_DDR[29][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m== 30){HBM_DDR[30][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                        if(m== 31){HBM_DDR[31][scale_addr_bias+n]=(HBM_wt_FP_scale[(i*WT_scale_group_nums*Tout/HBM_Port*HBM_Port + j*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*HBM_AXI_DATA_WIDTH/32+n]);}
                    }
                }
            }
       }
    }

    for(int i=0;i<CHout_div_Tout;i++) {
    	for(int j=0;j<WT_scale_group_nums;j++) {
    		for(int k=0;k<Tout/HBM_Port;k++) {
    			for(int m=0;m<HBM_Port;m++) {
                    wt_start_ch_in=j*WT_CH_Tgroup;
                    wt_end_ch_in=static_cast<int>(j==WT_scale_group_nums-1)?WT_CHin_Padding_with_Tin:(j+1)*WT_CH_Tgroup;
                    wt_addr_bias=static_cast<int>((i*CHin_WT_and_Scale_Bytes+j*Group_WT_and_Scale_Bytes)*8/32*(Tout/HBM_Port)+Group_Scale_Bytes*8/32
                                + ((j==WT_scale_group_nums-1)? (k*Last_Group_WT_and_Scale_Bytes*8/32) : (k*Group_WT_and_Scale_Bytes*8/32)));
                                //+cfg.HBM00_WT_BASE_ADDR/4+cfg.WT_base_addr_Bank_Step/4*m);
                    for(int n = WT_DW*wt_start_ch_in/32;n<WT_DW*wt_end_ch_in/32;n++) {
                        if( m == 0){ HBM_DDR[ 0][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];} //cfg.WT_DW*cfg.WT_CHin_Padding_with_Tin/32
                        if( m == 1){ HBM_DDR[ 1][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if( m == 2){ HBM_DDR[ 2][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if( m == 3){ HBM_DDR[ 3][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if( m == 4){ HBM_DDR[ 4][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if( m == 5){ HBM_DDR[ 5][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if( m == 6){ HBM_DDR[ 6][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if( m == 7){ HBM_DDR[ 7][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if( m == 8){ HBM_DDR[ 8][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if( m == 9){ HBM_DDR[ 9][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if(m == 10){ HBM_DDR[10][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if(m == 11){ HBM_DDR[11][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if(m == 12){ HBM_DDR[12][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if(m == 13){ HBM_DDR[13][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if(m == 14){ HBM_DDR[14][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if(m == 15){ HBM_DDR[15][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if(m == 16){ HBM_DDR[16][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if(m == 17){ HBM_DDR[17][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if(m == 18){ HBM_DDR[18][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if(m == 19){ HBM_DDR[19][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if(m == 20){ HBM_DDR[20][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if(m == 21){ HBM_DDR[21][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if(m == 22){ HBM_DDR[22][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if(m == 23){ HBM_DDR[23][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if(m == 24){ HBM_DDR[24][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if(m == 25){ HBM_DDR[25][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if(m == 26){ HBM_DDR[26][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if(m == 27){ HBM_DDR[27][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if(m == 28){ HBM_DDR[28][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if(m == 29){ HBM_DDR[29][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if(m == 30){ HBM_DDR[30][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                        if(m == 31){ HBM_DDR[31][wt_addr_bias+n-(WT_DW*wt_start_ch_in/32)]=HBM_wt_mem[(i*Tout/HBM_Port*HBM_Port + k*HBM_Port + m)*(WT_DW*WT_CHin_Padding_with_Tin/32)+n];}
                    }
                }
            }
        }
    }

    return HBM_DDR;
}


HBM_DLL int* BN_TRANS(int chout, int ch_size, uint16_t *bn_wt, uint16_t *bn_bias)
{
    int *DDR = (int*)malloc(ch_size);
    int CHout_div_Tout                = ((chout+Tout-1)/Tout);
    int CHout_Padding                 = (CHout_div_Tout*Tout);
    int BN_num_per_AXI_DW             = (AXI_DAT_WIDTH/(2*BN_DW));
    int BN_ch_group_times             = (CHout_Padding/BN_num_per_AXI_DW);
    int *bn_wt_and_bias = (int*)malloc(sizeof(int)*CHout_div_Tout*Tout);
    if (bn_wt_and_bias == NULL){printf("fail to malloc bn_wt_and_bias \n");}

    int *bn_wt_and_bias_mem[AXI_DAT_WIDTH/32];
    for(int i=0;i<AXI_DAT_WIDTH/32;i++) {
        bn_wt_and_bias_mem[i] = (int*)malloc(sizeof(int)*BN_ch_group_times);
        if (bn_wt_and_bias_mem[i] == NULL){printf("fail to malloc bn_wt_and_bias_mem \n");}
    }

    for(int i=0;i<CHout_div_Tout*Tout;i++) {
        bn_wt_and_bias[i] = (int)(bn_wt[i]<<16) + (int)bn_bias[i];
    }

    for(int i=0;i<BN_ch_group_times;i++) {
		for(int j=0;j<BN_num_per_AXI_DW;j++) {
			if(AXI_DAT_WIDTH>=(2*BN_DW))
                bn_wt_and_bias_mem[j][i] = bn_wt_and_bias[i*BN_num_per_AXI_DW+j];
	        else
                printf("Error!! AXI_DAT_WIDTH=%d, 2*BN_DW=%d,  AXI_DAT_WIDTH too small!",AXI_DAT_WIDTH, (2*BN_DW));
        }
    }

	for(int i=0;i<BN_ch_group_times;i++) {
		for(int j=0;j<AXI_DAT_WIDTH/32;j++) {
            DDR[i*AXI_DAT_WIDTH/32+j] = bn_wt_and_bias_mem[j][i];
        }
    }
    return DDR;
}

HBM_DLL int FP32_to_FP20(float fp32_i) {
    int fp32_i_s, fp32_i_e, fp32_i_f;
    int fp20_o_s, fp20_o_e, fp20_o_m_tmp, fp20_o_m;
    int overflow_tmp, underflow_tmp;

    fp32_i_s = 1 & (*((int*)&fp32_i) >> 31); // {1{1'b1}}
    fp32_i_e = 255 & (*((int*)&fp32_i) >> 23); // {8{1'b1}}
    fp32_i_f = 8388607 & (*((int*)&fp32_i)); // {23{1'b1}}

    if (fp32_i_e < 84) {
        fp20_o_s = 0;
        fp20_o_e = 0;
        fp20_o_m = 0;
        overflow_tmp = 0;
        underflow_tmp = 1;
    }
    else if (fp32_i_e >= 159) {
        fp20_o_s = fp32_i_s;
        fp20_o_e = 0b111110;
        fp20_o_m = 0b1111111111111;
        overflow_tmp = 1;
        underflow_tmp = 0;
    }
    else {
        fp20_o_s = fp32_i_s;
        overflow_tmp = 0;
        underflow_tmp = 0;
        if (fp32_i_e >= 97) {
            fp20_o_e = fp32_i_e - 96;
            int c_tmp = 1 & (fp32_i_f >> 9);
            if (c_tmp)
                fp20_o_m_tmp = fp32_i_f + 0b10000000000;
            else
                fp20_o_m_tmp = fp32_i_f;
            fp20_o_m = ((1 << 13) - 1) & (fp20_o_m_tmp >> 10);
        }
        else {
            fp20_o_e = 0;
            int r_cnt = 97 - fp32_i_e;
            int i_m_tmp0 = (1 << 23) + fp32_i_f;
            int c_tmp = 1 & (i_m_tmp0 >> (9 + r_cnt));
            if (c_tmp) {
                int i_m_tmp1 = (i_m_tmp0 >> r_cnt) + 0b10000000000;
                fp20_o_m = ((1 << 13) - 1) & (i_m_tmp1 >> 10);
            }
            else {
                int i_m_tmp1 = (i_m_tmp0 >> r_cnt);
                fp20_o_m = ((1 << 13) - 1) & (i_m_tmp1 >> 10);
            }
        }
    }

    return (fp20_o_s << 19) + (fp20_o_e << 13) + fp20_o_m;
}


HBM_DLL int** test(int* in) {
    int** HBM_DDR = (int**)malloc(sizeof(int*)*32);
    for (int i = 0; i < 32; i++) {
        HBM_DDR[i] = (int*)malloc(sizeof(int)*2);
        HBM_DDR[i][0] = in[i];
        HBM_DDR[i][1] = in[i] + 1;
        printf("%d, %d\n", i, HBM_DDR[i][0]);
    }
    return HBM_DDR;
}
