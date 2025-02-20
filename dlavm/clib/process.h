#ifndef __DRIVER_PROCESS_H__
#define __DRIVER_PROCESS_H__

#include <cstdint>
#include <cstdlib>
#include <cstdio>

//#define EXPORT_DLL
#ifdef EXPORT_DLL
#define HBM_DLL __declspec(dllexport)
#else
#define HBM_DLL extern "C"
#endif

#define Tb 1
#define base_Tin 128
#define Tin base_Tin
#define Tout     32
#define T_quant_block 128
#define HBM_AXI_DATA_WIDTH 256
#define WT_quant_scale_DW 16
#define MAX_DAT_DW 16
#define WT_DW 4
#define BN_DW 16
#define DAT_BRAM_NUM 1
#define HBM_Port 32
#define WT_BRAM_NUM HBM_Port
#define AXI_BURST_LEN Tout
#define AXI_BN_WIDTH (MAX_BN_DW*Tout*Tb)
#define AXI_DAT_WIDTH (MAX_DAT_DW*Tout*Tb)
#define BN_FIFO_DEP ((AXI_BURST_LEN*MAX_DAT_DW*Tb)/(MAX_BN_DW*2))
#define BN_FIFO_NUM ((MAX_BN_DW*2)/(MAX_DAT_DW*Tb))
#define Pixel_Data_Bytes ((AXI_DAT_WIDTH)>>3)        
#define WT_CH_Tgroup (T_quant_block*HBM_AXI_DATA_WIDTH/WT_quant_scale_DW)
#define DAT_BRAM_DEPTH ((1<<22)/base_Tin/MAX_DAT_DW/DAT_BRAM_NUM)  //18: 256Kb for ASIC.
#define WT_BRAM_DEPTH ((1<<24)/HBM_AXI_DATA_WIDTH/WT_BRAM_NUM)  //18: 256Kb for ASIC.
#define BN_SURFACE_STRIDE ((Tout*MAX_BN_DW*2) >> 3)

HBM_DLL int** WT_TRANS(int chin, int chout, int ch_size, int *wt, uint16_t *wt_FP_scale);

HBM_DLL int** WT_TRANS_INT4(int chin, int chout, int ch_size, int *wt, uint16_t *wt_FP_scale);

HBM_DLL int* BN_TRANS(int chout, int ch_size, uint16_t *bn_wt, uint16_t *bn_bias);

HBM_DLL int FP32_to_FP20(float fp32_i);

HBM_DLL int** test(int* in);

#endif
