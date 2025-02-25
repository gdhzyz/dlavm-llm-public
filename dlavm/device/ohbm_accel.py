import math
from functools import reduce
from dlavm.adr.base import DataType, DataEnum
from .. import ne
from .base_accel import Accel


class OHBM(Accel):
    name = "ohbm"
    version = 20240101

    Tb                          = 1
    Tout                        = 32
    HBM_Port                    = 32
    base_Tin                    = 128
    Tin                         = base_Tin
    MAX_DAT_DW                  = 16
    MAX_DW                      = 16
    HBM_AXI_DATA_WIDTH          = 256
    L_Tout                      = HBM_Port*HBM_AXI_DATA_WIDTH//MAX_DW
    ASYN_FACTOR                 = 2
    log2_CH                     = 19
    Pixel_Data_Width            = HBM_AXI_DATA_WIDTH
    Pixel_Data_Bytes            = int((HBM_AXI_DATA_WIDTH)>>3)

    DAT_BRAM_NUM                = HBM_Port
    log2_DAT_BRAM_NUM           = int(math.log2(DAT_BRAM_NUM))
    log2_TOTAL_DAT_BRAM_BITS    = (23) #23= 8Mb for VCU128, single BRAM buf is 512(depth)*72(width)= 36864 bit
    TOTAL_DAT_BRAM_BITS         = (1<<log2_TOTAL_DAT_BRAM_BITS)
    SINGLE_DAT_BRAM_BITS        = (TOTAL_DAT_BRAM_BITS//DAT_BRAM_NUM)  
    SINGLE_DAT_BRAM_WIDTH       = (HBM_AXI_DATA_WIDTH)
    SINGLE_DAT_BRAM_DEPTH       = (SINGLE_DAT_BRAM_BITS//SINGLE_DAT_BRAM_WIDTH)
    log2_SINGLE_DAT_BRAM_DEPTH  = int(math.log2(SINGLE_DAT_BRAM_DEPTH))
    log2_ID0_BRAM_DEPTH         = (log2_SINGLE_DAT_BRAM_DEPTH   )
    ID0_BRAM_DEPTH              = (1<<log2_ID0_BRAM_DEPTH       )
    ID0_BRAM_WIDTH              = (HBM_AXI_DATA_WIDTH*HBM_Port )

    log2_TOTAL_WT_BRAM_BITS     = (24) #24= 16Mb for VCU128, single BRAM buf is 512(depth)*72(width)= 36864 bit
    TOTAL_WT_BRAM_BITS          = (1<<log2_TOTAL_WT_BRAM_BITS)

    WT_BRAM_NUM                 = HBM_Port
    log2_WT_BRAM_NUM            = int(math.log2(WT_BRAM_NUM))
    log2_TOTAL_WT1_BRAM_BITS    = (log2_TOTAL_WT_BRAM_BITS-1)
    TOTAL_WT1_BRAM_BITS         = (1<<log2_TOTAL_WT1_BRAM_BITS)
    SINGLE_WT_BRAM_BITS         = (TOTAL_WT1_BRAM_BITS//WT_BRAM_NUM)  
    SINGLE_WT_BRAM_WIDTH        = (HBM_AXI_DATA_WIDTH)
    SINGLE_WT_BRAM_DEPTH        = (SINGLE_WT_BRAM_BITS//SINGLE_WT_BRAM_WIDTH)
    log2_SINGLE_WT_BRAM_DEPTH   = int(math.log2(SINGLE_WT_BRAM_DEPTH) )
    log2_ID1_BRAM_DEPTH         = (log2_SINGLE_WT_BRAM_DEPTH    )
    ID1_BRAM_DEPTH              = (1<<log2_ID1_BRAM_DEPTH       )
    ID1_BRAM_WIDTH              = (HBM_AXI_DATA_WIDTH*HBM_Port )

    log2_Tout = 5
    MAX_WT_DW = 4
    MAX_BN_DW = 16
    T_quant_block = 128
    log2_T_quant_block = 7
    WT_quant_scale_DW = 16
    DAT_BRAM_NUM = 1
    log2_Bank_Step = 28
    WT_BRAM_NUM = HBM_Port
    AXI_BURST_LEN = Tout
    log2_AXI_BURST_LEN = log2_Tout
    WT_CH_Tgroup = T_quant_block*HBM_AXI_DATA_WIDTH//WT_quant_scale_DW
    log2_WT_CH_Tgroup = 11
    DAT_BRAM_DEPTH = (1<<23)//base_Tin//MAX_DAT_DW//DAT_BRAM_NUM
    WT_BRAM_DEPTH = (1<<22)//HBM_AXI_DATA_WIDTH//WT_BRAM_NUM
    AXI_DAT_WIDTH = MAX_DAT_DW*Tout*Tb
    AXI_BN_WIDTH = MAX_BN_DW*Tout*Tb

    AXI_BURST_LEN_SOFTMAX = 4
    SINGLE_BN_FIFO_DEP = (AXI_BURST_LEN*MAX_DAT_DW*Tb)//(MAX_BN_DW*2)
    BN_FIFO_DEP = SINGLE_BN_FIFO_DEP * 4
    BN_FIFO_NUM = (MAX_BN_DW*2)//(MAX_DAT_DW*Tb)

    @classmethod
    def malloc_bytes(cls, shape, dtype, dynamic=False):
        if not dynamic:
            shape = [i.simplify(1).data if isinstance(i, ne.Expr) else i for i in shape]
        return 10000


