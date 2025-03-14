import math
from functools import reduce
from dlavm.adr.base import DataType, DataEnum
from .. import ne
from .base_accel import Accel
from ..utils import tools


class OHBM(Accel):
    name = "ohbm"
    version = 20250305
    description = """
        Only HBM EdgeLLM FPGA Accelerator, 存储设备从DDR-HBM变为Only-HBM
    """

    MAX_TOKEN                   = 2048

    Tb                          = 1
    Tout                        = 32
    HBM_Port                    = 32
    base_Tin                    = 128
    Tin                         = base_Tin
    s_Tin                       = 32
    MAX_DAT_DW                  = 16
    MAX_DW                      = 16
    HBM_AXI_DATA_WIDTH          = 256
    L_Tout                      = HBM_Port*HBM_AXI_DATA_WIDTH//MAX_DW
    ASYN_FACTOR                 = 2
    log2_CH                     = 19
    MAX_CH_per_HEAD             = 128
    Pixel_Data_Width            = HBM_AXI_DATA_WIDTH
    HBM_1Row_Bytes              = int((HBM_AXI_DATA_WIDTH)>>3)

    log2_P                      = 8
    log2_S                      = 8
    log2_K                      = 8

    MAX_BN_CH                   = 1024
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
        if dtype.dtype == DataEnum.fp16 and dtype.mapped == DataEnum.hbm:
            if len(shape) == 1: # BN: TODO, 2* cls.MAX_BN_DW maybe wrong
                bsize = (((shape[0] // 2) + cls.Tout - 1)// cls.Tout) * cls.Tout * 2 * 2 * cls.MAX_BN_DW // cls.HBM_Port
            elif len(shape) == 3: # Feature
                bsize = (((shape[0] * shape[-1]) + cls.L_Tout - 1)// cls.L_Tout) * cls.L_Tout * cls.MAX_DAT_DW // 8 * shape[1] // cls.HBM_Port
            elif len(shape) == 2: # Feature
                bsize = (((shape[-1]) + cls.L_Tout - 1)// cls.L_Tout) * cls.L_Tout * cls.MAX_DAT_DW // 8 * shape[0] // cls.HBM_Port
            else:
                raise RuntimeError(f"Unsupport ndim of shape {shape} in malloc bytes")
            return bsize
        elif dtype.dtype == DataEnum.int4 and dtype.mapped == DataEnum.hbm:
            Sparsity_Factor = 1
            CHout, CHin = shape
            CHout_div_Tout = tools.Ceil(CHout, cls.Tout)
            WT_CHin_div_Tin = tools.Ceil(CHin, cls.Tin)
            WT_CHin_Padding_with_Tin = WT_CHin_div_Tin*cls.Tin
            WT_CHout_Padding_with_Tout = CHout_div_Tout*cls.Tout

            WT_CH_Tgroup = (cls.T_quant_block*Sparsity_Factor*cls.HBM_AXI_DATA_WIDTH//cls.WT_quant_scale_DW)
            WT_scale_group_nums = ((WT_CHin_Padding_with_Tin+WT_CH_Tgroup-1)//WT_CH_Tgroup)
            WT_scale_bits = (WT_CHout_Padding_with_Tout*cls.HBM_AXI_DATA_WIDTH*WT_scale_group_nums)
            WT_SIZE_IN_BYTES = (((WT_CHout_Padding_with_Tout*WT_CHin_Padding_with_Tin*cls.MAX_WT_DW)>>3)+((WT_scale_bits)>>3))
            return WT_SIZE_IN_BYTES // cls.HBM_Port
        else:
            raise RuntimeError(f"Unsupport dtype of {dtype.dtype} and mapped of {dtype.mapped} in malloc bytes")


