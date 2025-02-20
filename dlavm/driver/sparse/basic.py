from ... import ne

DDR_WIDTH = 32
DDR_BYTES = DDR_WIDTH//8
MAX_AXI_DW = 512
log2_MAX_AXI_DW = 9

log2_CH = 16
log2_H = 12
log2_W = log2_H
log2_KyKx = 8
log2_P = 4
log2_S = 4
log2_K = 4
log2_scale = 6

base_Tin = 8
base_log2Tin = 3
log2_other = log2_CH-base_log2Tin
Tout = 8
log2Tout = 3				

base_Tin_div_Tout = base_Tin//Tout
MAX_DW = 8
MAX_log2DW = 3
MAX_DW_Ratio = MAX_DW

MAX_DAT_DW = MAX_DW
MAX_log2DAT_DW = MAX_log2DW
AXI_DAT_WIDTH = MAX_DAT_DW * Tout
log2AXI_DAT_WIDTH = MAX_log2DAT_DW + log2Tout

MAX_WT_DW = MAX_DW
MAX_log2WT_DW = MAX_log2DW
AXI_WT_WIDTH = MAX_WT_DW *Tout
log2AXI_WT_WIDTH = MAX_log2WT_DW + log2Tout

MAX_BN_DW = MAX_DAT_DW
MAX_log2BN_DW = MAX_log2DAT_DW
AXI_BN_WIDTH = MAX_BN_DW *Tout
log2AXI_BN_WIDTH = MAX_log2BN_DW +log2Tout

Pixel_Data_Width = Tout*MAX_DAT_DW
Pixel_Data_Bytes = (Tout*MAX_DAT_DW)>>3
Pixel_BN_Data_Bytes = (Tout*MAX_BN_DW)>>3

BRAM_NUM = 16
log2BRAM_NUM = 4
BRAM_DEPTH = (1<<22)//base_Tin//MAX_DAT_DW//BRAM_NUM

def CSB_Write(regs, addr, data):
    if data is None:
        regs.append([1, addr, 0, 0])
    elif isinstance(data, ne.Expr):
        regs.append([1, addr, data.simplify().export("cpp"), len(data.get_vars())])
    else:
        regs.append([1, addr, data & 0xffffffff, 0])


def CSB_Read(regs, addr, data):
    if data is None:
        regs.append([0, addr, 0, 0])
    elif isinstance(data, ne.Expr):
        regs.append([0, addr, data.simplify().export("cpp"), len(data.get_vars())])
    else:
        regs.append([0, addr, data & 0xffffffff, 0])

