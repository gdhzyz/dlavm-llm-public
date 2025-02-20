from functools import reduce
from ..adr.base import DataType, DataEnum
from .. import ne
from .base_accel import Accel


class HBM(Accel):
    name = "hbm"
    version = 20240101

    log2_CH = 16
    Tb = 1
    HBM_Port = 32
    base_Tin = 128
    Tout = 32
    log2_Tout = 5
    MAX_DAT_DW = 16
    MAX_WT_DW = 4
    MAX_BN_DW = 16
    T_quant_block = 128
    log2_T_quant_block = 7
    HBM_AXI_DATA_WIDTH = 256
    WT_quant_scale_DW = 16
    DAT_BRAM_NUM = 1
    HBM_Port = 32
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
    Pixel_Data_Width = AXI_DAT_WIDTH
    Pixel_Data_Bytes = int((AXI_DAT_WIDTH)>>3)
    AXI_BURST_LEN_SOFTMAX = 4
    SINGLE_BN_FIFO_DEP = (AXI_BURST_LEN*MAX_DAT_DW*Tb)//(MAX_BN_DW*2)
    BN_FIFO_DEP = SINGLE_BN_FIFO_DEP * 4
    BN_FIFO_NUM = (MAX_BN_DW*2)//(MAX_DAT_DW*Tb)

    @classmethod
    def malloc_bytes(cls, shape, dtype, dynamic=False):
        if not dynamic:
            shape = [i.simplify(1).data if isinstance(i, ne.Expr) else i for i in shape]
        # if len(shape) >= 3 and shape[0] > 1:
        #     shape[1] = max(shape[1], shape[2])
        #     shape[2] = shape[1]
        if dtype.mapped == DataEnum.ddr:
            new_shape = [i for i in shape]
            new_shape[-1] = (new_shape[-1] + cls.Tout - 1) // cls.Tout
            new_shape = [cls.Tout] + new_shape
            data_numb = reduce(lambda x, y: x*y, new_shape)
            return data_numb * dtype.get_bytes()
        elif dtype.mapped == DataEnum.hbm:
            in_ch, out_ch = shape 
            CHout_div_Tout = (out_ch + cls.Tout - 1) // cls.Tout
            WT_CHin_Padding_with_Tin = int((in_ch + cls.base_Tin - 1) // cls.base_Tin) * cls.base_Tin
            WT_scale_group_nums = (WT_CHin_Padding_with_Tin+cls.WT_CH_Tgroup-1) // cls.WT_CH_Tgroup
            require_bytes, wt_bit = 0, 4
            for _ in range(CHout_div_Tout):
                for _ in range(WT_scale_group_nums):
                    for _ in range(cls.Tout//cls.HBM_Port):
                        for _ in range(cls.HBM_AXI_DATA_WIDTH//32):
                            require_bytes += 1
            for _ in range(CHout_div_Tout):
                for j in range(WT_scale_group_nums):
                    for _ in range(cls.Tout // cls.HBM_Port):
                        wt_start_ch_in = j*cls.WT_CH_Tgroup
                        wt_end_ch_in = WT_CHin_Padding_with_Tin if (j==WT_scale_group_nums-1) else (j+1)*cls.WT_CH_Tgroup
                        for _ in range(wt_bit*wt_start_ch_in//32, wt_bit*wt_end_ch_in//32):
                            require_bytes += 1
            return require_bytes*4
        else:
            raise RuntimeError(f"HBM accelerator has no this storage: {dtype.mapped}")


class HBM0321(HBM):
    version = 20240321

    DAT_BRAM_DEPTH = (1<<22)//HBM.base_Tin//HBM.MAX_DAT_DW//HBM.DAT_BRAM_NUM
    WT_BRAM_DEPTH = (1<<23)//HBM.HBM_AXI_DATA_WIDTH//HBM.WT_BRAM_NUM
    AXI_BURST_LEN_SOFTMAX = 1
    BN_FIFO_DEP = HBM.SINGLE_BN_FIFO_DEP * 4
    MAX_TOKEN = 128
    MAX_CH_per_HEAD = 128
    MIN_WT_HEAD = 2


class ASYNHBM0402(HBM0321):
    name = "asyn_hbm"
    version = 20240402

    ASYN_FACTOR = 2
    WT_BRAM_DEPTH = HBM0321.WT_BRAM_DEPTH // ASYN_FACTOR


class HBM0424(HBM0321):
    version = 20240424

    MAX_TOKEN = 512
    MAX_CFG_NUM = 12


class HBM0507(HBM0424):
    version = 20240507

    AUX_WT_BUF_DEPTH = 1024


class HBM0603(HBM0507):
    version = 20240603

    MAX_TOKEN = 2048


class HBM0720(HBM0603):
    version = 20240720


class HBM0721(HBM0720):
    version = 20240721

    DAT_BRAM_DEPTH = (1<<23)//HBM0720.base_Tin//HBM0720.MAX_DAT_DW//HBM0720.DAT_BRAM_NUM


class EdgeLLMv1(HBM0721):
    version = 20240725

    AXI_BURST_LEN = HBM0721.Tout
    WT_AXI_BURST_LEN = 256
    AXI_BURST_LEN_SOFTMAX = AXI_BURST_LEN


class EdgeLLMv2(EdgeLLMv1):
    version = 20240824

    MAX_TOKEN = 2048
    log2_CH = 16


class EdgeLLMv3(EdgeLLMv2):
    version = 20240831


class HBM0912(EdgeLLMv3):
    version = 20240912

    log2_CH = 19
    ASYN_FACTOR = 1
    DAT_BRAM_DEPTH = (1<<23)//EdgeLLMv3.base_Tin//EdgeLLMv3.MAX_DAT_DW//EdgeLLMv3.DAT_BRAM_NUM
    WT_BRAM_DEPTH = (1<<24)//EdgeLLMv3.HBM_AXI_DATA_WIDTH//EdgeLLMv3.WT_BRAM_NUM

    s_Tin = EdgeLLMv3.Tout

    s_MAX_WT_DW = 16
    s_DAT_BRAM_NUM = EdgeLLMv3.Tout//s_Tin
    s_DAT_BRAM_DEPTH = ((1<<22)//s_Tin//EdgeLLMv3.MAX_DAT_DW//s_DAT_BRAM_NUM)
    s_DAT_BRAM_WIDTH = s_Tin*EdgeLLMv3.MAX_DAT_DW

    s_WT_BRAM_NUM = EdgeLLMv3.HBM_Port
    s_WT_BRAM_WIDTH = EdgeLLMv3.HBM_AXI_DATA_WIDTH
    s_TRUE_WT_BRAM_DEPTH = ((1<<22)//s_WT_BRAM_WIDTH//s_WT_BRAM_NUM)
    s_WT_BRAM_DEPTH = (s_TRUE_WT_BRAM_DEPTH//ASYN_FACTOR)

    description = """
        更新dat2hbm，包含了trp和f2w系列的更新，添加了新算子以对之前的算子做区分。
        WARNING！！！不可使用adr.hbm.mvm_afterXXX系列算子，需调用adr.hbm.xxx_mvm算子，
          或者直接调用adr.hbm.attention算子
        此版本开始提供driver.ir支持，通过编写通用driver驱动代码以对多个后端进行代码生成

    """

class HBM0923(HBM0912):
    version = 20240923

    MAX_CFG_NUM = 2
    DAT_BRAM_DEPTH = (1<<24)//HBM0912.base_Tin//HBM0912.MAX_DAT_DW//HBM0912.DAT_BRAM_NUM
    WT_BRAM_DEPTH = (1<<23)//HBM0912.HBM_AXI_DATA_WIDTH//HBM0912.WT_BRAM_NUM

    description = """
        此版本基本上是VCU128板上GLM、Lamma的最终版本，dat BRAM参数24，wt BRAM参数为23。
        MVM，TRP，F2W算子都做了分割，可以支持很大的token。MAXTOKEN仅受限于HBM的容量，设置为2048
    """

HBM0921BackUp = HBM0923
