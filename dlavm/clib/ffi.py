import os
import sys
import ctypes
import numpy as np


class HBMTensor(ctypes.Structure):

    _fields_ = [
        ("CH00", ctypes.POINTER(ctypes.c_int32)),
        ("CH01", ctypes.POINTER(ctypes.c_int32)),
        ("CH02", ctypes.POINTER(ctypes.c_int32)),
        ("CH03", ctypes.POINTER(ctypes.c_int32)),
        ("CH04", ctypes.POINTER(ctypes.c_int32)),
        ("CH05", ctypes.POINTER(ctypes.c_int32)),
        ("CH06", ctypes.POINTER(ctypes.c_int32)),
        ("CH07", ctypes.POINTER(ctypes.c_int32)),
        ("CH08", ctypes.POINTER(ctypes.c_int32)),
        ("CH09", ctypes.POINTER(ctypes.c_int32)),
        ("CH10", ctypes.POINTER(ctypes.c_int32)),
        ("CH11", ctypes.POINTER(ctypes.c_int32)),
        ("CH12", ctypes.POINTER(ctypes.c_int32)),
        ("CH13", ctypes.POINTER(ctypes.c_int32)),
        ("CH14", ctypes.POINTER(ctypes.c_int32)),
        ("CH15", ctypes.POINTER(ctypes.c_int32)),
        ("CH16", ctypes.POINTER(ctypes.c_int32)),
        ("CH17", ctypes.POINTER(ctypes.c_int32)),
        ("CH18", ctypes.POINTER(ctypes.c_int32)),
        ("CH19", ctypes.POINTER(ctypes.c_int32)),
        ("CH20", ctypes.POINTER(ctypes.c_int32)),
        ("CH21", ctypes.POINTER(ctypes.c_int32)),
        ("CH22", ctypes.POINTER(ctypes.c_int32)),
        ("CH23", ctypes.POINTER(ctypes.c_int32)),
        ("CH24", ctypes.POINTER(ctypes.c_int32)),
        ("CH25", ctypes.POINTER(ctypes.c_int32)),
        ("CH26", ctypes.POINTER(ctypes.c_int32)),
        ("CH27", ctypes.POINTER(ctypes.c_int32)),
        ("CH28", ctypes.POINTER(ctypes.c_int32)),
        ("CH29", ctypes.POINTER(ctypes.c_int32)),
        ("CH30", ctypes.POINTER(ctypes.c_int32)),
        ("CH31", ctypes.POINTER(ctypes.c_int32)),
    ]

current_dir = os.path.dirname(os.path.abspath(__file__))
if sys.platform.startswith("win"):
    lib = ctypes.CDLL(os.path.join(current_dir, "Project1.dll"), ctypes.RTLD_GLOBAL)
elif sys.platform.startswith("linux"):
    lib = ctypes.CDLL(os.path.join(current_dir, "mod.so"), ctypes.RTLD_GLOBAL)
else:
    raise RuntimeError("could not check OS")
_wt_trans, _bn_trans = lib.WT_TRANS, lib.BN_TRANS
_wt_trans_int4 = lib.WT_TRANS_INT4
_wt_trans.restype = ctypes.POINTER(HBMTensor)
_wt_trans_int4.restype = ctypes.POINTER(HBMTensor)
_bn_trans.restype = ctypes.POINTER(ctypes.c_int32)

def FP32_to_FP20(fp32):
	return lib.FP32_to_FP20(ctypes.c_float(fp32))


def Test():
    clib_test = lib.test
    clib_test.restype = ctypes.POINTER(HBMTensor)
    np_data = np.random.randint(-10, 10, size=(32,)).astype(np.int32)
    data = bytearray(np_data.tobytes())
    rptr = (ctypes.c_byte * len(data)).from_buffer(data)
    print(type(rptr))
    arr = ctypes.cast(rptr, ctypes.POINTER(ctypes.c_int))
    tensor = clib_test(arr)[0]
    print(np_data)
    print(tensor.CH00[0])
    print(tensor.CH01[0])
    print(tensor.CH02[0])
    print(tensor.CH03[0])
    print(tensor.CH04[0])


def MVMWeight():

    np_data = np.random.randint(-10, 10, size=(32,)).astype(np.int32)
    print(type(np_data))
    with open("../../../cpp/svcir/test/chatglm_page1/MVM_BN_Wq.bin", "rb") as f:
        weight = b"".join(f.readlines())
        np_weight = np.frombuffer(weight, dtype="int32").reshape(4096, 4096)
    with open("../../../cpp/svcir/test/chatglm_page1/MVM_BN_Scaleq.bin", "rb") as f:
        scale = b"".join(f.readlines())
        np_scale = np.frombuffer(scale, dtype="float16").reshape(4096, 32)
    baweight = bytearray(weight)
    wrptr = (ctypes.c_byte * len(baweight)).from_buffer(baweight)
    weight = ctypes.cast(wrptr, ctypes.POINTER(ctypes.c_int))
    bascale = bytearray(scale)
    srptr = (ctypes.c_byte * len(bascale)).from_buffer(bascale)
    scale = ctypes.cast(srptr, ctypes.POINTER(ctypes.c_uint16))

    in_ch, out_ch = 4096, 4096
    CHout_div_Tout = (out_ch + Tout - 1) // Tout
    WT_CHin_Padding_with_Tin = int((in_ch + base_Tin - 1) // base_Tin) * base_Tin
    WT_scale_group_nums = (WT_CHin_Padding_with_Tin+WT_CH_Tgroup-1) // WT_CH_Tgroup

    require_bytes, wt_bit = 0, 4
    for _ in range(CHout_div_Tout):
        for _ in range(WT_scale_group_nums):
            for _ in range(Tout//HBM_Port):
                for _ in range(HBM_AXI_DATA_WIDTH//32):
                    require_bytes += 1

    for _ in range(CHout_div_Tout):
        for j in range(WT_scale_group_nums):
            for _ in range(Tout // HBM_Port):
                wt_start_ch_in = j*WT_CH_Tgroup
                wt_end_ch_in = WT_CHin_Padding_with_Tin if (j==WT_scale_group_nums-1) else (j+1)*WT_CH_Tgroup
                for _ in range(wt_bit*wt_start_ch_in//32, wt_bit*wt_end_ch_in//32):
                    require_bytes += 1

    tensor = _wt_trans(4096, 4096, require_bytes*4, weight, scale)[0]
    print(tensor.CH00[0])
    result = np.zeros((32, require_bytes), dtype="int32")
    for i in range(32):
        result[i] = eval("tensor.CH%02d[0:require_bytes]"%i)
    print(result)


def BNWeight():
    with open("../../../cpp/svcir/test/chatglm_page1/MVM_BN_Biasq.bin", "rb") as f:
        bias = b"".join(f.readlines())
        weight = np.zeros((4096,), dtype=np.float16).tobytes()
    baweight = bytearray(weight)
    wrptr = (ctypes.c_byte * len(baweight)).from_buffer(baweight)
    weight = ctypes.cast(wrptr, ctypes.POINTER(ctypes.c_uint16))
    babias = bytearray(bias)
    brptr = (ctypes.c_byte * len(babias)).from_buffer(babias)
    bias = ctypes.cast(brptr, ctypes.POINTER(ctypes.c_uint16))

    require_bytes = 4096*2

    tensor = _bn_trans(4096, require_bytes*2, weight, bias)

    print(tensor[0])
    result = np.zeros((1, require_bytes//2), dtype="int32")
    result[0] = tensor[0:require_bytes//2]
    print(result[0, 0])


def WT_TRANS(weight: np.ndarray, scale: np.ndarray, require_bytes) -> np.ndarray:
    shape = weight.shape
    weight = weight.tobytes()
    scale = scale.tobytes()
    baweight = bytearray(weight)
    wrptr = (ctypes.c_byte * len(baweight)).from_buffer(baweight)
    weight = ctypes.cast(wrptr, ctypes.POINTER(ctypes.c_int))
    bascale = bytearray(scale)
    srptr = (ctypes.c_byte * len(bascale)).from_buffer(bascale)
    scale = ctypes.cast(srptr, ctypes.POINTER(ctypes.c_uint16))
    tensor = _wt_trans(shape[1], shape[0], require_bytes, weight, scale)[0]
    require_numb = require_bytes // 4
    result = np.zeros((32, require_numb), dtype="int32")
    for i in range(32):
        result[i] = eval("tensor.CH%02d[0:require_numb]"% i)
    return result


def WT_TRANS_INT4(weight: np.ndarray, scale: np.ndarray, require_bytes) -> np.ndarray:
    shape = weight.shape
    weight = weight.tobytes()
    scale = scale.tobytes()
    baweight = bytearray(weight)
    wrptr = (ctypes.c_byte * len(baweight)).from_buffer(baweight)
    weight = ctypes.cast(wrptr, ctypes.POINTER(ctypes.c_int))
    bascale = bytearray(scale)
    srptr = (ctypes.c_byte * len(bascale)).from_buffer(bascale)
    scale = ctypes.cast(srptr, ctypes.POINTER(ctypes.c_uint16))
    tensor = _wt_trans_int4(shape[1]*8, shape[0], require_bytes, weight, scale)[0]
    require_numb = require_bytes // 4
    result = np.zeros((32, require_numb), dtype="int32")
    for i in range(32):
        result[i] = eval("tensor.CH%02d[0:require_numb]"% i)
    return result


def BN_TRANS(weight: np.ndarray, bias: np.ndarray, require_bytes) -> np.ndarray:
    shape = weight.shape
    weight = weight.tobytes()
    bias = bias.tobytes()
    baweight = bytearray(weight)
    wrptr = (ctypes.c_byte * len(baweight)).from_buffer(baweight)
    weight = ctypes.cast(wrptr, ctypes.POINTER(ctypes.c_uint16))
    babias = bytearray(bias)
    brptr = (ctypes.c_byte * len(babias)).from_buffer(babias)
    bias = ctypes.cast(brptr, ctypes.POINTER(ctypes.c_uint16))
    tensor = _bn_trans(shape[-1], require_bytes, weight, bias)
    require_numb = require_bytes // 4
    result = np.zeros((1, require_numb), dtype="int32")
    result[0] = tensor[0: require_numb]
    return result


if __name__ == "__main__":
    # Test()
    Tb = 1
    HBM_Port = 32
    base_Tin = 128
    Tout = 32
    log2_Tout = 5
    MAX_DAT_DW = 16
    MAX_WT_DW = 4
    MAX_BN_DW = 16
    T_quant_block = 128
    HBM_AXI_DATA_WIDTH = 256
    WT_quant_scale_DW = 16
    DAT_BRAM_NUM = 1
    HBM_Port = 32
    WT_BRAM_NUM = HBM_Port
    AXI_BURST_LEN = Tout
    log2_AXI_BURST_LEN = log2_Tout
    WT_CH_Tgroup = T_quant_block*HBM_AXI_DATA_WIDTH//WT_quant_scale_DW
    DAT_BRAM_DEPTH = (1<<23)//base_Tin//MAX_DAT_DW//DAT_BRAM_NUM
    WT_BRAM_DEPTH = (1<<24)//HBM_AXI_DATA_WIDTH//WT_BRAM_NUM
    AXI_DAT_WIDTH = MAX_DAT_DW*Tout*Tb
    AXI_BN_WIDTH = MAX_BN_DW*Tout*Tb
    Pixel_Data_Width = AXI_DAT_WIDTH
    Pixel_Data_Bytes = int((AXI_DAT_WIDTH)>>3)
    AXI_BURST_LEN_SOFTMAX = 4
    BN_FIFO_DEP = (AXI_BURST_LEN*MAX_DAT_DW*Tb)//(MAX_BN_DW*2)
    BN_FIFO_NUM = (MAX_BN_DW*2)//(MAX_DAT_DW*Tb)
    MVMWeight()
    BNWeight()
