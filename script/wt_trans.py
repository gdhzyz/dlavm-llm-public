import numpy as np
from dlavm.adr import DataType, DataEnum
from dlavm.device import HBM0321
from dlavm.clib import WT_TRANS, BN_TRANS

def Test():
    with open("./test/MVM_BN_Wqkv.bin", "rb") as f:
        weight = b"".join(f.readlines())
        np_weight = np.frombuffer(weight, dtype="int32").reshape(4608, 4096)
        print(np_weight[0, 0:4])
    with open("./test/MVM_BN_Scaleqkv.bin", "rb") as f:
        scale = b"".join(f.readlines())
        np_scale = np.frombuffer(scale, dtype="float16").reshape(4608, 32)
    
    hbm_dtype = DataType(DataEnum.int4, DataEnum.hbm)
    require_bytes = HBM0321.malloc_bytes([4096, 4608], hbm_dtype)
    mapped_wt = WT_TRANS(np_weight, np_scale, require_bytes)
    print("success!")
    with open("./test/MVM_BN_write_to_HBM_bin/MVMBN0_HBM_DDR_31.bin", "rb") as f:
        target = b"".join(f.readlines())
        np_target = np.frombuffer(target, dtype="int32").reshape(require_bytes // 4)
    print(np.sum(np_target != mapped_wt[31]))

    with open("./test/MVM_BN_Biasqkv.bin", "rb") as f:
        bias = b"".join(f.readlines())
        np_bias = np.frombuffer(bias, dtype="float16").reshape(1, 4608)
    ddr_dtype = DataType(DataEnum.fp16, DataEnum.ddr)
    require_bytes = HBM0321.malloc_bytes([4608*2], ddr_dtype)
    mapped_bn = BN_TRANS(np.ones(shape=(1, 4608), dtype="float16"), np_bias, require_bytes)

    with open("./test/MVM_BN_DDR_bin/MVMBN0_wt_and_bias_in_DDR.bin", "rb") as f:
        target = b"".join(f.readlines())
        np_target = np.frombuffer(target, dtype="int32").reshape(1, require_bytes // 4)
    print(mapped_bn[0, 0])
    print(np_target[0, 0])
    print(np.sum(np_target != mapped_bn))


def out_layer_mvm():
    with open("./test/MVM_weight.bin", "rb") as f:
        weight = b"".join(f.readlines())
        np_weight = np.frombuffer(weight, dtype="int32").reshape(65024, 4096)
        print(np_weight[0, 0:4])
    with open("./test/MVM_scales.bin", "rb") as f:
        scale = b"".join(f.readlines())
        np_scale = np.frombuffer(scale, dtype="float16").reshape(65024, 32)
    
    hbm_dtype = DataType(DataEnum.int4, DataEnum.hbm)
    require_bytes = HBM0321.malloc_bytes([4096, 65024], hbm_dtype)
    mapped_wt = WT_TRANS(np_weight, np_scale, require_bytes)
    print("success!")
    print(mapped_wt.shape)
    for numb, mapped_port in enumerate(mapped_wt):
        with open(f"./test/out_layer/MVMBN_Argmax_HBM_DDR_{numb:02d}.bin", "wb") as f:
            f.write(mapped_port.tobytes())


def WT_QKV():
    with open("./test/MVM_BN_Wqkv.bin", "rb") as f:
        weight = b"".join(f.readlines())
        np_weight = np.frombuffer(weight, dtype="int32").reshape(4608, 4096)
    with open("./test/MVM_BN_Scaleqkv.bin", "rb") as f:
        scale = b"".join(f.readlines())
        np_scale = np.frombuffer(scale, dtype="float16").reshape(4608, 32)
    hbm_dtype = DataType(DataEnum.int4, DataEnum.hbm)
    np_weight = np_weight.reshape(36, 128, 4096)
    np_scale = np_scale.reshape(36, 128, 32)

    np_weight_qk = np_weight[:34, :, :].reshape(34*128, 4096)
    np_scale_qk = np_scale[:34, :, :].reshape(34*128, 32)
    require_bytes_qk = HBM0321.malloc_bytes([34*128, 4608], hbm_dtype)
    mapped_wt_qk = WT_TRANS(np_weight_qk, np_scale_qk, require_bytes_qk)
    print("qk success!")
    for numb, mapped_port in enumerate(mapped_wt_qk):
        with open(f"./test//MVMBN0_0_HBM_DDR_%02d.bin" % numb, "wb") as f:
            f.write(mapped_port.tobytes())

    np_weight_v = np_weight[34:, :, :].reshape(2*128, 4096)
    np_scale_v = np_scale[34:, :, :].reshape(2*128, 32)
    require_bytes_v = HBM0321.malloc_bytes([2*128, 4608], hbm_dtype)
    mapped_wt_v = WT_TRANS(np_weight_v, np_scale_v, require_bytes_v)
    print("v  success!")


# Test()
# out_layer_mvm()
WT_QKV()