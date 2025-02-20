from ... import device
from ...adr import Tensor, Constant
from ..basic import TestbenchSIM, Tasks
from ...clib import WT_TRANS, BN_TRANS


def MVMTestbench(args, output, attrs):
    if attrs["skip"] == 1:
        dtensor, wtensor = args[0], args[1]
        dshape, wshape = dtensor[0].shape, wtensor[0].shape
        daddrs, waddrs, oaddrs = dtensor[1], wtensor[1], output[1]
        define = {
            "log2_WT_base_addr_Bank_Step": attrs["log2_step"], 
            "Hin": dshape[0], "Win": dshape[1], "Height": dshape[0]*dshape[1],
            "Width_in": wshape[0], "Width_out": wshape[1],
            "DAT_IN_BASE_ADDR":  daddrs & 0xffffffff,
            "HBM00_WT_BASE_ADDR":  waddrs & 0xffffffff,
            "DAT_OUT_BASE_ADDR": oaddrs & 0xffffffff,
        }
        return TestbenchSIM("testbench_HBM_MVM", define)


def MVMBNTestbench(args, output, attrs):
    if attrs["skip"] == 1:
        dtensor, wtensor, btensor = args[0], args[1], args[2]
        dshape, wshape, bshape = dtensor[0].shape, wtensor[0].shape, btensor[0].shape
        daddrs, waddrs, baddrs, oaddrs = dtensor[1], wtensor[1], btensor[1], output[1]
        if attrs["padding"]:
            define = {
                "log2_WT_base_addr_Bank_Step": attrs["log2_step"], "Token": dshape[1],
                "Head": dshape[0], 
                "DAT_OUT_LINE_STRIDE": dtensor[0].device.Pixel_Data_Bytes*dtensor[0].device.MAX_TOKEN,
                "DAT_OUT_SURFACE_STRIDE": dtensor[0].device.Pixel_Data_Bytes*dtensor[0].device.MAX_TOKEN*dshape[0],
                "Width_in": wshape[0], "Width_out": wshape[1],
                "DAT_IN_BASE_ADDR":  daddrs & 0xffffffff,
                "HBM00_WT_BASE_ADDR":  waddrs & 0xffffffff,
                "BN_BASE_ADDR":  baddrs & 0xffffffff,
                "DAT_OUT_BASE_ADDR": oaddrs & 0xffffffff,
            }
        else:
            define = {
                "log2_WT_base_addr_Bank_Step": attrs["log2_step"], "Token": dshape[1],
                "Head": dshape[0], 
                "Width_in": wshape[0], "Width_out": wshape[1],
                "DAT_IN_BASE_ADDR":  daddrs & 0xffffffff,
                "HBM00_WT_BASE_ADDR":  waddrs & 0xffffffff,
                "BN_BASE_ADDR":  baddrs & 0xffffffff,
                "DAT_OUT_BASE_ADDR": oaddrs & 0xffffffff,
            }
        return TestbenchSIM("testbench_HBM_MVM_bn", define)


def MVMBNResTestbench(args, output, attrs):
    if attrs["skip"] == 1 and attrs["arg_max"] == 0:
        dtensor, wtensor, btensor, rtensor = args[0], args[1], args[2], args[3]
        dshape, wshape, bshape, rshape = dtensor[0].shape, wtensor[0].shape, btensor[0].shape, rtensor[0].shape
        daddrs, waddrs, baddrs, raddrs, oaddrs = dtensor[1], wtensor[1], btensor[1], rtensor[1], output[1]
        define = {
            "log2_WT_base_addr_Bank_Step": attrs["log2_step"], 
            "Head": dshape[0], "Token": dshape[1],
            "Width_in": wshape[0], "Width_out": wshape[1], "EW_MODE": attrs["mul_mode"],
            "DAT_IN_BASE_ADDR":  daddrs & 0xffffffff,
            "HBM00_WT_BASE_ADDR":  waddrs & 0xffffffff,
            "BN_BASE_ADDR":  baddrs & 0xffffffff,
            "Res_Add_BASE_ADDR":  raddrs & 0xffffffff,
            "DAT_OUT_BASE_ADDR": oaddrs & 0xffffffff,
        }
        return TestbenchSIM("testbench_HBM_MVM_bn_res", define)
    elif attrs["skip"] == 1 and attrs["arg_max"] == 1:
        dtensor, wtensor, btensor, rtensor = args[0], args[1], args[2], args[3]
        dshape, wshape, bshape, rshape = dtensor[0].shape, wtensor[0].shape, btensor[0].shape, rtensor[0].shape
        daddrs, waddrs, baddrs, raddrs, oaddrs = dtensor[1], wtensor[1], btensor[1], rtensor[1], output[1]
        define = {
            "log2_WT_base_addr_Bank_Step": attrs["log2_step"], 
            "Head": dshape[0], "Token": dshape[1],
            "Width_in": wshape[0], "Width_out": wshape[1], "EW_MODE": attrs["mul_mode"],
            "DAT_IN_BASE_ADDR":  daddrs & 0xffffffff,
            "HBM00_WT_BASE_ADDR":  waddrs & 0xffffffff,
            "BN_BASE_ADDR":  baddrs & 0xffffffff,
            "Res_Add_BASE_ADDR":  raddrs & 0xffffffff,
            "DAT_OUT_BASE_ADDR": oaddrs[0] & 0xffffffff,
            "AUGMAX_OUT_ADDR": oaddrs[1] & 0xffffffff,
        }
        return TestbenchSIM("testbench_HBM_MVM_bn_res_Argmax", define)


@Tasks.Register("tb.hbm.mvm_afterTRP", device.HBM0507)
def MVMafterTRPTestbench_0(args, output, attrs):
    dtensor, wtensor = args[0], args[1]
    dshape, wshape = dtensor[0].shape, wtensor[0].shape
    daddrs, waddrs, oaddrs = dtensor[1], wtensor[1], output[1]
    if attrs["kvcache"]:
        define = {
            "KV_cache_mode": attrs["kvcache"], 
            "Feature_Head": dshape[0],
            "Weight_Head": wshape[0],
            "Token": 19,
            "DAT_IN_BASE_ADDR":  daddrs & 0xffffffff,
            "WT_BASE_ADDR":  waddrs & 0xffffffff,
            "DAT_OUT_BASE_ADDR": oaddrs & 0xffffffff,
        }
    elif attrs["padding"]:
        max_token = dtensor[0].device.MAX_TOKEN
        define = {
            "KV_cache_mode": attrs["kvcache"], 
            "Feature_Head": dshape[0],
            "Weight_Head": wshape[0],
            "Token": wshape[1],
            "Win": max_token, "Wout": max_token, "CHout": max_token,
            "DAT_IN_BASE_ADDR":  daddrs & 0xffffffff,
            "WT_BASE_ADDR":  waddrs & 0xffffffff,
            "DAT_OUT_BASE_ADDR": oaddrs & 0xffffffff,
        }
    else:
        define = {
            "KV_cache_mode": attrs["kvcache"], 
            "Feature_Head": dshape[0],
            "Weight_Head": wshape[0],
            "Token": wshape[1],
            "Win": dshape[1], "Wout": dshape[1], "CHout": dshape[1],
            "DAT_IN_BASE_ADDR":  daddrs & 0xffffffff,
            "WT_BASE_ADDR":  waddrs & 0xffffffff,
            "DAT_OUT_BASE_ADDR": oaddrs & 0xffffffff,
        }
    return TestbenchSIM("testbench_HBM_MVM_afterTRP", define)


@Tasks.Register("tb.hbm.mvm_afterTRP", device.EdgeLLMv1)
def MVMafterTRPTestbench_1(args, output, attrs):
    dtensor, wtensor = args[0], args[1]
    dshape, wshape = dtensor[0].shape, wtensor[0].shape
    daddrs, waddrs, oaddrs = dtensor[1], wtensor[1], output[1]
    last_token = attrs.get("last_token", 0)
    define = {
        "Feature_Head": dshape[0],
        "Weight_Head": wshape[0],
        "Token": dshape[1],
        "last_token": dshape[1]-1 if attrs.get("kvcache", 0) else last_token,
        "DAT_IN_BASE_ADDR":  daddrs & 0xffffffff,
        "WT_BASE_ADDR":  waddrs & 0xffffffff,
        "DAT_OUT_BASE_ADDR": oaddrs & 0xffffffff,
    }
    return TestbenchSIM("testbench_HBM_MVM_afterTRP", define)

    
def MVMafterTRPTestbench(args, output, attrs):
    return Tasks.Get("tb.hbm.mvm_afterTRP", args[0][0].device)(args, output, attrs)


@Tasks.Register("tb.hbm.mvm_afterF2W", device.HBM0507)
def MVMafterF2WTestbench_0(args, output, attrs):
    dtensor, wtensor = args[0], args[1]
    dshape, wshape = dtensor[0].shape, wtensor[0].shape
    daddrs, waddrs, oaddrs = dtensor[1], wtensor[1], output[1]
    if attrs["kvcache"]:
        define = {
            "KV_cache_mode": attrs["kvcache"], 
            "Feature_Head": dshape[0],
            "Weight_Head": wshape[0],
            "Token": 19,
            "Wout": dshape[1],
            "DAT_IN_BASE_ADDR":  daddrs & 0xffffffff,
            "WT_BASE_ADDR":  waddrs & 0xffffffff,
            "DAT_OUT_BASE_ADDR": oaddrs & 0xffffffff,
        }
    elif attrs["padding"]:
        max_token = dtensor[0].device.MAX_TOKEN
        define = {
            "KV_cache_mode": attrs["kvcache"], 
            "Feature_Head": dshape[0],
            "Weight_Head": wshape[0],
            "Token": wshape[1],
            "Win": max_token, "Wout": wshape[1], "CHout": max_token,
            "DAT_IN_BASE_ADDR":  daddrs & 0xffffffff,
            "WT_BASE_ADDR":  waddrs & 0xffffffff,
            "DAT_OUT_BASE_ADDR": oaddrs & 0xffffffff,
        }
    else:
        define = {
            "KV_cache_mode": attrs["kvcache"], 
            "Feature_Head": dshape[0],
            "Weight_Head": wshape[0],
            "Token": wshape[1],
            "Win": dshape[1], "Wout": wshape[1], "CHout": wshape[1],
            "DAT_IN_BASE_ADDR":  daddrs & 0xffffffff,
            "WT_BASE_ADDR":  waddrs & 0xffffffff,
            "DAT_OUT_BASE_ADDR": oaddrs & 0xffffffff,
        }
    return TestbenchSIM("testbench_HBM_MVM_afterF2W", define)


@Tasks.Register("tb.hbm.mvm_afterF2W", device.EdgeLLMv1)
def MVMafterF2WTestbench_1(args, output, attrs):
    dtensor, wtensor = args[0], args[1]
    dshape, wshape = dtensor[0].shape, wtensor[0].shape
    daddrs, waddrs, oaddrs = dtensor[1], wtensor[1], output[1]
    last_token = attrs.get("last_token", 0)
    define = {
        "Feature_Head": dshape[0],
        "Weight_Head": wshape[0],
        "Token": dshape[1],
        "last_token": dshape[1]-1 if attrs.get("kvcache", 0) else last_token,
        "DAT_IN_BASE_ADDR":  daddrs & 0xffffffff,
        "WT_BASE_ADDR":  waddrs & 0xffffffff,
        "DAT_OUT_BASE_ADDR": oaddrs & 0xffffffff,
    }
    return TestbenchSIM("testbench_HBM_MVM_afterF2W", define)

    
def MVMafterF2WTestbench(args, output, attrs):
    return Tasks.Get("tb.hbm.mvm_afterF2W", args[0][0].device)(args, output, attrs)


def SoftmaxTestbench(args, output, attrs):
    dtensor = args[0]
    dshape = dtensor[0].shape
    daddrs, oaddrs = dtensor[1], output[1]
    if attrs["kvcache"]:
        define = {
            "KV_cache_mode": attrs["kvcache"],
            "Feature_Head": dshape[0],
            "Token": 19,
            "DAT_IN_BASE_ADDR":  daddrs & 0xffffffff,
            "DAT_OUT_BASE_ADDR": oaddrs & 0xffffffff,
        }
    elif attrs["padding"]:
        max_token = dtensor[0].device.MAX_TOKEN
        define = {
            "KV_cache_mode": attrs["kvcache"],
            "Feature_Head": dshape[0],
            "Token": 19,
            "Win": max_token, "Wout": max_token, "CHout": max_token,
            "DAT_IN_BASE_ADDR":  daddrs & 0xffffffff,
            "DAT_OUT_BASE_ADDR": oaddrs & 0xffffffff,
        }
    else:
        define = {
            "KV_cache_mode": attrs["kvcache"],
            "Feature_Head": dshape[0],
            "Token": dshape[1],
            "DAT_IN_BASE_ADDR":  daddrs & 0xffffffff,
            "DAT_OUT_BASE_ADDR": oaddrs & 0xffffffff,
        }
    return TestbenchSIM("testbench_SOFTMAX", define)


def LayerNormTestbench(args, output, attrs):
    dtensor, btensor = args[0], args[1]
    dshape, bshape = dtensor[0].shape, btensor[0].shape
    daddrs, baddrs, oaddrs = dtensor[1], btensor[1], output[1]
    define = {
        "Token": dshape[0]*dshape[1], "Width_in": dshape[2],
        "RMS_Norm": attrs["rms"],
        "DAT_IN_BASE_ADDR":  daddrs & 0xffffffff,
        "LN_WT_BASE_ADDR":  baddrs & 0xffffffff,
        "DAT_OUT_BASE_ADDR": oaddrs & 0xffffffff,
    }
    return TestbenchSIM("testbench_LN", define)


def PosEmbTestbench(args, output, attrs):
    dtensor, ptensor = args[0], args[1]
    dshape, oshape = dtensor[0].shape, output[0].shape
    daddrs, paddrs, oaddrs = dtensor[1], ptensor[1], output[1]
    if attrs["kvcache"]:
        define = {
            "Feature_Head": dshape[0], "Token": 19, "KV_cache_mode": attrs["kvcache"],
            "DAT_IN_BASE_ADDR":  daddrs & 0xffffffff,
            "POS_IN_BASE_ADDR":  paddrs & 0xffffffff,
            "DAT_OUT_BASE_ADDR": oaddrs & 0xffffffff,
        }
    elif attrs["padding"]:
        max_token = dtensor[0].device.MAX_TOKEN
        define = {
            "Feature_Head": dshape[0], "Token": dshape[1], "KV_cache_mode": attrs["kvcache"],
            "Win": max_token, "Wout": max_token,
            "DAT_IN_BASE_ADDR":  daddrs & 0xffffffff,
            "POS_IN_BASE_ADDR":  paddrs & 0xffffffff,
            "DAT_OUT_BASE_ADDR": oaddrs & 0xffffffff,
        }
    else:
        define = {
            "Feature_Head": dshape[0], "Token": dshape[1], "KV_cache_mode": attrs["kvcache"],
            "Win": dshape[1], "Wout": dshape[1],
            "DAT_IN_BASE_ADDR":  daddrs & 0xffffffff,
            "POS_IN_BASE_ADDR":  paddrs & 0xffffffff,
            "DAT_OUT_BASE_ADDR": oaddrs & 0xffffffff,
        }
    return TestbenchSIM("testbench_EMB", define)


def TransposeTestbench(args, output, attrs):
    dtensor = args[0]
    dshape = dtensor[0].shape
    daddrs, oaddrs = dtensor[1], output[1]
    define = {
        "log2_WT_base_addr_Bank_Step": attrs["log2_step"],
        "Token": dshape[-2], "Width_in": dshape[-1],
        "DAT_IN_BASE_ADDR":  daddrs & 0xffffffff,
        "HBM00_WT_BASE_ADDR": oaddrs & 0xffffffff,
    }
    return TestbenchSIM("testbench_TRANSPOSE_HBM", define)


def Feature2WeightTestbench(args, output, attrs):
    dtensor = args[0]
    dshape = dtensor[0].shape
    daddrs, oaddrs = dtensor[1], output[1]
    define = {
        "log2_WT_base_addr_Bank_Step": attrs["log2_step"], 
        "Token": dshape[-2], "Width_in": dshape[-1],
        "DAT_IN_BASE_ADDR":  daddrs & 0xffffffff,
        "HBM00_WT_BASE_ADDR":  oaddrs & 0xffffffff,
    }
    return TestbenchSIM("testbench_Feature2Weight_HBM", define)


def ActivateTestbench(args, output, attrs):
    dtensor, btensor = args[0], args[1]
    dshape, bshape = dtensor[0].shape, btensor[0].shape
    daddrs, baddrs, oaddrs = dtensor[1], btensor[1], output[1]
    define = {
        "Height": dshape[1], "Width_in": dshape[2],
        "DAT_IN_BASE_ADDR":  daddrs & 0xffffffff,
        "WT_BASE_ADDR":  baddrs & 0xffffffff,
        "DAT_OUT_BASE_ADDR": oaddrs & 0xffffffff,
    }
    return TestbenchSIM("testbench_ACT", define)