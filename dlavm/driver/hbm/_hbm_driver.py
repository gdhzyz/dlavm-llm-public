from . import tasks_0914
from . import tasks_0714 as tasks
from ..basic import Tasks


def MVMDriver(args, output, attrs):
    if attrs["skip"] == 1:
        dtensor, wtensor = args[0], args[1]
        dshape, wshape = dtensor[0].shape, wtensor[0].shape
        daddrs, waddrs, oaddrs = dtensor[1], wtensor[1], output[1]
        onchip = {}
        if len(dtensor) >= 3:
            onchip["DAT_IN_ONCHIP"] = dtensor[2]
        if len(output) >= 3:
            onchip["DAT_OUT_ONCHIP"] = output[2]
        define = {
            "log2_WT_base_addr_Bank_Step": attrs["log2_step"], 
            "Hin": dshape[0], "Token": dshape[1], "Height": dshape[0]*dshape[1],
            "Width_in": wshape[0], "Width_out": wshape[1],
            "DAT_IN_BASE_ADDR":  daddrs,
            "HBM00_WT_BASE_ADDR":  waddrs,
            "DAT_OUT_BASE_ADDR": oaddrs,
            "device": args[0][0].device,
            **onchip,
            **attrs
        }
        return Tasks.Get("accel.hbm.mvm", args[0][0].device)(**define)


def MVMBNDriver(args, output, attrs):
    if attrs["skip"] == 1:
        dtensor, wtensor, btensor = args[0], args[1], args[2]
        dshape, wshape, bshape = dtensor[0].shape, wtensor[0].shape, btensor[0].shape
        daddrs, waddrs, baddrs, oaddrs = dtensor[1], wtensor[1], btensor[1], output[1]
        onchip = {}
        if len(dtensor) >= 3:
            onchip["DAT_IN_ONCHIP"] = dtensor[2]
        if len(output) >= 3 and attrs.get("arg_max", 0) == 0:
            onchip["DAT_OUT_ONCHIP"] = output[2]
        if attrs["padding"]:
            define = {
                "log2_WT_base_addr_Bank_Step": attrs["log2_step"], "Token": dshape[1],
                "Head": dshape[0], 
                "DAT_OUT_LINE_STRIDE": dtensor[0].device.Pixel_Data_Bytes*dtensor[0].device.MAX_TOKEN,
                "DAT_OUT_SURFACE_STRIDE": dtensor[0].device.Pixel_Data_Bytes*dtensor[0].device.MAX_TOKEN,
                "Width_in": wshape[0], "Width_out": wshape[1],
                "DAT_IN_BASE_ADDR":  daddrs,
                "HBM00_WT_BASE_ADDR":  waddrs,
                "BN_BASE_ADDR":  baddrs,
                "DAT_OUT_BASE_ADDR": oaddrs,
                "device": args[0][0].device,
                **onchip,
                **attrs,
            }
        elif attrs.get("arg_max", 0):
            oaddrs, aaddrs = output[0][1], output[1][1]
            define = {
                "log2_WT_base_addr_Bank_Step": attrs["log2_step"], "Token": dshape[1],
                "Head": dshape[0], 
                "Width_in": wshape[0], "Width_out": wshape[1],
                "DAT_IN_BASE_ADDR":  daddrs,
                "HBM00_WT_BASE_ADDR":  waddrs,
                "BN_BASE_ADDR":  baddrs,
                "DAT_OUT_BASE_ADDR": oaddrs,
                "AUGMAX_OUT_ADDR": aaddrs,
                "device": args[0][0].device,
                **attrs,
            }
        else:
            define = {
                "log2_WT_base_addr_Bank_Step": attrs["log2_step"], "Token": dshape[1],
                "Head": dshape[0], 
                "Width_in": wshape[0], "Width_out": wshape[1],
                "DAT_IN_BASE_ADDR":  daddrs,
                "HBM00_WT_BASE_ADDR":  waddrs,
                "BN_BASE_ADDR":  baddrs,
                "DAT_OUT_BASE_ADDR": oaddrs,
                "device": args[0][0].device,
                **onchip,
                **attrs,
            }
        return Tasks.Get("accel.hbm.mvm", args[0][0].device)(**define)


def MVMBNResDriver(args, output, attrs):
    if attrs["skip"] == 1 and attrs["arg_max"] == 0:
        dtensor, wtensor, btensor, rtensor = args[0], args[1], args[2], args[3]
        dshape, wshape, bshape, rshape = dtensor[0].shape, wtensor[0].shape, btensor[0].shape, rtensor[0].shape
        daddrs, waddrs, baddrs, raddrs, oaddrs = dtensor[1], wtensor[1], btensor[1], rtensor[1], output[1]
        onchip = {}
        if len(dtensor) >= 3:
            onchip["DAT_IN_ONCHIP"] = dtensor[2]
        if len(rtensor) >= 3:
            onchip["RES_IN_ONCHIP"] = rtensor[2]
        if len(output) >= 3:
            onchip["DAT_OUT_ONCHIP"] = output[2]
        define = {
            "log2_WT_base_addr_Bank_Step": attrs["log2_step"], 
            "Head": dshape[0], "Token": dshape[1],
            "Width_in": wshape[0], "Width_out": wshape[1], "EW_MODE": attrs["mul_mode"],
            "DAT_IN_BASE_ADDR":  daddrs,
            "HBM00_WT_BASE_ADDR":  waddrs,
            "BN_BASE_ADDR":  baddrs,
            "Res_Add_BASE_ADDR":  raddrs,
            "DAT_OUT_BASE_ADDR": oaddrs,
            "device": args[0][0].device,
            **onchip,
            **attrs
        }
    elif attrs["skip"] == 1 and attrs["arg_max"] == 1:
        dtensor, wtensor, btensor, rtensor = args[0], args[1], args[2], args[3]
        dshape, wshape, bshape, rshape = dtensor[0].shape, wtensor[0].shape, btensor[0].shape, rtensor[0].shape
        daddrs, waddrs, baddrs, raddrs, oaddrs, aaddrs = dtensor[1], wtensor[1], btensor[1], rtensor[1], output[0][1], output[1][1]
        define = {
            "log2_WT_base_addr_Bank_Step": attrs["log2_step"], 
            "Head": dshape[0], "Token": dshape[1],
            "Width_in": wshape[0], "Width_out": wshape[1], "EW_MODE": attrs["mul_mode"],
            "DAT_IN_BASE_ADDR":  daddrs,
            "HBM00_WT_BASE_ADDR":  waddrs,
            "BN_BASE_ADDR":  baddrs,
            "Res_Add_BASE_ADDR":  raddrs,
            "DAT_OUT_BASE_ADDR": oaddrs,
            "AUGMAX_OUT_ADDR": aaddrs,
            "device": args[0][0].device,
            **attrs
        }
    return Tasks.Get("accel.hbm.mvm", args[0][0].device)(**define)


def MVMafterTRPDriver(args, output, attrs):
    dtensor, wtensor = args[0], args[1]
    dshape, wshape = dtensor[0].shape, wtensor[0].shape
    daddrs, waddrs, oaddrs = dtensor[1], wtensor[1], output[1]
    onchip = {}
    if len(dtensor) >= 3:
        onchip["DAT_IN_ONCHIP"] = dtensor[2]
    if len(output) >= 3:
        onchip["DAT_OUT_ONCHIP"] = output[2]
    define = {
        "KV_cache_mode": attrs["kvcache"], 
        "Feature_Head": dshape[0],
        "Weight_Head": wshape[0],
        "Token": wshape[1],
        "Win": dshape[1], "Wout": dshape[1], "CHout": dshape[1],
        "DAT_IN_BASE_ADDR":  daddrs,
        "WT_BASE_ADDR":  waddrs,
        "DAT_OUT_BASE_ADDR": oaddrs,
        "device": args[0][0].device,
        **onchip,
        **attrs
    }
    return Tasks.Get("accel.hbm.mvm_afterTRP", args[0][0].device)(**define)


def MVMafterF2WDriver(args, output, attrs):
    dtensor, wtensor = args[0], args[1]
    dshape, wshape = dtensor[0].shape, wtensor[0].shape
    daddrs, waddrs, oaddrs = dtensor[1], wtensor[1], output[1]
    onchip = {}
    if len(dtensor) >= 3:
        onchip["DAT_IN_ONCHIP"] = dtensor[2]
    if len(output) >= 3:
        onchip["DAT_OUT_ONCHIP"] = output[2]
    define = {
        "KV_cache_mode": attrs["kvcache"], 
        "Feature_Head": dshape[0],
        "Weight_Head": wshape[0],
        "Token": dshape[1],
        "Win": dshape[1], "Wout": wshape[1], "CHout": wshape[1],
        "DAT_IN_BASE_ADDR":  daddrs,
        "WT_BASE_ADDR":  waddrs,
        "DAT_OUT_BASE_ADDR": oaddrs,
        "device": args[0][0].device,
        **onchip,
        **attrs
    }
    return Tasks.Get("accel.hbm.mvm_afterF2W", args[0][0].device)(**define)


def AddDriver(args, output, attrs):
    dtensor, btensor = args[0], args[1]
    dshape, bshape = dtensor[0].shape, btensor[0].shape
    daddrs, baddrs, oaddrs = dtensor[1], btensor[1], output[1]
    onchip = {}
    if len(dtensor) >= 3:
        onchip["A_DAT_IN_ONCHIP"] = dtensor[2]
        onchip["B_DAT_IN_ONCHIP"] = btensor[2]
    if len(output) >= 3:
        onchip["DAT_OUT_ONCHIP"] = output[2]
    define = {
        "Token": dshape[0]*dshape[1], "Width_in": dshape[2],
        "Mode": 0,
        "A_DAT_IN_BASE_ADDR":  daddrs,
        "B_DAT_IN_BASE_ADDR":  baddrs,
        "DAT_OUT_BASE_ADDR": oaddrs,
        "device": args[0][0].device,
        **onchip,
        **attrs
    }
    return Tasks.Get("accel.hbm.eleminatewise", args[0][0].device)(**define)


def SoftmaxDriver(args, output, attrs):
    dtensor = args[0]
    dshape = dtensor[0].shape
    daddrs, oaddrs = dtensor[1], output[1]
    onchip = {}
    if len(dtensor) >= 3:
        onchip["DAT_IN_ONCHIP"] = dtensor[2]
    if len(output) >= 3:
        onchip["DAT_OUT_ONCHIP"] = output[2]
    define = {
        "KV_cache_mode": attrs["kvcache"],
        "Feature_Head": dshape[0],
        "Token": dshape[1],
        "DAT_IN_BASE_ADDR":  daddrs,
        "DAT_OUT_BASE_ADDR": oaddrs,
        "device": args[0][0].device,
        **onchip,
        **attrs
    }
    return Tasks.Get("accel.hbm.softmax", args[0][0].device)(**define)


def LayerNormDriver(args, output, attrs):
    dtensor, btensor = args[0], args[1]
    dshape, bshape = dtensor[0].shape, btensor[0].shape
    daddrs, baddrs, oaddrs = dtensor[1], btensor[1], output[1]
    onchip = {}
    if len(dtensor) >= 3:
        onchip["DAT_IN_ONCHIP"] = dtensor[2]
    if len(output) >= 3:
        onchip["DAT_OUT_ONCHIP"] = output[2]
    define = {
        "Token": dshape[1], "Width_in": dshape[2],
        "RMS_Norm": attrs["rms"],
        "DAT_IN_BASE_ADDR":  daddrs,
        "LN_WT_BASE_ADDR":  baddrs,
        "DAT_OUT_BASE_ADDR": oaddrs,
        "device": args[0][0].device,
        **onchip,
        **attrs
    }
    return Tasks.Get("accel.hbm.layer_norm", args[0][0].device)(**define)


def PosEmbDriver(args, output, attrs):
    dtensor, ptensor = args[0], args[1]
    dshape, oshape = dtensor[0].shape, output[0].shape
    daddrs, paddrs, oaddrs = dtensor[1], ptensor[1], output[1]
    onchip = {}
    if len(dtensor) >= 3:
        onchip["DAT_IN_ONCHIP"] = dtensor[2]
    if len(output) >= 3:
        onchip["DAT_OUT_ONCHIP"] = output[2]
    define = {
        "Feature_Head": dshape[0], "Token": dshape[1], "KV_cache_mode": attrs["kvcache"],
        "Win": dshape[1], "Wout": dshape[1],
        "DAT_IN_BASE_ADDR":  daddrs,
        "POS_IN_BASE_ADDR":  paddrs,
        "DAT_OUT_BASE_ADDR": oaddrs,
        "device": args[0][0].device,
        **onchip,
        **attrs
    }
    return Tasks.Get("accel.hbm.pos_emb", args[0][0].device)(**define)


def ActivateDriver(args, output, attrs):
    dtensor, btensor = args[0], args[1]
    dshape, bshape = dtensor[0].shape, btensor[0].shape
    daddrs, baddrs, oaddrs = dtensor[1], btensor[1], output[1]
    onchip = {}
    if len(dtensor) >= 3:
        onchip["DAT_IN_ONCHIP"] = dtensor[2]
    if len(output) >= 3:
        onchip["DAT_OUT_ONCHIP"] = output[2]
    define = {
        "Height": dshape[1], "Width_in": dshape[2],
        "DAT_IN_BASE_ADDR":  daddrs,
        "WT_BASE_ADDR":  baddrs,
        "DAT_OUT_BASE_ADDR": oaddrs,
        "device": args[0][0].device,
        **onchip,
        **attrs
    }
    return Tasks.Get("accel.hbm.act", args[0][0].device)(**define)


def TRPMVMDriver(args, output, attrs):
    dtensor, wtensor = args[0], args[1]
    dshape, wshape = dtensor[0].shape, wtensor[0].shape
    daddrs, waddrs, oaddrs = dtensor[1], wtensor[1], output[1]
    onchip = {}
    if len(dtensor) >= 3:
        onchip["DAT_IN_ONCHIP"] = dtensor[2]
    if len(output) >= 3:
        onchip["DAT_OUT_ONCHIP"] = output[2]
    define = {
        "KV_cache_mode": attrs["kvcache"], 
        "Feature_Head": dshape[0],
        "Weight_Head": wshape[0],
        "Token": wshape[1],
        "Win": dshape[1], "Wout": dshape[1], "CHout": dshape[1],
        "DAT_IN_BASE_ADDR":  daddrs,
        "WT_BASE_ADDR":  waddrs,
        "DAT_OUT_BASE_ADDR": oaddrs,
        "device": args[0][0].device,
        **onchip,
        **attrs
    }
    return Tasks.Get("accel.hbm.trp_mvm", args[0][0].device)(**define)


def F2WMVMDriver(args, output, attrs):
    dtensor, wtensor = args[0], args[1]
    dshape, wshape = dtensor[0].shape, wtensor[0].shape
    daddrs, waddrs, oaddrs = dtensor[1], wtensor[1], output[1]
    onchip = {}
    if len(dtensor) >= 3:
        onchip["DAT_IN_ONCHIP"] = dtensor[2]
    if len(output) >= 3:
        onchip["DAT_OUT_ONCHIP"] = output[2]
    define = {
        "KV_cache_mode": attrs["kvcache"], 
        "Feature_Head": dshape[0],
        "Weight_Head": wshape[0],
        "Token": dshape[1],
        "Win": dshape[1], "Wout": wshape[1], "CHout": wshape[1],
        "DAT_IN_BASE_ADDR":  daddrs,
        "WT_BASE_ADDR":  waddrs,
        "DAT_OUT_BASE_ADDR": oaddrs,
        "device": args[0][0].device,
        **onchip,
        **attrs
    }
    return Tasks.Get("accel.hbm.f2w_mvm", args[0][0].device)(**define)


def DAT2HBMDriver(args, output, attrs):
    dtensor = args[0]
    dshape = dtensor[0].shape
    daddrs, oaddrs = dtensor[1], output[1]
    define = {
        "KV_cache_mode": attrs["kvcache"], 
        "Feature_Head": dshape[0],
        "Token": dshape[1],
        "Win": dshape[1],
        "DAT_IN_BASE_ADDR":  daddrs,
        "DAT_OUT_BASE_ADDR": oaddrs,
        "device": args[0][0].device,
        **attrs
    }
    return Tasks.Get("accel.hbm.dat2hbm", args[0][0].device)(**define)

