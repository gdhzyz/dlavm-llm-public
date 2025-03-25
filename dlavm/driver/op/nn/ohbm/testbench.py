from dlavm import ne
from dlavm.adr import Op, Attrs
from dlavm.op.nn.attrs import RoPEMode
from dlavm.device import ohbm_accel
from .... import ir
from ....ir import CSB_Write, CSB_Read, While
from ....basic import TestbenchSIM, Tasks

def get_vars(targets):
    vars = []
    func = lambda n : [i for i in n if i not in vars]
    for n in targets:
        if isinstance(n, list):
            vars += func(get_vars(n))
        elif isinstance(n, Attrs):
            vars += func(get_vars(n.values()))
        elif isinstance(n, ne.Expr):
            vars += func(n.get_vars(True))
    return vars

#########################################################################################
#                            nn.mvm_f16xi4 testbench task                               #
#########################################################################################
@Tasks.Register("tb.nn.mvm_f16xi4.ohbm", ohbm_accel.OHBM)
def MVMF16xI4(args, output, attrs):
    if len(args) == 2:
        dtensor, wtensor = args[0], args[1]
        dshape, wshape = dtensor.shape, wtensor.shape
        daddr = dtensor.static_address
        waddr = wtensor.static_address
        oaddr = output[0].static_address
        macro_define = {
            "last_token" : attrs.get("last_token", 0),
            "Token" : dshape[-2] + attrs.get("last_token", 0),
            "RELU_EN" : 1 if attrs.get("relu") else 0,
            "Width_in" : dshape[-1],
            "Width_out" : wshape[0],
            # "DAT_IN_BASE_ADDR" : daddr,
            # "HBM_WT_BASE_ADDR" : waddr,
            # "DAT_OUT_BASE_ADDR" : oaddr,
        }
        return TestbenchSIM("testbench_HBM_MVM", macro_define)
    elif len(args) == 3:
        dtensor, wtensor = args[0], args[1]
        dshape, wshape = dtensor.shape, wtensor.shape
        daddr = dtensor.static_address
        waddr = wtensor.static_address
        oaddr = output[0].static_address
        macro_define = {
            "last_token" : attrs.get("last_token", 0),
            "Token" : dshape[-2] + attrs.get("last_token", 0),
            "RELU_EN" : 1 if attrs.get("relu") else 0,
            "Width_in" : dshape[-1],
            "Width_out" : wshape[0],
            # "DAT_IN_BASE_ADDR" : daddr,
            # "HBM_WT_BASE_ADDR" : waddr,
            # "DAT_OUT_BASE_ADDR" : oaddr,
        }
        if attrs.get("argmax"):
            return TestbenchSIM("testbench_HBM_MVM_BN_Argmax", macro_define)
        else:
            if attrs.get("out_heads") is not None:
                del macro_define["Width_out"]
                macro_define["Feature_Head"] = attrs.get("out_heads")[0]
                macro_define["Weight_Head"] = attrs.get("out_heads")[1]
                return TestbenchSIM("testbench_HBM_MVM_BN_output_head_mode", macro_define)
            if hasattr(dtensor, "heads"):
                del macro_define["Width_in"]
                macro_define["Feature_Head"] = dtensor.heads[0]
                macro_define["Weight_Head"] = dtensor.heads[1]
                return TestbenchSIM("testbench_HBM_MVM_BN_input_head_mode", macro_define)
            return TestbenchSIM("testbench_HBM_MVM_BN", macro_define)
    else:
        raise RuntimeError("not support mvm with bn or res in tb")

@Tasks.Register("tb.nn.mvm_f16xi4.ohbm", ohbm_accel.OHBM0314)
def MVMF16xI4(args, output, attrs):
    if len(args) == 2:
        dtensor, wtensor = args[0], args[1]
        dshape, wshape = dtensor.shape, wtensor.shape
        daddr = dtensor.static_address
        waddr = wtensor.static_address
        oaddr = output[0].static_address
        macro_define = {
            "last_token" : attrs.get("last_token", 0),
            "Token" : dshape[-2] + attrs.get("last_token", 0),
            "RELU_EN" : 1 if attrs.get("relu") else 0,
            "Width_in" : dshape[-1],
            "Width_out" : wshape[0],
            "BN_RELU_EN" : 0,
            # "DAT_IN_BASE_ADDR" : daddr,
            # "HBM_WT_BASE_ADDR" : waddr,
            # "DAT_OUT_BASE_ADDR" : oaddr,
        }
        return TestbenchSIM("testbench_HBM_MVM", macro_define)
    elif len(args) == 3:
        dtensor, wtensor = args[0], args[1]
        dshape, wshape = dtensor.shape, wtensor.shape
        daddr = dtensor.static_address
        waddr = wtensor.static_address
        oaddr = output[0].static_address
        macro_define = {
            "last_token" : attrs.get("last_token", 0),
            "Token" : dshape[-2] + attrs.get("last_token", 0),
            "RELU_EN" : 1 if attrs.get("relu") else 0,
            "Width_in" : dshape[-1],
            "Width_out" : wshape[0],
            "BN_RELU_EN" : 0,
            # "DAT_IN_BASE_ADDR" : daddr,
            # "HBM_WT_BASE_ADDR" : waddr,
            # "DAT_OUT_BASE_ADDR" : oaddr,
        }
        if attrs.get("argmax"):
            return TestbenchSIM("testbench_HBM_MVM_BN_Argmax", macro_define)
        else:
            if attrs.get("out_heads") is not None:
                del macro_define["Width_out"]
                macro_define["Original_Feature_Head"] = attrs.get("out_heads")[0]
                macro_define["Weight_Head"] = attrs.get("out_heads")[1]
                return TestbenchSIM("testbench_HBM_MVM_BN_output_head_mode", macro_define)
            if hasattr(dtensor, "heads"):
                del macro_define["Width_in"]
                macro_define["Original_Feature_Head"] = dtensor.heads[0]
                macro_define["Weight_Head"] = dtensor.heads[1]
                return TestbenchSIM("testbench_HBM_MVM_BN_input_head_mode", macro_define)
            return TestbenchSIM("testbench_HBM_MVM_BN", macro_define)
    else:
        raise RuntimeError("not support mvm with bn or res in tb")


@Tasks.Register("tb.nn.mvm_f16xi4.ohbm", ohbm_accel.OHBM0323)
def MVMF16xI4(args, output, attrs):
    if len(args) == 2:
        dtensor, wtensor = args[0], args[1]
        dshape, wshape = dtensor.shape, wtensor.shape
        daddr = dtensor.static_address
        waddr = wtensor.static_address
        oaddr = output[0].static_address
        macro_define = {
            "Last_Token" : attrs.get("last_token", 0),
            "This_Token" : dshape[-2] + attrs.get("last_token", 0),
            "RELU_EN" : 1 if attrs.get("relu") else 0,
            "Token_CHin" : dshape[-1],
            "Token_CHout" : wshape[0],
            "BN_RELU_EN" : 0,
            # "DAT_IN_BASE_ADDR" : daddr,
            # "HBM_WT_BASE_ADDR" : waddr,
            # "DAT_OUT_BASE_ADDR" : oaddr,
        }
        return TestbenchSIM("testbench_HBM_MVM", macro_define)
    elif len(args) == 3:
        dtensor, wtensor = args[0], args[1]
        dshape, wshape = dtensor.shape, wtensor.shape
        daddr = dtensor.static_address
        waddr = wtensor.static_address
        oaddr = output[0].static_address
        macro_define = {
            "Last_Token" : attrs.get("last_token", 0),
            "This_Token" : dshape[-2] + attrs.get("last_token", 0),
            "RELU_EN" : 1 if attrs.get("relu") else 0,
            "Token_CHin" : dshape[-1],
            "Token_CHout" : wshape[0],
            "BN_RELU_EN" : 0,
            # "DAT_IN_BASE_ADDR" : daddr,
            # "HBM_WT_BASE_ADDR" : waddr,
            # "DAT_OUT_BASE_ADDR" : oaddr,
        }
        if attrs.get("argmax"):
            return TestbenchSIM("testbench_HBM_MVM_BN_Argmax", macro_define)
        else:
            if attrs.get("out_heads") is not None:
                del macro_define["Token_CHout"]
                macro_define["Original_Feature_Head"] = attrs.get("out_heads")[0]
                macro_define["Weight_Head"] = attrs.get("out_heads")[1]
                return TestbenchSIM("testbench_HBM_MVM_BN_output_head_mode", macro_define)
            if hasattr(dtensor, "heads"):
                del macro_define["Token_CHin"]
                macro_define["Original_Feature_Head"] = dtensor.heads[0]
                macro_define["Weight_Head"] = dtensor.heads[1]
                return TestbenchSIM("testbench_HBM_MVM_BN_input_head_mode", macro_define)
            return TestbenchSIM("testbench_HBM_MVM_BN", macro_define)
    else:
        raise RuntimeError("not support mvm with bn or res in tb")


@Op.RegisterAttrs("nn.mvm_f16xi4", "testbench", ohbm_accel.OHBM)
def tb_nn_mvm_f16xi4(args, output, attrs):
    if len(get_vars([args[0].shape, attrs])):
        raise RuntimeError("Unsupport dynamic symbol control in testbench simulation")
    device = args[0].device
    with ir.Function([]) as func:
        csbs = Tasks.Get("tb.nn.mvm_f16xi4.ohbm", device)(args, output, attrs)
        for csb in csbs:
            if csb[0]:
                func += ir.CSB_Write(csb[1], csb[2])
            else:
                func += While(CSB_Read(csb[1]) != 1)
    return func


#########################################################################################
#                                 nn.norm testbench task                                #
#########################################################################################
@Tasks.Register("tb.nn.norm.ohbm", ohbm_accel.OHBM)
def Norm(args, output, attrs):
    dtensor, wtensor = args[0], args[1]
    dshape, wshape = dtensor.shape, wtensor.shape
    daddr = dtensor.static_address
    waddr = wtensor.static_address
    oaddr = output[0].static_address
    macro_define = {
        "last_token" : attrs.get("last_token", 0),
        "Token" : dshape[-2] + attrs.get("last_token", 0),
        "Width_in" : dshape[-1],
        "RMS_Norm" : 1 if attrs.get("rms") else 0,
        # "DAT_IN_BASE_ADDR" : daddr,
        # "HBM_WT_BASE_ADDR" : waddr,
        # "DAT_OUT_BASE_ADDR" : oaddr,
    }
    return TestbenchSIM("testbench_LN", macro_define)

@Tasks.Register("tb.nn.norm.ohbm", ohbm_accel.OHBM0323)
def Norm(args, output, attrs):
    dtensor, wtensor = args[0], args[1]
    dshape, wshape = dtensor.shape, wtensor.shape
    daddr = dtensor.static_address
    waddr = wtensor.static_address
    oaddr = output[0].static_address
    macro_define = {
        "Last_Token" : attrs.get("last_token", 0),
        "This_Token" : dshape[-2] + attrs.get("last_token", 0),
        "Token_CHin" : dshape[-1],
        "RMS_Norm" : 1 if attrs.get("rms") else 0,
        # "DAT_IN_BASE_ADDR" : daddr,
        # "HBM_WT_BASE_ADDR" : waddr,
        # "DAT_OUT_BASE_ADDR" : oaddr,
    }
    return TestbenchSIM("testbench_LN", macro_define)

@Op.RegisterAttrs("nn.norm", "testbench", ohbm_accel.OHBM)
def tb_nn_norm(args, output, attrs):
    if len(get_vars([args[0].shape, attrs])):
        raise RuntimeError("Unsupport dynamic symbol control in testbench simulation")
    device = args[0].device
    with ir.Function([]) as func:
        csbs = Tasks.Get("tb.nn.norm.ohbm", device)(args, output, attrs)
        for csb in csbs:
            if csb[0]:
                func += ir.CSB_Write(csb[1], csb[2])
            else:
                func += While(CSB_Read(csb[1]) != 1)
    return func


#########################################################################################
#                               nn.softmax testbench task                               #
#########################################################################################
@Tasks.Register("tb.nn.softmax.ohbm", ohbm_accel.OHBM)
def Softmax(args, output, attrs):
    dtensor = args[0]
    dshape = dtensor.shape
    daddr = dtensor.static_address
    oaddr = output[0].static_address
    mask = 1 if attrs.get("mask") else 0
    if attrs.get("auto_mask"):
        mask = 1 if dshape[-2] > 1 else 0
    macro_define = {
        "last_token" : dshape[-1] - dshape[-2],
        "Token" : dshape[-1],
        "Width_in" : dshape[-1],
        "Feature_Head" : dshape[0],
        "Need_Mask" : mask,
        # "DAT_IN_BASE_ADDR" : daddr,
        # "HBM_WT_BASE_ADDR" : waddr,
        # "DAT_OUT_BASE_ADDR" : oaddr,
    }
    return TestbenchSIM("testbench_SOFTMAX", macro_define)

@Tasks.Register("tb.nn.softmax.ohbm", ohbm_accel.OHBM0314)
def Softmax(args, output, attrs):
    dtensor = args[0]
    dshape = dtensor.shape
    daddr = dtensor.static_address
    oaddr = output[0].static_address
    mask = 1 if attrs.get("mask") else 0
    if attrs.get("auto_mask"):
        mask = 1 if dshape[-2] > 1 else 0
    macro_define = {
        "last_token" : dshape[-1] - dshape[-2],
        "Token" : dshape[-1],
        "Width_in" : dshape[-1],
        "Original_Feature_Head" : dshape[0],
        "Need_Mask" : mask,
        # "DAT_IN_BASE_ADDR" : daddr,
        # "HBM_WT_BASE_ADDR" : waddr,
        # "DAT_OUT_BASE_ADDR" : oaddr,
    }
    return TestbenchSIM("testbench_SOFTMAX", macro_define)

@Tasks.Register("tb.nn.softmax.ohbm", ohbm_accel.OHBM0323)
def Softmax(args, output, attrs):
    dtensor = args[0]
    dshape = dtensor.shape
    daddr = dtensor.static_address
    oaddr = output[0].static_address
    mask = 1 if attrs.get("mask") else 0
    if attrs.get("auto_mask"):
        mask = 1 if dshape[-2] > 1 else 0
    macro_define = {
        "Last_Token" : dshape[-1] - dshape[-2],
        "This_Token" : dshape[-1],
        "Token_CHin" : dshape[-1],
        "Original_Feature_Head" : dshape[0],
        "Need_Mask" : mask,
        # "DAT_IN_BASE_ADDR" : daddr,
        # "HBM_WT_BASE_ADDR" : waddr,
        # "DAT_OUT_BASE_ADDR" : oaddr,
    }
    return TestbenchSIM("testbench_SOFTMAX", macro_define)

@Op.RegisterAttrs("nn.softmax", "testbench", ohbm_accel.OHBM)
def tb_nn_softmax(args, output, attrs):
    if len(get_vars([args[0].shape, attrs])):
        raise RuntimeError("Unsupport dynamic symbol control in testbench simulation")
    device = args[0].device
    with ir.Function([]) as func:
        csbs = Tasks.Get("tb.nn.softmax.ohbm", device)(args, output, attrs)
        for csb in csbs:
            if csb[0]:
                func += ir.CSB_Write(csb[1], csb[2])
            else:
                func += While(CSB_Read(csb[1]) != 1)
    return func


#########################################################################################
#                            nn.elementwise testbench task                              #
#########################################################################################
@Tasks.Register("tb.nn.elementwise.ohbm", ohbm_accel.OHBM)
def Elementwise(args, output, attrs):
    dtensor, wtensor = args[0], args[1]
    dshape, wshape = dtensor.shape, wtensor.shape
    daddr = dtensor.static_address
    waddr = wtensor.static_address
    oaddr = output[0].static_address
    macro_define = {
        "last_token" : attrs.get("last_token", 0),
        "Token" : dshape[-2] + attrs.get("last_token", 0),
        "Width_in" : dshape[-1],
        "Feature_Head" : dshape[0],
        "ElementWise_Mode" : attrs.get("mode"),
        # "DAT_IN_BASE_ADDR" : daddr,
        # "HBM_WT_BASE_ADDR" : waddr,
        # "DAT_OUT_BASE_ADDR" : oaddr,
    }
    return TestbenchSIM("testbench_ElementWise", macro_define)

@Tasks.Register("tb.nn.elementwise.ohbm", ohbm_accel.OHBM0323)
def Elementwise(args, output, attrs):
    dtensor, wtensor = args[0], args[1]
    dshape, wshape = dtensor.shape, wtensor.shape
    daddr = dtensor.static_address
    waddr = wtensor.static_address
    oaddr = output[0].static_address
    macro_define = {
        "Last_Token" : attrs.get("last_token", 0),
        "This_Token" : dshape[-2] + attrs.get("last_token", 0),
        "Token_CHin" : dshape[-1],
        "Feature_Head" : dshape[0],
        "ElementWise_Mode" : attrs.get("mode"),
        # "DAT_IN_BASE_ADDR" : daddr,
        # "HBM_WT_BASE_ADDR" : waddr,
        # "DAT_OUT_BASE_ADDR" : oaddr,
    }
    return TestbenchSIM("testbench_ElementWise", macro_define)

@Op.RegisterAttrs("nn.elementwise", "testbench", ohbm_accel.OHBM)
def tb_nn_elementwise(args, output, attrs):
    if len(get_vars([args[0].shape, attrs])):
        raise RuntimeError("Unsupport dynamic symbol control in testbench simulation")
    device = args[0].device
    with ir.Function([]) as func:
        csbs = Tasks.Get("tb.nn.elementwise.ohbm", device)(args, output, attrs)
        for csb in csbs:
            if csb[0]:
                func += ir.CSB_Write(csb[1], csb[2])
            else:
                func += While(CSB_Read(csb[1]) != 1)
    return func


#########################################################################################
#                              nn.activate testbench task                               #
#########################################################################################
@Tasks.Register("tb.nn.activate.ohbm", ohbm_accel.OHBM)
def Activate(args, output, attrs):
    dtensor, wtensor = args[0], args[1]
    dshape, wshape = dtensor.shape, wtensor.shape
    daddr = dtensor.static_address
    waddr = wtensor.static_address
    oaddr = output[0].static_address
    macro_define = {
        "last_token" : attrs.get("last_token", 0),
        "Token" : dshape[-2] + attrs.get("last_token", 0),
        "Width_in" : dshape[-1],
        # "DAT_IN_BASE_ADDR" : daddr,
        # "HBM_WT_BASE_ADDR" : waddr,
        # "DAT_OUT_BASE_ADDR" : oaddr,
    }
    return TestbenchSIM("testbench_ACT", macro_define)

@Tasks.Register("tb.nn.activate.ohbm", ohbm_accel.OHBM0323)
def Activate(args, output, attrs):
    dtensor, wtensor = args[0], args[1]
    dshape, wshape = dtensor.shape, wtensor.shape
    daddr = dtensor.static_address
    waddr = wtensor.static_address
    oaddr = output[0].static_address
    macro_define = {
        "Last_Token" : attrs.get("last_token", 0),
        "This_Token" : dshape[-2] + attrs.get("last_token", 0),
        "Token_CHin" : dshape[-1],
        # "DAT_IN_BASE_ADDR" : daddr,
        # "HBM_WT_BASE_ADDR" : waddr,
        # "DAT_OUT_BASE_ADDR" : oaddr,
    }
    return TestbenchSIM("testbench_ACT", macro_define)

@Op.RegisterAttrs("nn.activate", "testbench", ohbm_accel.OHBM)
def tb_nn_activate(args, output, attrs):
    if len(get_vars([args[0].shape, attrs])):
        raise RuntimeError("Unsupport dynamic symbol control in testbench simulation")
    device = args[0].device
    with ir.Function([]) as func:
        csbs = Tasks.Get("tb.nn.activate.ohbm", device)(args, output, attrs)
        for csb in csbs:
            if csb[0]:
                func += ir.CSB_Write(csb[1], csb[2])
            else:
                func += While(CSB_Read(csb[1]) != 1)
    return func


#########################################################################################
#                            nn.mvm_f16xf16 testbench task                              #
#########################################################################################
@Tasks.Register("tb.nn.mvm_f16xf16.ohbm", ohbm_accel.OHBM)
def MVMF16xF16(args, output, attrs):
    dtensor, wtensor = args[0], args[1]
    dshape, wshape = dtensor.shape, wtensor.shape
    daddr = dtensor.static_address
    waddr = wtensor.static_address
    oaddr = output[0].static_address
    if attrs.get("w_trp"):
        macro_define = {
            "last_token" : wshape[-2] - dshape[-2],
            "Token" : wshape[-2],
            "Width_in" : dshape[-1],
            "Width_out" : wshape[-2],
            "Feature_Head" : dshape[0],
            "Weight_Head" : wshape[0],
            "MAX_TOKEN" : dtensor.device.MAX_TOKEN,
            # "DAT_IN_BASE_ADDR" : daddr,
            # "HBM_WT_BASE_ADDR" : waddr,
            # "DAT_OUT_BASE_ADDR" : oaddr,
        }
        if hasattr(dtensor, "heads"):
            macro_define["Feature_Head"] = dtensor.heads[0]
        return TestbenchSIM("testbench_HBM_MVM_afterTRP_input_head_mode", macro_define)
    else:
        macro_define = {
            "last_token" : wshape[-2] - dshape[-2],
            "Token" : wshape[-2],
            "Width_in" : dshape[-1],
            "Width_out" : wshape[-1],
            "Feature_Head" : dshape[0],
            "Weight_Head" : wshape[0],
            "MAX_TOKEN" : dtensor.device.MAX_TOKEN,
            # "DAT_IN_BASE_ADDR" : daddr,
            # "HBM_WT_BASE_ADDR" : waddr,
            # "DAT_OUT_BASE_ADDR" : oaddr,
        }
        return TestbenchSIM("testbench_HBM_MVM_afterF2W_output_head_mode", macro_define)

@Tasks.Register("tb.nn.mvm_f16xf16.ohbm", ohbm_accel.OHBM0314)
def MVMF16xF16(args, output, attrs):
    dtensor, wtensor = args[0], args[1]
    dshape, wshape = dtensor.shape, wtensor.shape
    daddr = dtensor.static_address
    waddr = wtensor.static_address
    oaddr = output[0].static_address
    if attrs.get("w_trp"):
        macro_define = {
            "last_token" : wshape[-2] - dshape[-2],
            "Token" : wshape[-2],
            "Width_in" : dshape[-1],
            "Width_out" : wshape[-2],
            "Original_Feature_Head" : dshape[0],
            "Weight_Head" : wshape[0],
            "MAX_TOKEN" : dtensor.device.MAX_TOKEN,
            # "DAT_IN_BASE_ADDR" : daddr,
            # "HBM_WT_BASE_ADDR" : waddr,
            # "DAT_OUT_BASE_ADDR" : oaddr,
        }
        if hasattr(dtensor, "heads"):
            macro_define["Feature_Head"] = dtensor.heads[0]
        return TestbenchSIM("testbench_HBM_MVM_afterTRP_input_head_mode", macro_define)
    else:
        macro_define = {
            "last_token" : wshape[-2] - dshape[-2],
            "Token" : wshape[-2],
            "Width_in" : dshape[-1],
            "Width_out" : wshape[-1],
            "Original_Feature_Head" : dshape[0],
            "Weight_Head" : wshape[0],
            "MAX_TOKEN" : dtensor.device.MAX_TOKEN,
            # "DAT_IN_BASE_ADDR" : daddr,
            # "HBM_WT_BASE_ADDR" : waddr,
            # "DAT_OUT_BASE_ADDR" : oaddr,
        }
        return TestbenchSIM("testbench_HBM_MVM_afterF2W_output_head_mode", macro_define)

@Tasks.Register("tb.nn.mvm_f16xf16.ohbm", ohbm_accel.OHBM0323)
def MVMF16xF16(args, output, attrs):
    dtensor, wtensor = args[0], args[1]
    dshape, wshape = dtensor.shape, wtensor.shape
    daddr = dtensor.static_address
    waddr = wtensor.static_address
    oaddr = output[0].static_address
    macro_define = {
        "Last_Token" : wshape[-2] - dshape[-2],
        "This_Token" : wshape[-2],
        "Token_CHin" : dshape[-1],
        "Original_Feature_Head" : dshape[0],
        "Weight_Head" : wshape[0],
        "MAX_TOKEN" : dtensor.device.MAX_TOKEN,
        # "DAT_IN_BASE_ADDR" : daddr,
        # "HBM_WT_BASE_ADDR" : waddr,
        # "DAT_OUT_BASE_ADDR" : oaddr,
    }
    if attrs.get("w_trp"):
        macro_define["Token_CHout"] = wshape[-2]
        if hasattr(dtensor, "heads"):
            macro_define["Feature_Head"] = dtensor.heads[0]
        return TestbenchSIM("testbench_HBM_MVM_afterTRP_input_head_mode", macro_define)
    else:
        macro_define["Token_CHout"] = wshape[-1]
        return TestbenchSIM("testbench_HBM_MVM_afterF2W_output_head_mode", macro_define)

@Op.RegisterAttrs("nn.mvm_f16xf16", "testbench", ohbm_accel.OHBM)
def tb_nn_mm_f16xf16(args, output, attrs):
    if len(get_vars([args[0].shape, attrs])):
        raise RuntimeError("Unsupport dynamic symbol control in testbench simulation")
    device = args[0].device
    with ir.Function([]) as func:
        csbs = Tasks.Get("tb.nn.mvm_f16xf16.ohbm", device)(args, output, attrs)
        for csb in csbs:
            if csb[0]:
                func += ir.CSB_Write(csb[1], csb[2])
            else:
                func += While(CSB_Read(csb[1]) != 1)
    return func


#########################################################################################
#                            nn.kvcache2hbm testbench task                              #
#########################################################################################
@Tasks.Register("tb.nn.kvcache2hbm.ohbm", ohbm_accel.OHBM)
def Kvcache2HBM(args, output, attrs):
    dtensor = args[0]
    dshape = dtensor.shape
    oshape = output[0].shape
    daddr = dtensor.static_address
    oaddr = output[0].static_address
    macro_define = {
        "last_token" : attrs.get("cache_len"),
        "Token" : oshape[-2],
        "Width_in" : dshape[-1],
        "Weight_Head" : dshape[0],
        "K_TRP_mode" : 1 if attrs.get("k_mode") else 0,
        "MAX_TOKEN" : attrs.get("cache_size"),
        # "DAT_IN_BASE_ADDR" : daddr,
        # "HBM_WT_BASE_ADDR" : waddr,
        # "DAT_OUT_BASE_ADDR" : oaddr,
    }
    return TestbenchSIM("testbench_KVcache2HBM", macro_define)


@Tasks.Register("tb.nn.kvcache2hbm.ohbm", ohbm_accel.OHBM0323)
def Kvcache2HBM(args, output, attrs):
    dtensor = args[0]
    dshape = dtensor.shape
    oshape = output[0].shape
    daddr = dtensor.static_address
    oaddr = output[0].static_address
    macro_define = {
        "Last_Token" : attrs.get("cache_len"),
        "This_Token" : oshape[-2],
        "Token_CHin" : dshape[-1],
        "Weight_Head" : dshape[0],
        "K_TRP_mode" : 1 if attrs.get("k_mode") else 0,
        "MAX_TOKEN" : attrs.get("cache_size"),
        # "DAT_IN_BASE_ADDR" : daddr,
        # "HBM_WT_BASE_ADDR" : waddr,
        # "DAT_OUT_BASE_ADDR" : oaddr,
    }
    return TestbenchSIM("testbench_KVcache2HBM", macro_define)


@Op.RegisterAttrs("nn.kvcache2hbm", "testbench", ohbm_accel.OHBM)
def tb_nn_kvcache2hbm(args, output, attrs):
    if len(get_vars([args[0].shape, attrs])):
        raise RuntimeError("Unsupport dynamic symbol control in testbench simulation")
    device = args[0].device
    with ir.Function([]) as func:
        csbs = Tasks.Get("tb.nn.kvcache2hbm.ohbm", device)(args, output, attrs)
        for csb in csbs:
            if csb[0]:
                func += ir.CSB_Write(csb[1], csb[2])
            else:
                func += While(CSB_Read(csb[1]) != 1)
    return func


#########################################################################################
#                                nn.rope testbench task                                 #
#########################################################################################
@Tasks.Register("tb.nn.rope.ohbm", ohbm_accel.OHBM)
def PosEmb(args, output, attrs):
    dtensor, wtensor = args[0], args[1]
    dshape, wshape = dtensor.shape, wtensor.shape
    daddr = dtensor.static_address
    waddr = wtensor.static_address
    oaddr = output[0].static_address
    macro_define = {
        "last_token" : attrs.get("last_token"),
        "Token" : dshape[-2] + attrs.get("last_token", 0),
        "Feature_Head" : dshape[0],
        "MAX_TOKEN" : dtensor.device.MAX_TOKEN,
        # "DAT_IN_BASE_ADDR" : daddr,
        # "HBM_WT_BASE_ADDR" : waddr,
        # "DAT_OUT_BASE_ADDR" : oaddr,
    }
    if attrs.get("mode") == RoPEMode.glm:
        return TestbenchSIM("testbench_EMB_GLM_inout_head_mode", macro_define)
    elif attrs.get("mode") == RoPEMode.qwen:
        return TestbenchSIM("testbench_EMB_Qwen_inout_head_mode", macro_define)
    else:
        raise RuntimeError("no found realize for this mode: " + attrs.get("mode"))
        return TestbenchSIM("testbench_EMB_Qwen_inout_head_mode", macro_define)

@Tasks.Register("tb.nn.rope.ohbm", ohbm_accel.OHBM0314)
def PosEmb(args, output, attrs):
    dtensor, wtensor = args[0], args[1]
    dshape, wshape = dtensor.shape, wtensor.shape
    daddr = dtensor.static_address
    waddr = wtensor.static_address
    oaddr = output[0].static_address
    macro_define = {
        "last_token" : attrs.get("last_token"),
        "Token" : dshape[-2] + attrs.get("last_token", 0),
        "Padding_Feature_Head" : dshape[0],
        "MAX_TOKEN" : dtensor.device.MAX_TOKEN,
        # "DAT_IN_BASE_ADDR" : daddr,
        # "HBM_WT_BASE_ADDR" : waddr,
        # "DAT_OUT_BASE_ADDR" : oaddr,
    }
    if attrs.get("mode") == RoPEMode.glm:
        return TestbenchSIM("testbench_EMB_GLM_inout_head_mode", macro_define)
    elif attrs.get("mode") == RoPEMode.qwen:
        return TestbenchSIM("testbench_EMB_Qwen_inout_head_mode", macro_define)
    else:
        raise RuntimeError("no found realize for this mode: " + attrs.get("mode"))

@Tasks.Register("tb.nn.rope.ohbm", ohbm_accel.OHBM0323)
def PosEmb(args, output, attrs):
    dtensor, wtensor = args[0], args[1]
    dshape, wshape = dtensor.shape, wtensor.shape
    daddr = dtensor.static_address
    waddr = wtensor.static_address
    oaddr = output[0].static_address
    macro_define = {
        "Last_Token" : attrs.get("last_token"),
        "This_Token" : dshape[-2] + attrs.get("last_token", 0),
        "Padding_Feature_Head" : dshape[0],
        "MAX_TOKEN" : dtensor.device.MAX_TOKEN,
        # "DAT_IN_BASE_ADDR" : daddr,
        # "HBM_WT_BASE_ADDR" : waddr,
        # "DAT_OUT_BASE_ADDR" : oaddr,
    }
    if attrs.get("mode") == RoPEMode.glm:
        return TestbenchSIM("testbench_EMB_GLM_inout_head_mode", macro_define)
    elif attrs.get("mode") == RoPEMode.qwen:
        return TestbenchSIM("testbench_EMB_Qwen_inout_head_mode", macro_define)
    else:
        raise RuntimeError("no found realize for this mode: " + attrs.get("mode"))

@Op.RegisterAttrs("nn.rope", "testbench", ohbm_accel.OHBM)
def tb_glm_pos_emb(args, output, attrs):
    if len(get_vars([args[0].shape, attrs])):
        raise RuntimeError("Unsupport dynamic symbol control in testbench simulation")
    device = args[0].device
    with ir.Function([]) as func:
        csbs = Tasks.Get("tb.nn.rope.ohbm", device)(args, output, attrs)
        for csb in csbs:
            if csb[0]:
                func += ir.CSB_Write(csb[1], csb[2])
            else:
                func += While(CSB_Read(csb[1]) != 1)
    return func


#########################################################################################
#                                nn.conv2d testbench task                               #
#########################################################################################
@Tasks.Register("tb.nn.conv2d.ohbm", ohbm_accel.OHBM0314)
def Conv2d(args, output, attrs):
    if len(args) == 3:
        dtensor, wtensor = args[0], args[1]
        dshape, wshape = dtensor.shape, wtensor.shape
        daddr = dtensor.static_address
        waddr = wtensor.static_address
        oaddr = output[0].static_address
        macro_define = {
            "last_token" : 0,
            "Token" : dshape[-2],
            "Hin": dshape[0],
            "RELU_EN" : 1 if attrs.get("relu") else 0,
            "Width_in" : dshape[-1],
            "Width_out" : wshape[-1],
            "BN_RELU_EN" : 0,
            "Ky": wshape[0],
            "Kx": wshape[1],
            "Sy": attrs.get("strides")[0],
            "Sx": attrs.get("strides")[1],
            "Py": attrs.get("padding")[0],
            "Px": attrs.get("padding")[1],
            # "DAT_IN_BASE_ADDR" : daddr,
            # "HBM_WT_BASE_ADDR" : waddr,
            # "DAT_OUT_BASE_ADDR" : oaddr,
        }
        return TestbenchSIM("testbench_HBM_CNN_BN", macro_define)
    else:
        raise RuntimeError("not support modes except conv2d+bn in tb")


@Tasks.Register("tb.nn.conv2d.ohbm", ohbm_accel.OHBM0323)
def Conv2d(args, output, attrs):
    if len(args) == 3:
        dtensor, wtensor = args[0], args[1]
        dshape, wshape = dtensor.shape, wtensor.shape
        daddr = dtensor.static_address
        waddr = wtensor.static_address
        oaddr = output[0].static_address
        macro_define = {
            "Last_Token" : 0,
            "This_Token" : dshape[-2],
            "Hin": dshape[0],
            "RELU_EN" : 1 if attrs.get("relu") else 0,
            "Token_CHin" : dshape[-1],
            "Token_CHout" : wshape[-1],
            "BN_RELU_EN" : 0,
            "Ky": wshape[0],
            "Kx": wshape[1],
            "Sy": attrs.get("strides")[0],
            "Sx": attrs.get("strides")[1],
            "Py": attrs.get("padding")[0],
            "Px": attrs.get("padding")[1],
            # "DAT_IN_BASE_ADDR" : daddr,
            # "HBM_WT_BASE_ADDR" : waddr,
            # "DAT_OUT_BASE_ADDR" : oaddr,
        }
        return TestbenchSIM("testbench_HBM_CNN_BN", macro_define)
    else:
        raise RuntimeError("not support modes except conv2d+bn in tb")


@Op.RegisterAttrs("nn.conv2d", "testbench", ohbm_accel.OHBM)
def tb_nn_conv2d(args, output, attrs):
    if len(get_vars([args[0].shape, attrs])):
        raise RuntimeError("Unsupport dynamic symbol control in testbench simulation")
    device = args[0].device
    with ir.Function([]) as func:
        csbs = Tasks.Get("tb.nn.conv2d.ohbm", device)(args, output, attrs)
        for csb in csbs:
            if csb[0]:
                func += ir.CSB_Write(csb[1], csb[2])
            else:
                func += While(CSB_Read(csb[1]) != 1)
    return func


