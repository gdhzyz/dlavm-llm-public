from dlavm import ne
from dlavm.adr import Op, Attrs
from dlavm.device import ohbm_accel
from ... import ir
from ...ir import CSB_Write, CSB_Read, While
from ...basic import TestbenchSIM, Tasks

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

@Tasks.Register("tb.nn.mvm.ohbm", ohbm_accel.OHBM)
def MVM(args, output, attrs):
    if len(args) == 2:
        dtensor, wtensor = args[0], args[1]
        dshape, wshape = dtensor.shape, wtensor.shape
        daddr = dtensor.static_address
        waddr = wtensor.static_address
        oaddr = output[0].static_address
        macro_define = {
            "last_token" : attrs.get("last_token", 0),
            "Token" : dshape[-2],
            "RELU_EN" : attrs.get("relu", 0),
            "Width_in" : dshape[-1],
            "Width_out" : wshape[0],
            # "DAT_IN_BASE_ADDR" : daddr,
            # "HBM_WT_BASE_ADDR" : waddr,
            # "DAT_OUT_BASE_ADDR" : oaddr,
        }
        return TestbenchSIM("testbench_HBM_MVM", macro_define)
    else:
        raise RuntimeError("not support mvm with bn or res in tb")

@Op.RegisterAttrs("nn.mvm", "testbench", ohbm_accel.OHBM)
def tb_nn_mvm(args, output, attrs):
    if len(get_vars([args[0].shape, attrs])):
        raise RuntimeError("Unsupport dynamic symbol control in testbench simulation")
    device = args[0].device
    with ir.Function([]) as func:
        csbs = Tasks.Get("tb.nn.mvm.ohbm", device)(args, output, attrs)
        for csb in csbs:
            if csb[0]:
                func += ir.CSB_Write(csb[1], csb[2])
            else:
                func += While(CSB_Read(csb[1]) != 1)
    return func
