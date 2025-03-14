from dlavm import ne
from dlavm.adr import Op, Attrs
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


#####################################################################
@Tasks.Register("tb.glm.pos_emb.ohbm", ohbm_accel.OHBM)
def PosEmb(args, output, attrs):
    dtensor, wtensor = args[0], args[1]
    dshape, wshape = dtensor.shape, wtensor.shape
    daddr = dtensor.static_address
    waddr = wtensor.static_address
    oaddr = output[0].static_address
    macro_define = {
        "last_token" : attrs.get("last_token", 0),
        "Token" : dshape[-2] + attrs.get("last_token", 0),
        "Feature_Head" : dshape[0],
        "MAX_TOKEN" : dtensor.device.MAX_TOKEN,
        # "DAT_IN_BASE_ADDR" : daddr,
        # "HBM_WT_BASE_ADDR" : waddr,
        # "DAT_OUT_BASE_ADDR" : oaddr,
    }
    if hasattr(dtensor, "heads"):
        macro_define["Feature_Head"] = dtensor.heads[-1]
    return TestbenchSIM("testbench_EMB_GLM_inout_head_mode", macro_define)

@Op.RegisterAttrs("glm.pos_emb", "testbench", ohbm_accel.OHBM)
def tb_glm_pos_emb(args, output, attrs):
    if len(get_vars([args[0].shape, attrs])):
        raise RuntimeError("Unsupport dynamic symbol control in testbench simulation")
    device = args[0].device
    with ir.Function([]) as func:
        csbs = Tasks.Get("tb.glm.pos_emb.ohbm", device)(args, output, attrs)
        for csb in csbs:
            if csb[0]:
                func += ir.CSB_Write(csb[1], csb[2])
            else:
                func += While(CSB_Read(csb[1]) != 1)
    return func


