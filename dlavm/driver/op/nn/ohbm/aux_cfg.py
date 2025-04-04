from dlavm import ne
from dlavm.adr import Op, Attrs
from dlavm.device import ohbm_accel
from .... import ir
from ....basic import Tasks, get_vars, ten2list
from ....ir import CSB_Write, CSB_Read, While


#########################################################################################
#                                nn.softmax aux-cfg task                                #
#########################################################################################
@Tasks.Register("ohbm.nn.softmax.aux-cfg", ohbm_accel.OHBM)
def Softmax(args, attrs):
    return [0b0000_0100, 2]


#########################################################################################
#                                nn.rope aux-cfg task                                   #
#########################################################################################
@Tasks.Register("ohbm.nn.rope.aux-cfg", ohbm_accel.OHBM)
def RoPE(args, attrs):
    return [0b0000_0010, 2]


#########################################################################################
#                             nn.kvcache2hbm aux-cfg task                               #
#########################################################################################
@Tasks.Register("ohbm.nn.kvcache2hbm.aux-cfg", ohbm_accel.OHBM)
def Kvcache2hbm(args, attrs):
    return [0b0000_0011, 2]


#########################################################################################
#                                nn.norm aux-cfg task                                   #
#########################################################################################
@Tasks.Register("ohbm.nn.norm.aux-cfg", ohbm_accel.OHBM)
def Norm(args, attrs):
    return [0b0000_0001, 2]


#########################################################################################
#                             nn.elementwise aux-cfg task                               #
#########################################################################################
@Tasks.Register("ohbm.nn.elementwise.aux-cfg", ohbm_accel.OHBM)
def Elementwise(args, attrs):
    return [0b0000_0110, 2]


#########################################################################################
#                               nn.activate aux-cfg task                                #
#########################################################################################
@Tasks.Register("ohbm.nn.activate.aux-cfg", ohbm_accel.OHBM)
def Activate(args, attrs):
    return [0b0000_0101, 2]


#########################################################################################
#                               nn.mvm_f16xi4 aux-cfg task                              #
#########################################################################################
@Tasks.Register("ohbm.nn.mvm_f16xi4.aux-cfg", ohbm_accel.OHBM)
def MVM(args, attrs):
    if len(args) == 2:
        return [0b0001_0000, 2]
    elif len(args) == 3:
        if attrs.get("argmax"):
            return [0b0111_0000, 2]
        return [0b0011_0000, 2]
    else:
        raise RuntimeError(f"No support MVM config of: length of args {len(args)}")


#########################################################################################
#                               nn.mvm_f16xf16 aux-cfg task                             #
#########################################################################################
@Tasks.Register("ohbm.nn.mvm_f16xf16.aux-cfg", ohbm_accel.OHBM)
def MVMF16xF16(args, attrs):
    if len(args) == 2:
        return [0b0001_0000, 2]
    else:
        raise RuntimeError(f"No support MVM F16xF16 config of: length of args {len(args)}")


#########################################################################################
#                                 nn.conv2d aux-cfg task                                #
#########################################################################################
@Tasks.Register("ohbm.nn.conv2d.aux-cfg", ohbm_accel.OHBM)
def Conv2d(args, attrs):
    if len(args) == 2:
        return [0b0001_0000, 2]
    elif len(args) == 3:
        return [0b0011_0000, 2]
    else:
        raise RuntimeError(f"No support MVM config of: length of args {len(args)}")


#########################################################################################
#                                 op attrs aux-cfg tasks                                #
#########################################################################################
@Tasks.Register("atom.ohbm.aux", ohbm_accel.OHBM)
def AUX(func, inst_addr, aux_numb, task_numb, upt_calls=[]):
    func += CSB_Write(64+1, inst_addr)
    func += CSB_Write(64+2, aux_numb)
    func += CSB_Write(64+3, task_numb)
    func += CSB_Write(64+9, 1)
    for call in upt_calls:
        func += call
        func.update_args(call.func.args)
    func += While(CSB_Read(64) != 1)


@Op.RegisterAttrs("nn.mvm_f16xi4", "aux-cfg", ohbm_accel.OHBM)
def MVM(args, attrs):
    device = args[0].device
    return Tasks.Get("ohbm.nn.mvm_f16xi4.aux-cfg", device)(args, attrs)


@Op.RegisterAttrs("nn.mvm_f16xf16", "aux-cfg", ohbm_accel.OHBM)
def MVMF16xF16(args, attrs):
    device = args[0].device
    return Tasks.Get("ohbm.nn.mvm_f16xf16.aux-cfg", device)(args, attrs)


@Op.RegisterAttrs("nn.conv2d", "aux-cfg", ohbm_accel.OHBM)
def Conv2d(args, attrs):
    device = args[0].device
    return Tasks.Get("ohbm.nn.conv2d.aux-cfg", device)(args, attrs)


@Op.RegisterAttrs("nn.norm", "aux-cfg", ohbm_accel.OHBM)
def Norm(args, attrs):
    device = args[0].device
    return Tasks.Get("ohbm.nn.norm.aux-cfg", device)(args, attrs)


@Op.RegisterAttrs("nn.softmax", "aux-cfg", ohbm_accel.OHBM)
def Softmax(args, attrs):
    device = args[0].device
    return Tasks.Get("ohbm.nn.softmax.aux-cfg", device)(args, attrs)


@Op.RegisterAttrs("nn.elementwise", "aux-cfg", ohbm_accel.OHBM)
def Elementwise(args, attrs):
    device = args[0].device
    return Tasks.Get("ohbm.nn.elementwise.aux-cfg", device)(args, attrs)


@Op.RegisterAttrs("nn.activate", "aux-cfg", ohbm_accel.OHBM)
def Activate(args, attrs):
    device = args[0].device
    return Tasks.Get("ohbm.nn.activate.aux-cfg", device)(args, attrs)


@Op.RegisterAttrs("nn.kvcache2hbm", "aux-cfg", ohbm_accel.OHBM)
def Kvcache2hbm(args, attrs):
    device = args[0].device
    return Tasks.Get("ohbm.nn.kvcache2hbm.aux-cfg", device)(args, attrs)


@Op.RegisterAttrs("nn.rope", "aux-cfg", ohbm_accel.OHBM)
def PosEmb(args, attrs):
    device = args[0].device
    return Tasks.Get("ohbm.nn.rope.aux-cfg", device)(args, attrs)

