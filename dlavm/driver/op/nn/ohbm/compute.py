from dlavm import ne
from dlavm.adr import Op, Attrs
from dlavm.device import ohbm_accel
from .... import ir
from ....basic import Tasks, get_vars, ten2list
from . import (
    tasks_0219,
    tasks_0316,
    tasks_0323,
    tasks_0326,
    tasks_0407
)


@Op.RegisterAttrs("nn.mvm_f16xi4", "compute", ohbm_accel.OHBM)
def MVM(args, outputs, attrs):
    device = args[0].device
    vars_args = ten2list(args[0]) + [attrs]
    with ir.Function(get_vars(vars_args)) as func:
        Tasks.Get("ohbm.nn.mvm", device)(func, args, outputs, attrs)
    return func


@Op.RegisterAttrs("nn.mvm_f16xf16", "compute", ohbm_accel.OHBM)
def MVMF16xF16(args, outputs, attrs):
    device = args[0].device
    vars_args = ten2list(args[0]) + ten2list(args[1]) + [attrs]
    with ir.Function(get_vars(vars_args)) as func:
        Tasks.Get("ohbm.nn.mvm_f16xf16", device)(func, args, outputs, attrs)
    return func


@Op.RegisterAttrs("nn.conv2d", "compute", ohbm_accel.OHBM)
def Conv2d(args, outputs, attrs):
    device = args[0].device
    vars_args = ten2list(args[0]) + [attrs]
    with ir.Function(get_vars(vars_args)) as func:
        Tasks.Get("ohbm.nn.conv2d", device)(func, args, outputs, attrs)
    return func


@Op.RegisterAttrs("nn.norm", "compute", ohbm_accel.OHBM)
def Norm(args, outputs, attrs):
    device = args[0].device
    vars_args = ten2list(args[0]) + [attrs]
    with ir.Function(get_vars(vars_args)) as func:
        Tasks.Get("ohbm.nn.norm", device)(func, args, outputs, attrs)
    return func


@Op.RegisterAttrs("nn.softmax", "compute", ohbm_accel.OHBM)
def Softmax(args, outputs, attrs):
    device = args[0].device
    vars_args = ten2list(args[0]) + [attrs]
    with ir.Function(get_vars(vars_args)) as func:
        Tasks.Get("ohbm.nn.softmax", device)(func, args, outputs, attrs)
    return func


@Op.RegisterAttrs("nn.elementwise", "compute", ohbm_accel.OHBM)
def Elementwise(args, outputs, attrs):
    device = args[0].device
    vars_args = ten2list(args[0]) + [attrs]
    with ir.Function(get_vars(vars_args)) as func:
        Tasks.Get("ohbm.nn.elementwise", device)(func, args, outputs, attrs)
    return func


@Op.RegisterAttrs("nn.activate", "compute", ohbm_accel.OHBM)
def Activate(args, outputs, attrs):
    device = args[0].device
    vars_args = ten2list(args[0]) + [attrs]
    with ir.Function(get_vars(vars_args)) as func:
        Tasks.Get("ohbm.nn.activate", device)(func, args, outputs, attrs)
    return func


@Op.RegisterAttrs("nn.kvcache2hbm", "compute", ohbm_accel.OHBM)
def Kvcache2hbm(args, outputs, attrs):
    device = args[0].device
    vars_args = ten2list(args[0]) + [attrs]
    with ir.Function(get_vars(vars_args)) as func:
        Tasks.Get("ohbm.nn.kvcache2hbm", device)(func, args, outputs, attrs)
    return func


@Op.RegisterAttrs("nn.rope", "compute", ohbm_accel.OHBM)
def PosEmb(args, outputs, attrs):
    device = args[0].device
    vars_args = ten2list(args[0]) + [attrs]
    with ir.Function(get_vars(vars_args)) as func:
        Tasks.Get("ohbm.nn.rope", device)(func, args, outputs, attrs)
    return func


@Op.RegisterAttrs("nn.empty", "compute", ohbm_accel.OHBM)
def Empty(args, outputs, attrs):
    return ir.Empty

