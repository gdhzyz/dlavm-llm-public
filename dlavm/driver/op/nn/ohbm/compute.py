from dlavm import ne
from dlavm.adr import Op, Attrs
from dlavm.device import ohbm_accel
from .... import ir
from ....basic import Tasks
from . import (
    tasks_0219
)

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

@Op.RegisterAttrs("nn.mvm_f16xi4", "compute", ohbm_accel.OHBM)
def MVM(args, outputs, attrs):
    device = args[0].device
    with ir.Function(get_vars([args[0].shape, attrs])) as func:
        Tasks.Get("ohbm.nn.mvm", device)(func, args, outputs, attrs)
    return func


@Op.RegisterAttrs("nn.norm", "compute", ohbm_accel.OHBM)
def Norm(args, outputs, attrs):
    device = args[0].device
    with ir.Function(get_vars([args[0].shape, attrs])) as func:
        Tasks.Get("ohbm.nn.norm", device)(func, args, outputs, attrs)
    return func


@Op.RegisterAttrs("nn.softmax", "compute", ohbm_accel.OHBM)
def Softmax(args, outputs, attrs):
    device = args[0].device
    with ir.Function(get_vars([args[0].shape, attrs])) as func:
        Tasks.Get("ohbm.nn.softmax", device)(func, args, outputs, attrs)
    return func


@Op.RegisterAttrs("nn.elementwise", "compute", ohbm_accel.OHBM)
def Elementwise(args, outputs, attrs):
    device = args[0].device
    with ir.Function(get_vars([args[0].shape, attrs])) as func:
        Tasks.Get("ohbm.nn.elementwise", device)(func, args, outputs, attrs)
    return func


@Op.RegisterAttrs("nn.activate", "compute", ohbm_accel.OHBM)
def Activate(args, outputs, attrs):
    device = args[0].device
    with ir.Function(get_vars([args[0].shape, attrs])) as func:
        Tasks.Get("ohbm.nn.activate", device)(func, args, outputs, attrs)
    return func
