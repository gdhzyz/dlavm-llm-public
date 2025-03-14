from dlavm import ne
from dlavm.adr import Op, Attrs
from dlavm.device import ohbm_accel
from .... import ir
from ....basic import Tasks, get_vars
from . import (
    tasks_0303
)


@Op.RegisterAttrs("glm.pos_emb", "compute", ohbm_accel.OHBM)
def PosEmb(args, outputs, attrs):
    device = args[0].device
    with ir.Function(get_vars([args[0].shape, attrs])) as func:
        Tasks.Get("ohbm.glm.pos_emb", device)(func, args, outputs, attrs)
    return func
