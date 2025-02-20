import os
from .. import adr, driver
from ..adr import Functor, Call, VM, Tensor


class OfflineStore(Functor):

    def __init__(self, prefix="./weight"):
        super(OfflineStore, self).__init__()
        self.prefix = prefix
        self.id = 0

    def visit_constant(self, expr: adr.Constant):
        if expr.dtype.mapped == adr.DataEnum.hbm:
            for numb, mapped_port in enumerate(expr.data):
                with open(os.path.join(self.prefix, f"wt{self.id}_hbm_{numb}.bin"), "wb") as f:
                    f.write(mapped_port.tobytes())
            expr.data = os.path.join(self.prefix, f"wt{self.id}_hbm_%d.bin")
        elif expr.dtype.mapped == adr.DataEnum.ddr:
            with open(os.path.join(self.prefix, f"wt{self.id}_ddr.bin"), "wb") as f:
                f.write(expr.data.tobytes())
            expr.data = os.path.join(self.prefix, f"wt{self.id}_ddr.bin")
        self.id += 1
        return expr


class OfflineProcess(Functor):

    def visit_call(self, expr):
        expr.args = [self.visit(arg) for arg in expr.args]
        if "process" in expr.op.attrs.keys():
            func = expr.op.attrs["process"]
            func(expr.args, expr.attrs)
        return expr


def offline_process(expr):
    return OfflineProcess().visit(expr)


def offline(expr, prefix="weight"):
    expr = OfflineProcess().visit(expr)
    expr = OfflineStore(prefix).visit(expr)
    return expr