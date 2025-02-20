from ..adr import Functor, Call
from .fused_lib import FusedStrategy


class FusedOps(Functor):

    def __init__(self, opt_level=1):
        super().__init__()
        self.opt_level = opt_level

    def visit_call(self, expr):
        new_args = [self.visit(arg) for arg in expr.args]
        expr.args = new_args
        funcs = FusedStrategy.Get(expr.op.name, self.opt_level)
        if funcs is None:
            return expr
        new_expr = expr
        for func in funcs:
            state, new_expr = func.CheckFused(expr)
            if state:
                break
        return new_expr

    def visit_vm(self, expr):
        new_args = [self.visit(arg) for arg in expr.args]
        expr.args = new_args
        funcs = FusedStrategy.Get(expr.op.name, self.opt_level)
        if funcs is None:
            return expr
        new_expr = expr
        for func in funcs:
            state, new_expr = func.CheckFused(expr)
            if state:
                break
        return new_expr


def fused_ops(expr, opt_level):
    return FusedOps(opt_level=1).visit(expr)