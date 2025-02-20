from copy import deepcopy
from dlavm import ne
from .. import ir


class FunctorLoop(ir.Functor):

    def __init__(self, var_map):
        super().__init__()
        self.var_map = var_map

    def VisitNe(self, expr: ne.Expr):
        new_expr = ne.expr_var_from_dict(super().VisitNe(expr), self.var_map)
        if isinstance(new_expr, ne.Numb):
            new_expr = new_expr.data
        return new_expr


class LoopSimplify(ir.Functor):

    def __init__(self, min_loop=2, eliminate=False):
        super().__init__()
        self.min_loop = min_loop
        self.eliminate = eliminate

    def VisitNe(self, expr: ne.Expr):
        new_expr = super().VisitNe(expr)
        if isinstance(new_expr, ne.Numb):
            new_expr = new_expr.data
        return new_expr

    def VisitFor(self, stmt: ir.For):
        new_stmt = super().VisitFor(stmt)
        if self.eliminate:
            with ir.Block() as block:
                functor = FunctorLoop(var_map={new_stmt.var.name : new_stmt.init})
                for b in new_stmt.body:
                    block += functor.Visit(b)
            return block
        elif isinstance(new_stmt.init, (int, float)) and \
           isinstance(new_stmt.extent, (int, float)) and \
           isinstance(new_stmt.stride, (int, float)):
            py_loop = range(new_stmt.init, new_stmt.extent, new_stmt.stride)
            loop_numb = len(py_loop)
            if loop_numb <= self.min_loop:
                with ir.Block() as block:
                    for i in py_loop:
                        functor = FunctorLoop(var_map={new_stmt.var.name : i})
                        for b in new_stmt.body:
                            block += functor.Visit(b)
                return block
        return new_stmt

