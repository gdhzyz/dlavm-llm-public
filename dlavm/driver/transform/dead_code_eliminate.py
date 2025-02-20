from copy import deepcopy
from dlavm import ne
from .. import ir


class DeadCodeEliminateCheck(ir.Visitor):

    def __init__(self):
        super().__init__()
    
    def main(self, stmt: ir.Function) -> list:
        self.unused_vars = {}
        self.Visit(stmt)
        return self.unused_vars.values()

    def VisitAssign(self, stmt: ir.Assign):
        new_stmt = super().VisitAssign(stmt)
        self.unused_vars[new_stmt.var.name] = new_stmt
        return new_stmt

    def VisitNe(self, expr: ne.Expr):
        new_expr = super().VisitNe(expr)
        vars = new_expr.get_vars()
        for v in vars:
            vname = v[0]
            if vname in list(self.unused_vars.keys()):
                del self.unused_vars[vname]
        return new_expr


class DeadCodeEliminate(ir.Functor):

    def __init__(self):
        super().__init__()

    def VisitFunction(self, stmt: ir.Function):
        self.unused_assign = DeadCodeEliminateCheck().main(stmt)
        return super().VisitFunction(stmt)

    def VisitAssign(self, stmt: ir.Assign):
        def check(t, s):
            return t.value == s.value and t.dtype == s.dtype and t.var == s.var
        for i in self.unused_assign:
            if check(i, stmt):
                return ir.Empty
        new_stmt = super().VisitAssign(stmt)
        return new_stmt