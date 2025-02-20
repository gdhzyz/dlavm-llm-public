from copy import deepcopy
from dlavm import ne
from .. import ir


class FoldConstant(ir.Functor):

    def __init__(self):
        super().__init__()

    def VisitFunction(self, stmt: ir.Function):
        self.var2numb = {}
        return super().VisitFunction(stmt)

    def VisitAssign(self, stmt: ir.Assign):
        new_stmt = super().VisitAssign(stmt)
        if isinstance(new_stmt.value, ne.Expr):
            new_stmt.value = new_stmt.value.simplify()
            if isinstance(new_stmt.value, ne.Numb):
                self.var2numb[new_stmt.var.name] = new_stmt.value.data
        elif isinstance(new_stmt.value, (str, int, float)):
            self.var2numb[new_stmt.var.name] = new_stmt.value
        return new_stmt

    def VisitNe(self, expr: ne.Expr):
        new_expr = ne.expr_var_from_dict(expr, self.var2numb).simplify()
        if isinstance(new_expr, ne.Numb):
            new_expr = new_expr.data
        return new_expr
