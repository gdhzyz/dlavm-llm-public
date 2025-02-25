from copy import deepcopy
from dlavm import ne
from .. import ir


class FoldConstant(ir.Functor):

    def __init__(self):
        super().__init__()
        self.branch = False

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

    def VisitAssignVar(self, stmt: ir.AssignVar):
        new_stmt = super().VisitAssignVar(stmt)
        if isinstance(new_stmt.value, ne.Expr):
            new_stmt.value = new_stmt.value.simplify()
            if isinstance(new_stmt.value, ne.Numb) and not self.branch:
                self.var2numb[new_stmt.var.name] = new_stmt.value.data
        elif isinstance(new_stmt.value, (str, int, float)) and not self.branch:
            self.var2numb[new_stmt.var.name] = new_stmt.value
        if self.branch and new_stmt.var.name in self.var2numb.keys():
            del self.var2numb[new_stmt.var.name]
        return new_stmt

    def VisitNe(self, expr: ne.Expr):
        new_expr = ne.expr_var_from_dict(expr, self.var2numb).simplify()
        if isinstance(new_expr, ne.Numb):
            new_expr = new_expr.data
        return new_expr

    def VisitIf(self, stmt: ir.If):
        new_stmt = deepcopy(stmt)
        new_stmt.then_block = deepcopy(stmt.then_block)
        new_stmt.else_block = deepcopy(stmt.else_block)
        new_stmt.judge = self.Visit(stmt.judge)
        if isinstance(new_stmt.judge, (int, float)):
            new_block = ir.Block()
            if new_stmt.judge:
                new_block.body = self.RmEmpty([self.Visit(b) for b in stmt.then_block.body])
            else:
                new_block.body = self.RmEmpty([self.Visit(b) for b in stmt.else_block.body])
            return new_block
        self.branch = True
        new_stmt.then_block.body = self.RmEmpty([self.Visit(b) for b in stmt.then_block.body])
        new_stmt.else_block.body = self.RmEmpty([self.Visit(b) for b in stmt.else_block.body])
        self.branch = False
        return new_stmt
