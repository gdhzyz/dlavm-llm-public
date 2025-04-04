from copy import deepcopy
from dlavm import ne
from .. import ir


class GetSubIRFromVars(ir.Functor):

    enable_type = (ir.Function, ir.If, ir.For, ir.Block, ir.BlockSplit, ir.Assign, ir.AssignVar)

    def __init__(self, vars:list):
        super().__init__()
        self.vars = set(vars)

    def VisitFunction(self, stmt: ir.Function):
        new_stmt = deepcopy(stmt)
        new_stmt.body.reverse()
        new_stmt.body = self.RmEmpty([self.Visit(b) for b in new_stmt.body])
        new_stmt.body.reverse()
        if len(new_stmt.body) == 0:
            return ir.Empty
        return new_stmt

    def VisitIf(self, stmt):
        new_stmt = deepcopy(stmt)
        new_stmt.then_block = deepcopy(stmt.then_block)
        new_stmt.else_block = deepcopy(stmt.else_block)
        new_stmt.judge = self.Visit(stmt.judge)
        new_stmt.then_block.body.reverse()
        new_stmt.else_block.body.reverse()
        new_stmt.then_block.body = self.RmEmpty([self.Visit(b) for b in new_stmt.then_block.body])
        new_stmt.else_block.body = self.RmEmpty([self.Visit(b) for b in new_stmt.else_block.body])
        new_stmt.then_block.body.reverse()
        new_stmt.else_block.body.reverse()
        if len(new_stmt.then_block.body) == 0 and len(new_stmt.else_block.body) == 0:
            return ir.Empty
        return new_stmt

    def VisitFor(self, stmt):
        new_stmt = deepcopy(stmt)
        new_stmt.init = self.Visit(stmt.init)
        new_stmt.extent = self.Visit(stmt.extent)
        new_stmt.stride = self.Visit(stmt.stride)
        new_stmt.body.reverse()
        new_stmt.body = self.RmEmpty([self.Visit(b) for b in new_stmt.body])
        new_stmt.body.reverse()
        if len(new_stmt.body) == 0:
            return ir.Empty
        return new_stmt

    def VisitBlock(self, stmt: ir.Block):
        new_stmt = deepcopy(stmt)
        new_stmt.body.reverse()
        new_stmt.body = self.RmEmpty([self.Visit(b) for b in new_stmt.body])
        new_stmt.body.reverse()
        if len(new_stmt.body) == 0:
            return ir.Empty
        return new_stmt

    def VisitBlockSplit(self, stmt: ir.BlockSplit):
        new_stmt = deepcopy(stmt)
        new_stmt.body.reverse()
        new_stmt.body = self.RmEmpty([self.Visit(b) for b in new_stmt.body])
        new_stmt.body.reverse()
        if len(new_stmt.body) == 0:
            return ir.Empty
        return new_stmt

    def _wrap(self, var):
        if isinstance(var, ne.Expr):
            for new_var in var.get_vars():
                self.vars.add(new_var[0])

    def VisitAssign(self, stmt: ir.Assign):
        new_stmt = deepcopy(stmt)
        new_stmt.value = self.Visit(stmt.value)
        if new_stmt.var.name in self.vars:
            self._wrap(new_stmt.value)
            return new_stmt
        else:
            return ir.Empty

    def VisitAssignVar(self, stmt: ir.AssignVar):
        new_stmt = deepcopy(stmt)
        new_stmt.value = self.Visit(stmt.value)
        if new_stmt.var.name in self.vars:
            self._wrap(new_stmt.value)
            return new_stmt
        else:
            return ir.Empty

    def VisitMacroDefine(self, stmt: ir.MacroDefine):
        return ir.Empty

    def VisitWhile(self, stmt):
        return ir.Empty

    def VisitStmt(self, stmt):
        if isinstance(stmt, self.enable_type):
            result = super().VisitStmt(stmt)
        else:
            result = ir.Empty
        return result
