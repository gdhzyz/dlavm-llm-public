from dlavm import ne
from . import base as ir
from .functor import Functor


class Visitor(Functor):

    def VisitOp(self, expr: ir.Op):
        new_expr = expr
        new_expr.arg0 = self.Visit(expr.arg0)
        new_expr.arg1 = self.Visit(expr.arg1)
        return new_expr

    def VisitCSBRead(self, expr: ir.CSB_Read):
        new_expr = expr
        self.Visit(expr.addr)
        return new_expr

    def VisitNe(self, expr: ne.Expr):
        return expr

    def VisitData(self, data):
        return data

    def VisitFunction(self, stmt: ir.Function):
        new_stmt = stmt
        [self.Visit(b) for b in new_stmt.body]
        return new_stmt

    def VisitBlock(self, stmt: ir.Block):
        new_stmt = stmt
        [self.Visit(b) for b in new_stmt.body]
        return new_stmt

    def VisitCall(self, stmt: ir.Call):
        new_stmt = stmt
        self.Visit(new_stmt.func)
        return new_stmt

    def VisitReturn(self, stmt: ir.Return):
        new_stmt = stmt
        new_stmt.data = self.Visit(new_stmt.data)
        return new_stmt

    def VisitWhile(self, stmt: ir.While):
        new_stmt = stmt
        new_stmt.judge = self.Visit(new_stmt.judge)
        new_stmt.body = self.RmEmpty([self.Visit(b) for b in stmt.body])
        return new_stmt

    def VisitFor(self, stmt: ir.For):
        new_stmt = stmt
        self.Visit(new_stmt.init)
        self.Visit(new_stmt.extent)
        self.Visit(new_stmt.stride)
        [self.Visit(b) for b in stmt.body]
        return new_stmt

    def VisitIf(self, stmt: ir.If):
        new_stmt = stmt
        self.Visit(new_stmt.judge)
        [self.Visit(b) for b in stmt.then_block.body]
        [self.Visit(b) for b in stmt.else_block.body]
        return new_stmt

    def VisitAssign(self, stmt: ir.Assign):
        new_stmt = stmt
        self.Visit(stmt.value)
        return new_stmt

    def VisitCSBWrite(self, stmt: ir.CSB_Write):
        new_stmt = stmt
        self.Visit(stmt.addr)
        self.Visit(stmt.data)
        return new_stmt

    def VisitMemWriteFile(self, stmt: ir.MemWriteFile):
        new_stmt = stmt
        self.Visit(stmt.addr)
        self.Visit(stmt.file)
        self.Visit(stmt.size)
        return new_stmt

    def VisitMemInit(self, stmt: ir.MemInit):
        new_stmt = stmt
        self.Visit(stmt.addr)
        self.Visit(stmt.size)
        return new_stmt

    def VisitStrFormat(self, stmt: ir.StrFormat):
        new_stmt = stmt
        self.Visit(stmt.target)
        [self.Visit(arg) for arg in stmt.args]
        return new_stmt

    def VisitInplace(self, stmt: ir.Inplace):
        new_stmt = stmt
        self.Visit(stmt.data)
        return new_stmt

    def VisitCast(self, expr: ir.Cast):
        new_expr = expr
        self.Visit(expr.var)
        return new_expr

    def VisitVar(self, expr: ir.Var):
        new_expr = expr
        self.Visit(expr.var)
        return new_expr

