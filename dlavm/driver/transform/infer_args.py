from copy import deepcopy
from dlavm import ne
from .. import ir


class InferArgs(ir.Functor):

    def __init__(self, handle=True):
        super().__init__()
        self.handle = handle
        self.device, self.h2cx, self.c2hx = False, False, False
        self.used_args = []

    def VisitFunction(self, stmt: ir.Function):
        self.used_args = []
        self.device, self.h2cx, self.c2hx = False, False, False
        new_stmt = super().VisitFunction(stmt)
        func_args = [arg for arg in stmt.args if arg in self.used_args]
        if self.handle:
            if self.c2hx and ne.Var("c2hx", -1, "HANDLE&") not in func_args:
                func_args.insert(0, ne.Var("c2hx", -1, "HANDLE&"))
            if self.h2cx and ne.Var("h2cx", -1, "HANDLE&") not in func_args:
                func_args.insert(0, ne.Var("h2cx", -1, "HANDLE&"))
            if self.device and ne.Var("device", -1, "HANDLE&") not in func_args:
                func_args.insert(0, ne.Var("device", -1, "HANDLE&"))
        new_stmt.args = func_args
        return new_stmt

    def VisitCall(self, stmt: ir.Call):
        new_stmt = deepcopy(stmt)
        functor = InferArgs(self.handle)
        new_func = functor.Visit(new_stmt.func)
        self.device = self.device | functor.device
        self.h2cx = self.h2cx | functor.h2cx
        self.c2hx = self.c2hx | functor.c2hx
        new_stmt.func = new_func
        new_stmt.args = {arg: arg for arg in new_func.args}
        self.used_args += new_func.args
        return new_stmt

    def VisitCSBWrite(self, stmt: ir.CSB_Write):
        self.device = True
        return super().VisitCSBWrite(stmt)

    def VisitCSBRead(self, expr: ir.CSB_Read):
        self.device = True
        return super().VisitCSBRead(expr)

    def VisitMemWriteFile(self, stmt: ir.MemWriteFile):
        self.h2cx = True
        return super().VisitMemWriteFile(stmt)

    def VisitMemWrite(self, stmt: ir.MemWrite):
        self.h2cx = True
        return super().VisitMemWrite(stmt)

    def VisitMemInit(self, stmt: ir.MemInit):
        self.h2cx = True
        return super().VisitMemInit(stmt)

    def VisitNe(self, expr: ne.Expr):
        self.used_args += expr.get_vars(True)
        return expr