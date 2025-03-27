from copy import deepcopy
from dlavm import ne
from .. import ir


class AddDebugSign(ir.Functor):

    def VisitFunction(self, stmt: ir.Function):
        self.func_name = stmt.name
        self.sign = 0
        return super().VisitFunction(stmt)
    
    def VisitCSBRead(self, expr):
        self.sign = 1
        return super().VisitCSBRead(expr)

    def VisitWhile(self, stmt: ir.While):
        new_stmt = deepcopy(stmt)
        self.sign = 0
        new_stmt.judge = self.Visit(stmt.judge)
        new_stmt.body = self.RmEmpty([self.Visit(b) for b in stmt.body])
        if self.sign:
            with ir.Block() as b:
                with ir.MacroDefine("PRINT_STEP") as m:
                    m += ir.ExternCall("printf", [f"start {self.func_name}\\n"])
                b += m
                b += new_stmt
            return b
        return new_stmt