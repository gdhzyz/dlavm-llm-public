from copy import deepcopy
from dlavm import ne
from .. import ir


class AddDebugSign(ir.Functor):

    def VisitFunction(self, stmt: ir.Function):
        self.func_name = stmt.name
        return super().VisitFunction(stmt)


    def VisitWhile(self, stmt: ir.While):
        new_stmt = deepcopy(stmt)
        new_stmt.judge = self.Visit(stmt.judge)
        new_stmt.body = self.RmEmpty([self.Visit(b) for b in stmt.body])
        #if isinstance(new_stmt.judge, ir.CSB_Read):
        print(type(stmt.judge))
        with ir.Block() as b:
            with ir.MacroDefine("PRINT_STEP") as m:
                m += ir.ExternCall("printf", [f"start {self.func_name}"])
            b += m
            b += new_stmt
        return b
        print(new_stmt)
        return new_stmt