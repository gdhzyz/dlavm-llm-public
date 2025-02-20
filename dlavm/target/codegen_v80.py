from dlavm.driver import ir, transform
from . import targets
from .codegen_base import CodeGenEngine
from .codegen_h import CodeGenH


@CodeGenEngine.Register(targets.v80)
class CodeGenV80(CodeGenH):

    def main(self, stmt: ir.Function):
        stmt = transform.InferArgs(handle=False).Visit(stmt)
        self._memo_lib = []
        source = self.Visit(stmt)
        return source

    def VisitCSBWrite(self, stmt: ir.CSB_Write):
        tabs = self.tab * self.tab_num
        addr = self.Visit(stmt.addr)
        data = self.Visit(stmt.data)
        return tabs + f"CSB_Write({addr}, {data});"

    def VisitCSBRead(self, expr: ir.CSB_Read):
        addr = self.Visit(expr.addr)
        return f"CSB_Read({addr})"

    def VisitMemWrite(self, stmt: ir.MemWrite):
        tabs = self.tab * self.tab_num
        addr = self.Visit(stmt.addr)
        data = self.Visit(stmt.data)
        return tabs + f"DDR_Update({addr}, {data});"

    def VisitMemWriteFile(self, stmt: ir.MemWriteFile):
        self.h2cx = True
        tabs = self.tab * self.tab_num
        addr = self.Visit(stmt.addr)
        size = self.Visit(stmt.size)
        return tabs + f"DDR_Write_bin({stmt.file}, {addr}, {size});"

    def VisitMemInit(self, stmt: ir.MemInit):
        self.h2cx = True
        tabs = self.tab * self.tab_num
        addr = self.Visit(stmt.addr)
        size = self.Visit(stmt.size)
        return tabs + f"// init({addr}, {size});"