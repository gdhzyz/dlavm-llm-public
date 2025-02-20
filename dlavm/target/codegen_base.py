from dlavm import ne
from dlavm.driver import ir

class CodeGenBase(ir.Functor):

    tab = "  "

    def __init__(self) -> None:
        super().__init__()
        self.id = 0
        self.tab_num = 0
        self.func_name = "step%d"
        self.source = ""
        self.module = []

    def main(self, stmt: ir.Function):
        pass

    def BriefAnnotation(self, text):
        return text

    def MultiAnnotation(self, texts):
        return texts

    def VisitFunction(self, stmt: ir.Function):
        self.tab_num += 1
        body = "\n".join([self.Visit(b) for b in stmt.body])
        self.tab_num -= 1
        self.id += 1
        return body

    def VisitBlockSplit(self, stmt: ir.BlockSplit):
        body = "\n\n".join([self.Visit(b) for b in stmt.body])
        return body

    def VisitBlock(self, stmt: ir.Block):
        body = "\n".join([self.Visit(b) for b in stmt.body])
        return body

    def VisitFor(self, stmt: ir.For):
        self.tab_num += 1
        body = "\n".join([self.Visit(b) for b in stmt.body])
        self.tab_num -= 1
        return body

    def VisitWhile(self, stmt: ir.While):
        if stmt.body:
            self.tab_num += 1
            body = "\n".join([self.Visit(b) for b in stmt.body])
            self.tab_num -= 1
            return body
        return ""



class CodeGenEngine:

    memo = {}

    @classmethod
    def Register(cls, target):
        def _register_task(codegen):
            cls.memo[target] = codegen
            return codegen
        return _register_task

    @classmethod
    def Get(cls, target):
        if target in cls.memo.keys():
            return cls.memo[target]
        msg = f"CodeGen Engine has no target of {target}, please check or register"
        raise RuntimeError(msg)