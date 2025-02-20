from dlavm import ne
from dlavm.driver import ir, transform
from . import targets
from .codegen_base import CodeGenBase, CodeGenEngine


@CodeGenEngine.Register(targets.hpp)
class CodeGenH(CodeGenBase):

    data_type = {
        "int32": "int",
        "int64": "int64_t",
    }

    def __init__(self) -> None:
        super().__init__()

    def main(self, stmt: ir.Function):
        stmt = transform.InferArgs(handle=True).Visit(stmt)
        self._memo_lib = []
        source = self.Visit(stmt)
        return source

    def BriefAnnotation(self, text):
        return f"// {text}"

    def VisitData(self, data):
        return str(data)

    def VisitOp(self, expr: ir.Op):
        arg0 = self.Visit(expr.arg0)
        arg1 = self.Visit(expr.arg1)
        return arg0 + expr.op_type["cpp"] + arg1

    def VisitBlockSplit(self, stmt: ir.BlockSplit):
        body = super().VisitBlockSplit(stmt)
        if stmt.namespace is not None:
            body = f"namespace {str(stmt.namespace)}" + " {\n" + body + "\n};"
        return body

    def VisitFunction(self, stmt: ir.Function):
        self.lib = []
        body = super().VisitFunction(stmt)
        def_args = ", ".join([f"{i.dtype} {i.name}" for i in stmt.args])
        func_name = "temp"
        if stmt.name is not None:
            func_name = stmt.name
        tabs = self.tab * self.tab_num
        ret = stmt.ret
        source = tabs + f"{ret} {func_name}({def_args})" + " {\n"
        source += body + "\n"
        source += tabs + "}"
        if len(self.lib):
            lib_src = [self.Visit(l) for l in self.lib]
            source = "\n".join(lib_src) + "\n\n" + source
        return source

    def VisitCall(self, stmt: ir.Call):
        if isinstance(stmt.func, ir.Function):
            if stmt.func.name not in self._memo_lib:
                self.lib.append(stmt.func)
                self._memo_lib.append(stmt.func.name)
            def_args = ", ".join([f"{i.name}" for i in stmt.args.values()])
            tabs = self.tab * self.tab_num
            if stmt.ret is None:
                source = tabs + f"{stmt.func.name}({def_args});"
            else:
                source = tabs + f"{stmt.ret} = {stmt.func.name}({def_args});"
            return source
        else:
            print("Warning! CodegenH Visit Call!")
            print(stmt)
            return ""

    def VisitReturn(self, stmt: ir.Return):
        data = self.Visit(stmt.data)
        tabs = self.tab * self.tab_num
        source = tabs + f"return {data};"
        return source

    def VisitIf(self, stmt: ir.If):
        tabs = self.tab * self.tab_num
        judge = self.Visit(stmt.judge)
        self.tab_num += 1
        then_block = [self.Visit(b) for b in stmt.then_block.body]
        else_block = [self.Visit(b) for b in stmt.else_block.body]
        self.tab_num -= 1
        if len(then_block) == 1:
            then_src = then_block[0][(self.tab_num+1)*len(self.tab):]
            source = tabs + f"if ({judge}) {then_src}"
        else:
            source = tabs + f"if ({judge})" + " {\n"
            source += "\n".join(then_block) + "\n"
            source += tabs + "}"
        if len(else_block) == 1:
            source += f" else {else_block[0]}"
        elif len(else_block) == 0:
            pass
        else:
            source += " else {\n"
            source += "\n".join(else_block) + "\n"
            source += tabs + "}"
        return source

    def VisitFor(self, stmt: ir.For):
        body = super().VisitFor(stmt)
        var, dtype = stmt.var.name, stmt.var.dtype
        tabs = self.tab * self.tab_num
        init, extent, stride = self.Visit(stmt.init), self.Visit(stmt.extent), self.Visit(stmt.stride)
        source = tabs + f"for ({dtype} {var} = {init}; {var} < {extent}; {var} += {stride}) " + "{\n"
        source += body + "\n"
        source += tabs + "}"
        return source

    def VisitWhile(self, stmt: ir.While):
        body = super().VisitWhile(stmt)
        judge = self.Visit(stmt.judge)
        tabs = self.tab * self.tab_num
        source = tabs + f"while ({judge}) "
        if len(body):
            source += "{\n" + body + "\n"
            source += tabs + "}"
        else:
            source += "{}"
        return source

    def VisitAnnotation(self, stmt: ir.Annotation):
        tabs = self.tab * self.tab_num
        return tabs + f"// {stmt.text}"

    def VisitAssign(self, stmt: ir.Assign):
        tabs = self.tab * self.tab_num
        value = self.Visit(stmt.value)
        dtype = stmt.dtype
        if stmt.dtype in self.data_type.keys():
            dtype = self.data_type[stmt.dtype]
        return tabs + f"{dtype} {stmt.var.name} = {value};"

    def VisitCSBWrite(self, stmt: ir.CSB_Write):
        tabs = self.tab * self.tab_num
        addr = self.Visit(stmt.addr)
        data = self.Visit(stmt.data)
        return tabs + f"CSB_Write(device, {addr}, {data});"

    def VisitCSBRead(self, expr: ir.CSB_Read):
        addr = self.Visit(expr.addr)
        return f"CSB_Read(device, {addr})"

    def VisitMemWrite(self, stmt: ir.MemWrite):
        self.h2cx = True
        tabs = self.tab * self.tab_num
        addr = self.Visit(stmt.addr)
        data = self.Visit(stmt.data)
        return tabs + f"DDR_Update(h2cx, {addr}, {data});"

    def VisitMemWriteFile(self, stmt: ir.MemWriteFile):
        self.h2cx = True
        tabs = self.tab * self.tab_num
        addr = self.Visit(stmt.addr)
        size = self.Visit(stmt.size)
        return tabs + f"DDR_Write_bin(h2cx, {stmt.file}, {addr}, {size});"

    def VisitMemInit(self, stmt: ir.MemInit):
        self.h2cx = True
        tabs = self.tab * self.tab_num
        addr = self.Visit(stmt.addr)
        size = self.Visit(stmt.size)
        return tabs + f"init(h2cx, {addr}, {size});"

    def VisitStrFormat(self, stmt: ir.StrFormat):
        tabs = self.tab * self.tab_num
        str_args = ", ".join([str(self.Visit(i)) for i in stmt.args])
        source = tabs + f"char {stmt.var.name}[100];\n"
        source += tabs + f"sprintf({stmt.var.name}, \"{stmt.target}\", {str_args});"
        return source

    def VisitInplace(self, stmt: ir.Inplace):
        tabs = self.tab * self.tab_num
        op = stmt.op["inplace"]
        data = self.Visit(stmt.data)
        return tabs + f"{stmt.var.name} {op}= {data};"

    def VisitCast(self, expr: ir.Cast):
        var = self.Visit(expr.var)
        return f"(({expr.dtype}){var})"

    def VisitVar(self, expr: ir.Var):
        var = self.Visit(expr.var)
        return f"{var}"

    def VisitNe(self, expr: ne.Expr):
        return expr.simplify().export("cpp")
