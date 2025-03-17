from copy import deepcopy

from dlavm import ne
from dlavm.driver import ir

import math


class ExecutorBase(ir.Functor):

    def __init__(self):
        """
        TODO:
            作用域的控制和使用极为松散，仅考虑了function存在为作用域的情况，其他情况均未考虑
        """
        super().__init__()
        self.rt_vars = {}
        self.tp_vars = {}
        self.in_func = False
        self.funcs = {}

    def Visit(self, ir_: ir.IR):
        """
        enable multi-visit for one node
        """
        if isinstance(ir_, ir.Stmt):
            result = self.VisitStmt(ir_)
        elif isinstance(ir_, ir.Expr):
            result = self.VisitExpr(ir_)
        elif isinstance(ir_, ne.Expr):
            result = self.VisitNe(ir_)
        elif isinstance(ir_, (int, float, str)):
            result = self.VisitData(ir_)
        else:
            msg = f"CodeGen not support type of {type(ir_)}, needs driver.ir"
            raise RuntimeError(msg)
        return result

    def VisitOp(self, expr: ir.Op):
        new_expr = super().VisitOp(expr)
        return eval(str(new_expr))

    def VisitNe(self, expr: ne.Expr):
        new_expr = ne.expr_var_from_dict(expr, {**self.rt_vars, **self.tp_vars}).simplify()
        if isinstance(new_expr, ne.Numb):
            return new_expr.data
        else:
            print(new_expr)
            vars = new_expr.get_vars()
            print(vars)
            raise RuntimeError("no found [] when compute ne, please check!")

    def VisitData(self, data):
        if isinstance(data, str) and "0x" in data:
            data = int(data, 16)
        return data

    def VisitFunction(self, stmt: ir.Function):
        if self.in_func:
            new_stmt = deepcopy(stmt)
            _tp = [self.Visit(b) for b in new_stmt.body]
            if isinstance(new_stmt.body[-1], ir.Return):
                return _tp[-1]
        else:
            self.funcs[stmt.name] = stmt

    def VisitBlock(self, stmt: ir.Block):
        new_stmt = deepcopy(stmt)
        [self.Visit(b) for b in new_stmt.body]

    def VisitBlockSplit(self, stmt: ir.BlockSplit):
        new_stmt = deepcopy(stmt)
        [self.Visit(b) for b in new_stmt.body]

    def VisitCall(self, stmt: ir.Call):
        new_stmt = deepcopy(stmt)
        _vars = {}
        for k, v in new_stmt.args.items():
            _vars[k.name] = self.Visit(v)
        backup_vars = self.tp_vars
        self.tp_vars = _vars
        self.in_func = True
        ret = self.Visit(new_stmt.func)
        self.tp_vars = backup_vars
        if stmt.ret is not None:
            self.tp_vars[stmt.ret] = ret
        self.in_func = False
        return ret

    def VisitExternCall(self, stmt: ir.ExternCall):
        new_stmt = deepcopy(stmt)
        args = [self.Visit(b) for b in new_stmt.args]
        _vars = {**self.rt_vars, **self.tp_vars}
        str_args = ", ".join([str(a) for a in args])
        call_func = f"{new_stmt.name}({str_args})"
        ret = eval(call_func)
        if stmt.ret is not None:
            self.tp_vars[stmt.ret] = ret

    def VisitReturn(self, stmt: ir.Return):
        new_stmt = deepcopy(stmt)
        return self.Visit(new_stmt.data)

    def VisitFor(self, stmt: ir.For):
        new_stmt = deepcopy(stmt)
        init = self.Visit(new_stmt.init)
        extent = self.Visit(new_stmt.extent)
        stride = self.Visit(new_stmt.stride)
        for i in range(init, extent, stride):
            self.tp_vars[new_stmt.var.name] = i
            [self.Visit(b) for b in new_stmt.body]

    def VisitWhile(self, stmt: ir.While):
        new_stmt = deepcopy(stmt)
        new_stmt.judge = self.Visit(stmt.judge)
        while self.Visit(new_stmt.judge):
            [self.Visit(b) for b in new_stmt.body]

    def VisitIf(self, stmt: ir.If):
        new_stmt = deepcopy(stmt)
        new_stmt.then_block = deepcopy(stmt.then_block)
        new_stmt.else_block = deepcopy(stmt.else_block)
        if self.Visit(new_stmt.judge):
            [self.Visit(b) for b in new_stmt.then_block.body]
        else:
            [self.Visit(b) for b in new_stmt.else_block.body]

    def VisitAnnotation(self, stmt: ir.Annotation):
        pass

    def VisitAssign(self, stmt: ir.Assign):
        new_stmt = deepcopy(stmt)
        if self.in_func:
            self.tp_vars[new_stmt.var.name] = self.Visit(new_stmt.value)
        else:
            self.rt_vars[new_stmt.var.name] = self.Visit(new_stmt.value)

    def VisitAssignVar(self, stmt: ir.AssignVar):
        new_stmt = deepcopy(stmt)
        if self.in_func and new_stmt.var in self.tp_vars.keys():
            self.tp_vars[new_stmt.var.name] = self.Visit(new_stmt.value)
        else:
            self.rt_vars[new_stmt.var.name] = self.Visit(new_stmt.value)

    def VisitCSBWrite(self, stmt: ir.CSB_Write):
        raise RuntimeError("no realize CSB Write node!")

    def VisitCSBRead(self, expr: ir.CSB_Read):
        # new_expr = deepcopy(expr)
        # pass
        raise RuntimeError("no realize CSB Read node!")

    def VisitMemWriteFile(self, stmt: ir.MemWriteFile):
        new_stmt = deepcopy(stmt)
        pass

    def VisitMemWrite(self, stmt: ir.MemWrite):
        new_stmt = deepcopy(stmt)
        pass

    def VisitMemInit(self, stmt: ir.MemInit):
        new_stmt = deepcopy(stmt)
        pass

    def VisitStrFormat(self, stmt: ir.StrFormat):
        new_stmt = deepcopy(stmt)
        pass

    def VisitInplace(self, stmt: ir.Inplace):
        new_stmt = deepcopy(stmt)
        op = new_stmt.op["py"]
        if in_func and new_stmt.var in self.tp_vars.keys():
            self.tp_vars[new_stmt.var.name] = eval(f"{self.tp_vars[new_stmt.var.name]} {op} {self.Visit(new_stmt.data)}")
        else:
            self.rt_vars[new_stmt.var.name] = eval(f"{self.rt_vars[new_stmt.var.name]} {op} {self.Visit(new_stmt.data)}")

    def VisitCast(self, expr: ir.Cast):
        new_expr = deepcopy(expr)
        var = self.Visit(new_expr.var)
        if "int" in new_expr.dtype:
            return int(var)
        else:
            return var

    def VisitVar(self, expr: ir.Var):
        new_expr = deepcopy(expr)
        pass

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class RuntimeBase(ExecutorBase):

    def __init__(self, lib):
        super().__init__()
        self.lib = lib
        self.Visit(self.lib)

    def main(self, name, **kwargs):
        args = {}
        func = self.funcs[name]
        for a in func.args:
            args[a] = kwargs[a.name]
        call_ir = ir.Call(func, args)
        return self.Visit(call_ir)

    def VisitCSBWrite(self, stmt: ir.CSB_Write):
        new_stmt = deepcopy(stmt)
        addr = self.Visit(new_stmt.addr)
        data = self.Visit(new_stmt.data)
        print(addr, data)

    def VisitCSBRead(self, expr: ir.CSB_Read):
        return 1
