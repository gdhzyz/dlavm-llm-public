import copy
import numpy as np

class Op:

    add = {"py": " + ", "cpp": " + ", "inplace": "+"}
    sub = {"py": " - ", "cpp": " - ", "inplace": "-"}
    mul = {"py": " * ", "cpp": " * ", "inplace": "*"}
    div = {"py": " / ", "cpp": " / ", "inplace": "/"}
    mod = {"py": " % ", "cpp": " % "}
    equ = {"py": " == ", "cpp": " == "}
    gt = {"py": " > ", "cpp": " > "}
    ge = {"py": " >= ", "cpp": " >= "}
    lt = {"py": " < ", "cpp": " < "}
    le = {"py": " <= ", "cpp": " <= "}
    fdiv = {"py": " // ", "cpp": " / "}
    nequ = {"py": " != ", "cpp": " != "}
    lshift = {"py": " << ", "cpp": " << "}
    rshift = {"py": " >> ", "cpp": " >> "}


class Expr:

    def __init__(self, args, op):
        self.args = args
        self.op = op

    def get_vars(self, var=False):
        vars = []
        for arg in self.args:
            vars += arg.get_vars(var)
        return vars

    def eq(self, data):
        new_expr = data
        if isinstance(data, str):
            new_expr = Var(data)
        elif isinstance(data, (int, float)):
            new_expr = Numb(data)
        new_args = [self, new_expr]
        return Expr(new_args, Op.equ)

    def neq(self, data):
        new_expr = data
        if isinstance(data, str):
            new_expr = Var(data)
        elif isinstance(data, (int, float)):
            new_expr = Numb(data)
        new_args = [self, new_expr]
        return Expr(new_args, Op.nequ)

    def simplify(self, max_numb=0):
        new_args = [copy.deepcopy(arg).simplify(max_numb) for arg in self.args]
        new_expr = Expr(new_args, self.op)
        if isinstance(new_args[0], Numb) and isinstance(new_args[1], Numb):
            return Numb(eval(str(new_expr)))
        else:
            if self.op in [Op.add, Op.sub, Op.mul, Op.fdiv]:
                if isinstance(new_args[0], Numb):
                    if new_args[0].data == 0 and self.op in [Op.add, Op.sub]:
                        return new_args[1]
                    elif new_args[0].data == 1 and self.op in [Op.mul, Op.fdiv]:
                        return new_args[1]
                elif isinstance(new_args[1], Numb):
                    if new_args[1].data == 0 and self.op in [Op.add, Op.sub]:
                        return new_args[0]
                    elif new_args[1].data == 1 and self.op == Op.mul:
                        return new_args[0]
            return new_expr

    def __add__(self, data):
        new_expr = data
        if isinstance(data, str):
            new_expr = Var(data)
        elif isinstance(data, (int, float)):
            new_expr = Numb(data)
        new_args = [self, new_expr]
        return Expr(new_args, Op.add)

    def __sub__(self, data):
        new_expr = data
        if isinstance(data, str):
            new_expr = Var(data)
        elif isinstance(data, (int, float)):
            new_expr = Numb(data)
        new_args = [self, new_expr]
        return Expr(new_args, Op.sub)

    def __mul__(self, data):
        new_expr = data
        if isinstance(data, str):
            new_expr = Var(data)
        elif isinstance(data, (int, float)):
            new_expr = Numb(data)
        new_args = [self, new_expr]
        return Expr(new_args, Op.mul)

    def __truediv__(self, data):
        new_expr = data
        if isinstance(data, str):
            new_expr = Var(data)
        elif isinstance(data, (int, float)):
            new_expr = Numb(data)
        new_args = [self, new_expr]
        return Expr(new_args, Op.div)

    def __floordiv__(self, data):
        new_expr = data
        if isinstance(data, str):
            new_expr = Var(data)
        elif isinstance(data, (int, float)):
            new_expr = Numb(data)
        new_args = [self, new_expr]
        return Expr(new_args, Op.fdiv)

    def __mod__(self, data):
        new_expr = data
        if isinstance(data, str):
            new_expr = Var(data)
        elif isinstance(data, (int, float)):
            new_expr = Numb(data)
        new_args = [self, new_expr]
        return Expr(new_args, Op.mod)

    def __lshift__(self, data):
        new_expr = data
        if isinstance(data, str):
            new_expr = Var(data)
        elif isinstance(data, (int, float)):
            new_expr = Numb(data)
        new_args = [self, new_expr]
        return Expr(new_args, Op.lshift)

    def __rshift__(self, data):
        new_expr = data
        if isinstance(data, str):
            new_expr = Var(data)
        elif isinstance(data, (int, float)):
            new_expr = Numb(data)
        new_args = [self, new_expr]
        return Expr(new_args, Op.rshift)

    def __radd__(self, data):
        new_expr = data
        if isinstance(data, str):
            new_expr = Var(data)
        elif isinstance(data, (int, float)):
            new_expr = Numb(data)
        new_args = [new_expr, self]
        return Expr(new_args, Op.add)

    def __rsub__(self, data):
        new_expr = data
        if isinstance(data, str):
            new_expr = Var(data)
        elif isinstance(data, (int, float)):
            new_expr = Numb(data)
        new_args = [new_expr, self]
        return Expr(new_args, Op.sub)

    def __rmul__(self, data):
        new_expr = data
        if isinstance(data, str):
            new_expr = Var(data)
        elif isinstance(data, (int, float)):
            new_expr = Numb(data)
        new_args = [new_expr, self]
        return Expr(new_args, Op.mul)

    def __rtruediv__(self, data):
        new_expr = data
        if isinstance(data, str):
            new_expr = Var(data)
        elif isinstance(data, (int, float)):
            new_expr = Numb(data)
        new_args = [new_expr, self]
        return Expr(new_args, Op.div)

    def __rfloordiv__(self, data):
        new_expr = data
        if isinstance(data, str):
            new_expr = Var(data)
        elif isinstance(data, (int, float)):
            new_expr = Numb(data)
        new_args = [new_expr, self]
        return Expr(new_args, Op.fdiv)

    def __rmod__(self, data):
        new_expr = data
        if isinstance(data, str):
            new_expr = Var(data)
        elif isinstance(data, (int, float)):
            new_expr = Numb(data)
        new_args = [new_expr, self]
        return Expr(new_args, Op.mod)

    def __gt__(self, data):
        new_expr = data
        if isinstance(data, str):
            new_expr = Var(data)
        elif isinstance(data, (int, float)):
            new_expr = Numb(data)
        return Expr([self, new_expr], Op.gt)

    def __ge__(self, data):
        new_expr = data
        if isinstance(data, str):
            new_expr = Var(data)
        elif isinstance(data, (int, float)):
            new_expr = Numb(data)
        return Expr([self, new_expr], Op.ge)

    def __lt__(self, data):
        new_expr = data
        if isinstance(data, str):
            new_expr = Var(data)
        elif isinstance(data, (int, float)):
            new_expr = Numb(data)
        #return self.simplify(max_numb=0) < new_expr
        return Expr([self, new_expr], Op.lt)

    def __le__(self, data):
        new_expr = data
        if isinstance(data, str):
            new_expr = Var(data)
        elif isinstance(data, (int, float)):
            new_expr = Numb(data)
        #return self.simplify(max_numb=0) <= new_expr
        return Expr([self, new_expr], Op.le)

    def __eq__(self, new_expr):
        vars0 = self.get_vars()
        if isinstance(new_expr, int):
            new_expr = Numb(new_expr)
        vars1 = new_expr.get_vars()
        if vars0 != vars1:
            return False
        self_str = self.export("py")
        expr_str = new_expr.export("py")
        if self_str == expr_str:
            return True
        index = [0 for i in range(len(vars0))]
        while 1:
            self_numb = self_str
            expr_numb = expr_str
            for n in range(len(vars0)):
                self_numb = self_numb.replace(vars0[n][0], str(index[n]))
                expr_numb = expr_numb.replace(vars0[n][0], str(index[n]))
            if eval(self_numb) != eval(expr_numb):
                return False
            carry_over = 1
            for n in range(len(index))[::-1]:
                if index[n]+carry_over != vars0[n][1]:
                    carry_over = 0
                    index[n] += 1
                else:
                    index[n] = 0
            if carry_over:
                break
        return True
        '''
        if isinstance(new_expr, Expr):
            if self.op == new_expr.op and len(self.args) == len(new_expr.args):
                for i in range(len(self.args)):
                    if self.args[i] != new_expr.args[i]:
                        return False
                return True
        return False
        '''

    def export(self, tag):
        return f"({self.args[0].export(tag)}{self.op[tag]}{self.args[1].export(tag)})"

    def __str__(self):
        return self.export(tag="py")

    def __hash__(self) -> int:
        return hash(id(self))


class Var(Expr):

    expr_type = "var"

    def __init__(self, name, max_data=128, dtype="int"):
        if name is None:
            raise RuntimeError("The name of ne.Var should not be None, please check!")
        self.name = name
        self.max_data = max_data
        self.dtype = dtype

    def __eq__(self, new_expr):
        if isinstance(new_expr, Var):
            return self.name == new_expr.name
        return super().__eq__(new_expr)

    def get_vars(self, var=False):
        vars = [[self.name, self.max_data, self.dtype]]
        if var:
            vars = [Var(self.name, self.max_data, self.dtype)]
        return vars

    def simplify(self, max_numb=0):
        if max_numb:
            return Numb(self.max_data)
        else:
            return Var(self.name, self.max_data)

    def export(self, tag):
        return self.name

    def __hash__(self) -> int:
        return hash(id(self))


class Numb(Expr):

    def __init__(self, data):
        self.data = data

    def __hash__(self) -> int:
        return hash(self.data)

    def __eq__(self, new_expr):
        if isinstance(new_expr, Numb):
            return self.data == new_expr.data
        return super().__eq__(new_expr)

    def get_vars(self, var=False):
        vars = []
        return vars

    def simplify(self, max_numb=0):
        return Numb(self.data)

    def export(self, tag):
        return self.data
    
    def __str__(self):
        return str(self.data)

    def __gt__(self, data):
        if isinstance(data, str):
            return self.data > Var(data).max_data
        elif isinstance(data, (int, float)):
            return self.data > data
        else:
            return self.data > data.simplify(1).data

    def __ge__(self, data):
        if isinstance(data, str):
            return self.data >= Var(data).max_data
        elif isinstance(data, (int, float)):
            return self.data >= data
        else:
            return self.data >= data.simplify(1).data

    def __lt__(self, data):
        if isinstance(data, str):
            return self.data < Var(data).max_data
        elif isinstance(data, (int, float)):
            return self.data < data
        else:
            return self.data < data.simplify(1).data

    def __le__(self, data):
        if isinstance(data, str):
            return self.data <= Var(data).max_data
        elif isinstance(data, (int, float)):
            return self.data <= data
        else:
            return self.data <= data.simplify(1).data


class If(Expr):

    def __init__(self, judge_expr, then_expr, else_expr):
        self.judge_expr = judge_expr
        self.then_expr = then_expr
        self.else_expr = else_expr
        if isinstance(judge_expr, str):
            self.judge_expr = Var(judge_expr)
        elif isinstance(judge_expr, (int, float)):
            self.judge_expr = Numb(judge_expr)
        if isinstance(then_expr, str):
            self.then_expr = Var(then_expr)
        elif isinstance(then_expr, (int, float)):
            self.then_expr = Numb(then_expr)
        if isinstance(else_expr, str):
            self.else_expr = Var(else_expr)
        elif isinstance(else_expr, (int, float)):
            self.else_expr = Numb(else_expr)

    def get_vars(self, var=False):
        vars = []
        for arg in [self.judge_expr, self.then_expr, self.else_expr]:
            vars += arg.get_vars(var)
        return vars

    def simplify(self, max_numb=0):
        bool_expr = self.judge_expr
        if isinstance(self.then_expr, Expr):
            self.then_expr = self.then_expr.simplify(max_numb)
        if isinstance(self.else_expr, Expr):
            self.else_expr = self.else_expr.simplify(max_numb)
        if isinstance(self.judge_expr, Expr):
            self.judge_expr = self.judge_expr.simplify(max_numb)
            if isinstance(self.judge_expr, Numb):
                if self.judge_expr.data:
                    return self.then_expr
                else:
                    return self.else_expr
            else:
                return self
        if bool_expr:
            return self.then_expr
        else:
            return self.else_expr

    def export(self, tag):
        if tag == "cpp":
            return f"({self.judge_expr.export(tag)} ? {self.then_expr.export(tag)} : {self.else_expr.export(tag)})"
        elif tag == "py":
            return f"({self.then_expr.export(tag)} if {self.judge_expr.export(tag)} else {self.else_expr.export(tag)})"
        else:
            raise RuntimeError("No support target code")


if __name__ == "__main__":
    a = 12 * (Var("test") + 2)
    b = a.simplify()
    print(b.export("cpp"))
    print(a)
