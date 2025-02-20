from typing import Optional, Union, List, Dict
from dlavm import ne


Empty = "IR.Empty"


class IR:

    ir_type = "Base"

    def __init__(self) -> None:
        pass


class Expr(IR):

    ir_type = "Expr"

    def __init__(self) -> None:
        super().__init__()

    def __hash__(self):
        return hash(id(self))

    def __ne__(self, expr):
        return Op(self, expr, ne.Op.nequ)


class Stmt(IR):

    ir_type = "Stmt"

    def __init__(self) -> None:
        super().__init__()

    def _str_tab(self, tab_num):
        pass

    def __str__(self) -> str:
        return self._str_tab(tab_num=0)


class Var(Expr):

    def __init__(
            self,
            var: Union[ne.Expr],
            stmt: Union[Stmt]
        ) -> None:
        super().__init__()
        self.var = var
        self.stmt = stmt

    def __str__(self):
        if isinstance(self.var, ne.Expr):
            return f"({self.dtype}){self.var.simplify()}"
        return f"({self.dtype}){self.var}"


class Op(Expr):

    def __init__(
            self,
            arg0: Union[ne.Expr, Expr, int, float],
            arg1: Union[ne.Expr, Expr, int, float],
            op_type: Union[str],
        ) -> None:
        super().__init__()
        self.arg0 = arg0
        self.arg1 = arg1
        self.op_type = op_type

    def __str__(self):
        _str = ""
        if isinstance(self.arg0, ne.Expr):
            _str += self.arg0.simplify()
        else:
            _str += str(self.arg0)
        _str += self.op_type["py"]
        if isinstance(self.arg1, ne.Expr):
            _str += self.arg1.simplify()
        else:
            _str += str(self.arg1)
        return _str


class Cast(Expr):

    def __init__(
            self,
            var: Union[ne.Expr, Expr, int, float],
            dtype: Union[str]
        ) -> None:
        super().__init__()
        self.var = var
        self.dtype = dtype

    def __str__(self):
        if isinstance(self.var, ne.Expr):
            return f"({self.dtype}){self.var.simplify()}"
        return f"({self.dtype}){self.var}"


class Inplace(Stmt):

    def __init__(
            self,
            var: Union[ne.Var, Expr],
            op: Union[Dict[str, str]], # type of ne.Op.xxx, example: ne.Op.add
            data: Union[ne.Expr, Expr, int, float]
    ):
        super().__init__()
        self.var = var
        self.op = op
        self.data = data

    def _str_tab(self, tab_num):
        op = self.op["inplace"]
        return " "*tab_num + f"{self.var.name} {op}= {self.data}"


class Annotation(Stmt):

    def __init__(
            self,
            text: Union[str],
        ):
        super().__init__()
        self.text = text

    def _str_tab(self, tab_num):
        return " "*tab_num + f"# {self.text}"


class Assign(Stmt):

    def __init__(
            self,
            name: Union[str],
            value: Union[ne.Expr, Expr, int, float, str],
            dtype: Union[str]="int32"
        ):
        super().__init__()
        self.value = value
        self.dtype = dtype
        self.var = ne.Var(name, -1, dtype)

    def _str_tab(self, tab_num):
        return " "*tab_num + f"{self.dtype} {self.var.name} = {self.value}"


class StrFormat(Stmt):

    def __init__(
            self,
            name: Union[str],
            target: Union[ne.Expr, Expr, str],
            *args
    ) -> None:
        super().__init__()
        self.var = ne.Var(name, -1, "string")
        self.target = target
        self.args = args

    def _str_tab(self, tab_num):
        str_args = ", ".join([str(i) for i in self.args])
        return " "*tab_num + f"{self.var.dtype} {self.var.name} = {self.target}%({str_args})" 


class Block(Stmt):

    def __init__(self) -> None:
        super().__init__()
        self.body = []

    def assign(self, name, value, dtype):
        var = Assign(name, value, dtype)
        self.body.append(var)
        return var.var

    def __iadd__(self, stmt):
        self.body.append(stmt)
        return self

    def __enter__(self):
        return self

    def __exit__(self, type, value, trace):
        self.body = tuple(self.body)
        pass

    def __getitem__(self, stmt: Stmt):
        self.body.append(stmt)
        return stmt

    def _str_tab(self, tab_num):
        str_body = "\n".join([i._str_tab(tab_num) for i in self.body])
        return str_body


class BlockSplit(Block):

    def __init__(
            self,
            namespace: Optional[str]=None,
        ):
        super().__init__()
        self.namespace = namespace

    def _str_tab(self, tab_num):
        str_body = "\n\n".join([i._str_tab(tab_num) for i in self.body])
        return str_body


class Function(Block):

    def __init__(
            self,
            args: Union[List[ne.Expr], List[Expr]],
            ret : Optional[str]="void",
            name: Optional[str]=None
        ):
        super().__init__()
        self.args = args
        self.ret  = ret
        self.name = name

    def update_args(
            self,
            args: Union[List[ne.Expr], List[Expr]],
    ):
        for arg in args:
            if arg not in self.args:
                self.args.append(arg)

    def _str_tab(self, tab_num):
        name = "func"
        if hasattr(self, "name"):
            name = self.name
        vars = ", ".join([f"{i.dtype} {i.name}: {i.max_data}" for i in self.args])
        strs = [" "*tab_num + f"def {name}({vars})" + " {"]
        strs += [i._str_tab(tab_num+1) for i in self.body]
        strs += [" "*tab_num + "}"]
        return "\n".join(strs)


class For(Block):

    def __init__(
            self,
            var_name: Union[str],
            init: Union[ne.Expr, Expr, int, float],
            extent: Union[ne.Expr, Expr, int, float],
            stride: Union[ne.Expr, Expr, int, float]=1,
            var_dtype: Union[str]="int",
            max_data: Union[int, float, None]=None
        ) -> None:
        super().__init__()
        if max_data is None:
            max_data=extent
        self.var = ne.Var(var_name, max_data=max_data, dtype=var_dtype)
        self.init = init
        self.extent = extent
        self.stride = stride

    def _str_tab(self, tab_num):
        max_data = self.extent
        strs = [" "*tab_num + f"for {str(self.var)} in range({self.init}, {max_data}, {self.stride})" + " {"]
        strs += [i._str_tab(tab_num+1) for i in self.body]
        strs += [" "*tab_num + "}"]
        return "\n".join(strs)


class While(Block):

    def __init__(
            self, 
            judge: Union[ne.Expr, Expr, int, bool]
        ) -> None:
        super().__init__()
        self.judge = judge

    def _str_tab(self, tab_num):
        strs = [" "*tab_num + f"while {str(self.judge)}" + " {"]
        if len(self.body):
            strs += [i._str_tab(tab_num+1) for i in self.body]
        strs += [" "*tab_num + "}"]
        return "\n".join(strs)


class If(Stmt):
    
    def __init__(
            self,
            judge: Union[ne.Expr, Expr, int, bool]
        ) -> None:
        super().__init__()
        self.judge = judge
        self.then_block = Block()
        self.else_block = Block()

    def __enter__(self):
        return self

    def __exit__(self, type, value, trace):
        pass

    def _str_tab(self, tab_num):
        strs = [" "*tab_num + f"if ({str(self.judge)})" + " {"]
        strs += [i._str_tab(tab_num+1) for i in self.then_block.body]
        strs += [" "*tab_num + "} else {"]
        strs += [i._str_tab(tab_num+1) for i in self.else_block.body]
        strs += [" "*tab_num + "}"]
        return "\n".join(strs)


class Call(Stmt):

    def __init__(
            self,
            func: Union[Function, str],
            args: Dict[ne.Expr, ne.Expr]=None,
            ret: Optional[str]=None
        ):
        super().__init__()
        self.func = func
        self.ret = ret
        if args is None:
            args = dict(zip(func.args, func.args))
        self.args = args

    def _str_tab(self, tab_num):
        str_args = ", ".join([str(arg) for arg in self.func.args])
        return " "*tab_num + f"{self.func.name}({str_args})"


class Return(Stmt):

    def __init__(
            self,
            data: Union[Expr, ne.Expr, int, float, str],
        ):
        super().__init__()
        self.data = data

    def _str_tab(self, tab_num):
        return " "*tab_num + f"return {self.data}"


'''
special accel lib function
'''

class CSB_Write(Stmt):

    def __init__(
            self,
            addr: Union[ne.Expr, Expr, int],
            data: Union[ne.Expr, Expr, int],
        ) -> None:
        super().__init__()
        self.addr = addr
        self.data = data

    def _str_tab(self, tab_num):
        if isinstance(self.data, ne.Expr):
            return " "*tab_num + f"CSB_Write({self.addr}, {self.data.simplify()})"
        else:
            return " "*tab_num + f"CSB_Write({self.addr}, {self.data})"


class CSB_Read(Expr):

    def __init__(
            self,
            addr: Union[ne.Expr, Expr, int],
        ) -> None:
        super().__init__()
        self.addr = addr

    def __str__(self):
        return f"CSB_Read({self.addr})"


class MemWrite(Stmt):

    def __init__(
            self,
            addr: Union[ne.Expr, Expr, int],
            data: Union[ne.Expr, Expr, str],
        ) -> None:
        super().__init__()
        self.addr = addr
        self.data = data

    def _str_tab(self, tab_num):
        return " "*tab_num + f"Mem_Write({self.addr}, {self.data})"


class MemWriteFile(Stmt):

    def __init__(
            self,
            addr: Union[ne.Expr, Expr, int],
            file: Union[ne.Expr, Expr, str],
            size: Union[ne.Expr, Expr, int],
        ) -> None:
        super().__init__()
        self.addr = addr
        self.file = file
        self.size = size

    def _str_tab(self, tab_num):
        return " "*tab_num + f"Mem_Write_bin({self.addr}, {self.file}, {self.size})"


class MemInit(Stmt):

    def __init__(
            self,
            addr: Union[ne.Expr, Expr, int],
            size: Union[ne.Expr, Expr, int],
        ) -> None:
        super().__init__()
        self.addr = addr
        self.size = size

    def _str_tab(self, tab_num):
        return " "*tab_num + f"Mem_Init({self.addr}, {self.size})"

