from ...adr import Call, VM


class ConvertCall:

    def __init__(self, op_name: str, args: list, process):
        self.op_name = op_name
        self.args = args
        self.process = process

    def CheckFused(self, expr):
        if not isinstance(expr, (Call, VM)):
            return False, expr
        if expr.op.name != self.op_name:
            return False, expr
        new_args = [expr]
        for num, arg in enumerate(self.args):
            if isinstance(arg, ConvertCall):
                state, tp_arg = arg.CheckFused(expr.args[num])
                if not state:
                    return False, expr
                new_args.append(tp_arg)
            elif isinstance(arg, type(lambda x: x)):
                if not arg(expr.args[num].checked_type, expr.checked_type):
                    return False, expr
            elif isinstance(arg, list):
                if not arg[0](expr.args[num].checked_type, expr.checked_type):
                    return False, expr
                new_args.append(arg[1](expr.args[num]))
        return True, self.process(*new_args)


class FusedStrategy:

    _fused_map = {}

    @classmethod
    def Register(cls, op_name, strategy: ConvertCall, opt_level=1, reverse=False):
        if op_name in cls._fused_map.keys():
            cls._fused_map[op_name].append([strategy, opt_level])
            if reverse:
                new_args = [strategy.args[1], strategy.args[0]]
                new_strategy = ConvertCall(strategy.op_name, new_args, lambda z, x, y: strategy.process(z, y, x))
                cls._fused_map[op_name].append([new_strategy, opt_level])
        else:
            cls._fused_map[op_name] = [[strategy, opt_level]]

    @classmethod
    def Get(cls, op_name, opt_level=1):
        if op_name in cls._fused_map.keys():
            opt_pass = []
            for _fused in cls._fused_map[op_name]:
                if opt_level >= _fused[1]:
                    opt_pass.append(_fused[0])
            return opt_pass
        return None


class OpConvertStrategy:

    _convert_map = {}

    @classmethod
    def Register(cls, op_name, strategy: ConvertCall):
        if op_name in cls._convert_map.keys():
            cls._convert_map[op_name].append(strategy)
        else:
            cls._convert_map[op_name] = [strategy]

    @classmethod
    def Get(cls, op_name):
        if op_name in cls._convert_map.keys():
            return cls._convert_map[op_name]
        else:
            return None