from ..adr import Functor, Call, VM, Tensor, Function


class InferType(Functor):

    def __init__(self, device=None, attrs={}):
        super(InferType, self).__init__()
        self.device = device
        self.attrs = attrs
        self.glb_const = {}

    def set_attrs(self, attrs):
        for key, value in self.attrs.items():
            if attrs.get(key) is None:
                attrs[key] = value
        return attrs

    def set_prefix(self, name):
        if "::" in name:
            return name.split("::")
        else:
            return None
    
    def visit_function(self, expr):
        new_args = [self.visit(arg) for arg in expr.args]
        new_expr = self.visit(expr.expr)
        return Function(expr.name, new_args, new_expr)

    def visit_var(self, expr):
        device = expr.device
        if self.device is not None:
            device = self.device
        expr.checked_type = Tensor(expr.shape, expr.dtype, device)
        return expr

    def visit_constant(self, expr):
        if expr.dtype is None:
            return expr
        device = expr.device
        if self.device is not None:
            device = self.device
        expr.checked_type = Tensor(expr.shape, expr.dtype, device)
        prefix = self.set_prefix(expr.name)
        if prefix is not None:
            expr.prefix = prefix[0]
            expr.name = prefix[1]
        return expr

    def visit_tupleitem(self, expr):
        arg = self.visit(expr.expr)
        expr.expr = arg
        expr.checked_type = arg.checked_type.tensors[expr.index]
        return expr

    def visit_call(self, expr):
        new_attrs = self.set_attrs(expr.attrs)
        new_args = [self.visit(arg) for arg in expr.args]
        new_type = [arg.get_checked_type() for arg in new_args]
        func = expr.op.attrs["rel"]
        state, checked_type = func(new_type, new_attrs)
        if state:
            new_expr = Call(expr.op, new_args, new_attrs, expr.prefix, checked_type)
            return new_expr
        else:
            raise RuntimeError("Check Error! " + expr.op.name + ", " + checked_type)

    def visit_vm(self, expr):
        new_attrs = self.set_attrs(expr.attrs)
        new_args = [self.visit(arg) for arg in expr.args]
        new_type = [arg.get_checked_type() for arg in new_args]
        func = expr.op.attrs["rel"]
        state, checked_type = func(new_type, new_attrs)
        if state:
            new_expr = VM(expr.op, new_args, new_attrs, checked_type)
            return new_expr
        else:
            raise RuntimeError("Check Error! " + expr.op.name + ", " + checked_type)


def infer_type(expr, device=None, attrs={}):
    expr = InferType(device, attrs).visit(expr)
    return expr
