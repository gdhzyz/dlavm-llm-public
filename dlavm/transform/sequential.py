from .infer_type import infer_type


class Sequential:

    def __init__(self, pass_opt_list: list):
        self.transforms = pass_opt_list

    def __call__(self, expr):
        expr = infer_type(expr)
        for t in self.transforms:
            expr = t.visit(expr)
            expr = infer_type(expr)
        return expr
