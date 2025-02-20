from .. import ir


class Sequence:

    def __init__(self, opt_pass:list):
        self.opt_pass = opt_pass

    def __call__(self, stmt: ir.Function) -> ir.Function:
        for opt in self.opt_pass:
            stmt = opt.Visit(stmt)
        return stmt