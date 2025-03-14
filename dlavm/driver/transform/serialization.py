from copy import deepcopy
from dlavm import ne
from .. import ir


class Serialization(ir.Functor):

    def main(self, stmt):
        self.tp_list = []
        self.mems = {}
        self.Visit(stmt)
        return self.mems

    def VisitCSBWrite(self, stmt: ir.CSB_Write):
        new_stmt = super().VisitCSBWrite(stmt)
        self.tp_list.append([0, new_stmt.addr, new_stmt.data])
        return new_stmt

    def VisitCSBRead(self, expr: ir.CSB_Read):
        new_expr = super().VisitCSBRead(expr)
        self.tp_list.append([1, new_expr.addr])
        return new_expr

    def VisitFunction(self, stmt: ir.Function):
        new_stmt = super().VisitFunction(stmt)
        self.name = stmt.name if stmt.name is not None else self.get_tp_name()
        self.mems[self.name] = self.tp_list
        self.tp_list = []
        return new_stmt

    def get_tp_name(self):
        if not hasattr(self, "_id"):
            self._id = 0
        name = f"tp_{self._id}"
        self._id += 1
        return name
