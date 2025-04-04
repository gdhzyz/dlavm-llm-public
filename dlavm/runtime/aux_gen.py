from .base import RuntimeBase
from copy import deepcopy

from dlavm import ne
from dlavm.driver import ir

import math
import numpy as np


class AUXGen(RuntimeBase):

    def __init__(self, lib, device):
        super().__init__()
        self.device = device
        self.Visit(lib)
        self.aux_dat_wth = self.device.aux_dat_width

    def main(self, name, **kwargs):
        args = {}
        self.regs = []
        self.aux_task_ids = np.zeros([self.aux_dat_wth], dtype="uint8")
        func = self.funcs[name]
        for a in func.args:
            args[a] = kwargs[a.name]
        call_ir = ir.Call(func, args)
        return self.Visit(call_ir)

    def VisitCSBWrite(self, stmt: ir.CSB_Write):
        new_stmt = deepcopy(stmt)
        addr = self.Visit(new_stmt.addr)
        data = self.Visit(new_stmt.data)
        self.regs.append([0, addr, data])

    def VisitCSBRead(self, expr: ir.CSB_Read):
        new_expr = deepcopy(expr)
        addr = self.Visit(new_expr.addr)
        self.regs.append([1, addr])
        return 1

