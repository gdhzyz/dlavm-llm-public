from .base import RuntimeBase
from copy import deepcopy

from dlavm import ne
from dlavm.driver import ir, Tasks
from dlavm.backend import BuildModule

import math
import numpy as np


class AUXTask:

    def __init__(self, aux_dat_wth):
        self.dat_wth = aux_dat_wth
        self.init()

    def init(self):
        self.task_stg = 1 # 默认ids占用1*aux_dat_wth
        self.task_num = 0
        self.task_aux = 0
        self.task_ids = []
        self.task_reg = []

    def __len__(self):
        return len(self.task_ids)

    def add(self, task_cfg, task_reg):
        task_id, task_wth = task_cfg
        size = len(task_reg)
        task_reg = np.array(task_reg, dtype="uint32")
        tp_regs = np.zeros([task_wth*self.dat_wth], dtype="uint32")
        tp_regs[:size] = task_reg
        self.task_ids.append(task_id)
        self.task_reg.append(tp_regs)
        self.task_num += 1
        self.task_aux += task_wth

    def export(self):
        task_ids = np.array(self.task_ids, dtype="uint8")
        task_reg = np.array(self.task_reg, dtype="uint32")
        stg_size = self.task_stg * self.dat_wth
        task_bin = task_ids.tobytes() + task_reg.tobytes()
        task_num, task_aux = self.task_num, self.task_aux
        self.init()
        return task_bin, stg_size, task_num, task_aux


class AUXGen(RuntimeBase):

    prefix = "insts"

    def __init__(self, mod:BuildModule, device):
        super().__init__(mod.lib)
        self.mod = mod
        self.device = device
        self.aux_dat_wth = self.device.aux_dat_width
        self.aux_task = AUXTask(self.aux_dat_wth)
        self.task_bins = []
        self.aux_func_main = None
        self.aux_block_ids = 1

    def main(self, name, **kwargs):
        args = {}
        func = self.funcs[name]
        judge = 1
        for a in func.args:
            args[a] = kwargs[a.name]
            judge = a.eq(kwargs[a.name]) & judge
        self.regs = []
        self.aux_func_run = ir.Function([], name=name+"_aux_"+"_".join([str(i) for i in list(args.values())]))
        if self.aux_func_main is None:
            self.aux_func_main = ir.Function([], name=name+"_aux")
            self.aux_func_main.update_args(func.args)
        # self.aux_task_map = self.device.aux_task_map
        call_ir = ir.Call(func, args)
        self.Visit(call_ir)
        if len(self.aux_task):
            self.aux_finish()
        _if = ir.If(judge)
        with _if.then_block as _then:
            _then += ir.Call(self.aux_func_run)
        self.aux_func_main += _if

    def export(self, init_addr, addr_dtype="uint64_t"):
        self.mod.storage.set_address(init_addr)
        new_block = ir.Block()
        new_block.body = [self.mod.storage.export(addr_dtype)] + self.mod.lib.body[1:] + [self.aux_func_main]
        self.mod.lib = new_block
        return self.mod, self.task_bins

    def VisitCSBWrite(self, stmt: ir.CSB_Write):
        new_stmt = deepcopy(stmt)
        addr = self.Visit(new_stmt.addr)
        data = self.Visit(new_stmt.data)
        self.regs.append(data)

    def VisitCSBRead(self, expr: ir.CSB_Read):
        new_expr = deepcopy(expr)
        addr = self.Visit(new_expr.addr)
        # TODO: 1. task_id应该存在转换或字典查询，task_stg的值应可变
        #     : 2. max_task_num应可变
        task_id = self.regs[-1]
        self.regs.pop(-1)
        self.aux_task.add([task_id, 2], self.regs)
        self.regs = []
        if len(self.aux_task) == 16:
            self.aux_finish()
        return 1

    def aux_finish(self):
        task_bin, stg_size, task_num, task_aux = self.aux_task.export()
        self.task_bins.append(task_bin)
        storage_id = self.mod.storage.malloc(self.prefix, stg_size)
        storage_id = ir.Cast(ne.Var(storage_id, -1), "uint32_t")
        with ir.Function([], name=f"aux_block_{self.aux_block_ids}") as f:
            aux_func = Tasks.Get(f"atom.{self.device.name}.aux", self.device)
            aux_func(f, storage_id, task_aux, task_num)
        self.aux_func_run += ir.Call(f)
        self.aux_block_ids += 1
