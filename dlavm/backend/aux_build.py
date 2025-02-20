import numpy as np
from ..adr import Functor, VM, Tensor, Tuple, DataEnum
from .graph_build import GraphBuild 
from .regs_build import RegsBuild
from dlavm.driver import Tasks, ir, transform
from dlavm import ne


class CsbUpdate(ir.Functor):

    def __init__(self):
        super().__init__()
    
    def main(self, stmt, storage_id, size, kvcache_var):
        self.storage_id = ne.Var(storage_id, -1)
        self.numb = 0
        self.kvcache_var = kvcache_var
        self.assign_vars = {}
        self.insts = np.zeros([size], dtype="int32")
        new_stmt = self.Visit(stmt)
        return new_stmt, self.insts

    def VisitFunction(self, stmt: ir.Function):
        self.vars = {}
        self.upt_full = ne.Var("upt_full", 1)
        new_args = [ne.Var("upt_"+arg.name, arg.max_data, arg.dtype) for arg in stmt.args]
        self.dict_args = dict(zip([arg.name for arg in stmt.args], new_args))
        new_stmt = super().VisitFunction(stmt)
        new_stmt.name = "update_" + new_stmt.name
        new_stmt.args = new_args + [self.upt_full]
        return new_stmt
    
    def VisitAssign(self, stmt: ir.Assign):
        new_stmt = super().VisitAssign(stmt)
        if isinstance(new_stmt.value, ne.Expr):
            self.assign_vars[new_stmt.var.name] = new_stmt
        return new_stmt

    def VisitWhile(self, stmt: ir.While):
        if isinstance(stmt.judge, ir.Op) and isinstance(stmt.judge.arg0, ir.CSB_Read):
            return ir.Empty
        return super().VisitWhile(stmt)

    def VisitCSBWrite(self, stmt: ir.CSB_Write):
        if isinstance(stmt.data, ne.Expr):
            vars = stmt.data.get_vars(True)
            def _wrap(var):
                if isinstance(var, list):
                    results = []
                    for v in var:
                        results += _wrap(v)
                    return results
                if var.name in self.vars.keys():
                    results = []
                    for new_var in self.vars[var.name].value.get_vars(True):
                        results += _wrap(new_var)
                    return results
                elif var.name in self.dict_args.keys():
                    return [var.name]
                else:
                    return []
            arg_vars = _wrap(vars)
            stmt = super().VisitCSBWrite(stmt)
            if len(arg_vars) == 1 and arg_vars[0] == self.kvcache_var.name:
                with ir.If(self.upt_full) as _if:
                    _if.then_block += ir.MemWrite(self.storage_id + 4*self.numb, stmt.data)
                new_stmt = _if
            elif len(arg_vars) == 0:
                new_stmt = ir.Empty
            else:
                new_stmt = ir.MemWrite(self.storage_id + 4*self.numb, stmt.data)
        else:
            new_stmt = ir.Empty
            self.insts[self.numb] = stmt.data
        self.numb += 1
        return new_stmt

    def VisitNe(self, expr: ne.Expr):
        new_expr = ne.expr_var_from_dict(expr, self.dict_args).simplify()
        if isinstance(new_expr, ne.Numb):
            new_expr = new_expr.data
        return new_expr


class DynamicTasks(ir.Functor):

    def main(self, stmt: ir.Function):
        self.task_cnt = 0
        self.tasks = 0
        self.vars = {}
        new_stmt = self.Visit(stmt)
        if isinstance(self.tasks, int):
            return self.tasks, True, None
        elif isinstance(self.tasks, ne.Expr):
            self.tasks = self.tasks.simplify()
            used_vars = []
            def _wrap(var):
                used_vars.append(var.name)
                if var.name in self.vars.keys():
                    for new_var in self.vars[var.name].value.get_vars(True):
                        _wrap(new_var)
            task_vars = self.tasks.get_vars(True)
            for var in task_vars:
                _wrap(var)
            args = []
            for arg in new_stmt.args:
                if arg.name in used_vars:
                    args.append(arg)
            f = ir.Function(args)
            for var, assign in self.vars.items():
                if var in used_vars:
                    f += assign
            tasks = f[ir.Assign("tasks", self.tasks)]
            return tasks.var, False, f
        else:
            raise RuntimeError("What?????????")

    def VisitWhile(self, stmt: ir.While):
        if isinstance(stmt.judge, ir.Op) and isinstance(stmt.judge.arg0, ir.CSB_Read):
            self.task_cnt += 1
        return stmt
    
    def VisitAssign(self, stmt: ir.Assign):
        new_stmt = super().VisitAssign(stmt)
        if isinstance(new_stmt.value, ne.Expr):
            self.vars[new_stmt.var.name] = new_stmt
        return new_stmt

    def VisitFor(self, stmt: ir.For):
        beg_task = self.task_cnt
        new_stmt = super().VisitFor(stmt)
        for_task = self.task_cnt - beg_task
        self.tasks = self.tasks + for_task * (new_stmt.extent + new_stmt.init) // new_stmt.stride
        return new_stmt


class AuxBuild(RegsBuild, GraphBuild):

    def aux_task_append(self, func, task_id):
        if self.aux_task_numb == 0:
            self.aux_task_ids = np.zeros([self.aux_dat_wth], dtype="uint8")
            self.aux_storage = self.storage.malloc("insts", self.aux_dat_wth)
        task_numb, static, aux_stmt = DynamicTasks().main(func)
        if static:
            storage_id = self.storage.malloc("insts", self.aux_dat_wth*task_id[1])
        else:
            self.aux_task_finish()
            self.aux_task_ids = np.zeros([self.aux_dat_wth], dtype="uint8")
            self.aux_storage = self.storage.malloc("insts", self.aux_dat_wth)
            storage_id = self.storage.malloc("insts", self.aux_dat_wth*task_id[1]*self.aux_max_numb)
        upt_func, insts = CsbUpdate().main(func, storage_id, self.aux_dat_wth*task_id[1]//4, self.aux_kvcahce_var)
        upt_call = ir.Call(upt_func)
        self.model_upt += upt_call
        self.model_upt.update_args(upt_func.args)
        self.model_run.update_args(func.args)
        if static:
            self.insts_block.append(insts)
            self.aux_task_ids[self.aux_task_numb] = task_id[0]
            self.aux_axis_numb += task_id[1]
            self.aux_task_numb += 1
            self.aux_upt_cache.append(upt_call)
            if self.aux_task_numb > self.aux_max_numb:
                self.aux_task_finish()
        else:
            for i in range(self.aux_max_numb):
                self.aux_task_ids[i] = task_id[0]
                self.insts_block.append(insts)
            aux_stmt.name = f"aux_block_{self.aux_block_ids}"
            aux_func = Tasks.Get("atom.hbm.aux", self.device)
            aux_func(aux_stmt, self.aux_storage, task_numb*task_id[1], task_numb, [upt_call])
            self.model_run += ir.Call(self.aux_opt_pass(aux_stmt))
            self.model_run.update_args(aux_stmt.args)
            self.insts.append([self.aux_task_ids] + self.insts_block)
            self.insts_block = []
            self.aux_block_ids += 1
    
    def aux_task_finish(self):
        if self.aux_task_numb == 0:
            return
        with ir.Function([], name=f"aux_block_{self.aux_block_ids}") as f:
            aux_func = Tasks.Get("atom.hbm.aux", self.device)
            aux_func(f, self.aux_storage, self.aux_axis_numb, self.aux_task_numb, self.aux_upt_cache)
        self.model_run += ir.Call(self.aux_opt_pass(f))
        self.insts.append([self.aux_task_ids] + self.insts_block)
        self.insts_block = []
        self.aux_axis_numb, self.aux_task_numb = 0, 0
        self.aux_upt_cache = []
        self.aux_block_ids += 1

    def build(self, expr, init_addr, mod_name):
        self.device = expr.get_device()
        self.aux_dat_wth = self.device.AXI_DAT_WIDTH // 8
        self.aux_storage = None
        self.aux_max_numb = 16
        self.aux_task_numb = 0
        self.aux_axis_numb = 0
        self.aux_block_ids = 1
        self.aux_upt_cache = []
        self.aux_kvcahce_var = None
        self.aux_opt_pass = transform.Sequence([
            transform.FoldConstant(),
        ])

        self.insts = []
        self.insts_block = []
        self.inputs = ir.Block()
        self.outputs = ir.Block()
        self.load_params = ir.Function([], name=mod_name+"_load_params")
        self.model_upt = ir.Function([], name=mod_name+"_update")
        self.model_run = ir.Function([], name=mod_name)
        lib = ir.BlockSplit()
        if self.namespace:
            lib.namespace = mod_name
            self.load_params.name = "load_params"
            self.model_upt.name = "model_update"
            self.model_run.name = "model_run"

        graphs = GraphBuild.build(self, expr)

        self.aux_task_finish()
        self.storage.set_address(init_addr)
        lib.body = [self.storage.export(), self.inputs, self.outputs, self.load_params, self.model_upt, self.model_run]
        return lib, graphs, self.storage, None, self.insts

    def visit_var(self, expr):
        expr = super().visit_var(expr)
        self.aux_kvcahce_var = expr.checked_type.shape[-2]
        if not isinstance(self.aux_kvcahce_var, ne.Expr):
            raise RuntimeError("Aux Build Error! No dynamic kvcache token found!")
        return expr

    def visit_call(self, expr):
        expr = GraphBuild.visit_call(self, expr)
        args = [arg.checked_type for arg in expr.args]
        if isinstance(expr.checked_type, Tuple):
            func = expr.op.attrs["compute"](args, expr.checked_type.tensors, **expr.attrs)
        elif isinstance(expr.checked_type, Tensor):
            func = expr.op.attrs["compute"](args, [expr.checked_type], **expr.attrs)
        else:
            raise RuntimeError("GraphModule: infer_type first!")
        func.name = expr.ir_name
        func_ir = self.opt_pass(func)
        if "cfg_id" not in expr.op.attrs.keys():
            msg = f"{expr.op.name} has no cfg_id attribute, which could not finish aux build, please check!"
            raise RuntimeError(msg)
        cfg_id = expr.op.attrs["cfg_id"]
        self.aux_task_append(func_ir, cfg_id)
        return expr
