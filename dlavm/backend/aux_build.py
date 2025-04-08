import numpy as np
from copy import deepcopy
from ..adr import Functor, VM, Tensor, Tuple, DataEnum
from .graph_build import GraphBuild 
from .regs_build import RegsBuild
from dlavm.driver import Tasks, ir, transform
from dlavm import ne


# TODO: 需要重构，主要内容如下
#       1. 由于添加了block模块，对于所有与动态参数参与运算的参数判别出错
#       2. 地址相关的内容，由于地址在此时未分配，所以给入的参数有误，需添加init控制
#       3. kvcache_var参数的设定有些老旧，需要更新
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
            self.vars[new_stmt.var.name] = new_stmt
        return new_stmt

    def VisitAssignVar(self, stmt: ir.AssignVar):
        new_stmt = super().VisitAssignVar(stmt)
        if isinstance(new_stmt.value, ne.Expr):
            self.vars[new_stmt.var.name] = new_stmt
        return new_stmt

    def VisitWhile(self, stmt: ir.While):
        if isinstance(stmt.judge, ir.Op) and isinstance(stmt.judge.arg0, ir.CSB_Read):
            return ir.Empty
        return super().VisitWhile(stmt)

    def VisitCSBWrite(self, stmt: ir.CSB_Write):
        if isinstance(stmt.data, (ne.Expr, ir.Cast)):
            if isinstance(stmt.data, ir.Cast):
                vars = stmt.data.var.get_vars(True)
            else:
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

    def VisitIf(self, stmt: ir.If):
        new_stmt = deepcopy(stmt)
        new_stmt.then_block = deepcopy(stmt.then_block)
        new_stmt.else_block = deepcopy(stmt.else_block)
        new_stmt.judge = self.Visit(stmt.judge)
        beg_task = [self.insts, self.numb, self.vars]
        new_stmt.then_block.body = self.RmEmpty([self.Visit(b) for b in stmt.then_block.body])
        then_task = [self.insts, self.numb, self.vars]
        self.insts, self.numb, self.vars = beg_task
        new_stmt.else_block.body = self.RmEmpty([self.Visit(b) for b in stmt.else_block.body])
        else_task = [self.insts, self.numb, self.vars]
        if (then_task[0] != else_task[0]).any():
            raise RuntimeError("*AUX-BUILD ERROR* : if code found different inst config")
        self.insts, self.numb, self.vars = then_task
        return new_stmt

    def VisitNe(self, expr: ne.Expr):
        new_expr = ne.expr_var_from_dict(expr, self.dict_args).simplify()
        if isinstance(new_expr, ne.Numb):
            new_expr = new_expr.data
        return new_expr


class DynamicTasks(ir.Functor):

    def main(self, stmt: ir.Function):
        """
        返回值：
          task_cnt: 所需执行的算子数
          static: 是否是静态数量
          task_func: 计算真实task数量的ir.func，即aux block的前缀
        仅支持以下格式的ir代码
        for (...):
          for (...):
            csb_write()
            ...
            csb_read()
        
        其中，需要注意的是，for循环内不能存在if判断，否则条件过多，难以直接计算算子调用次数
        """
        self.task_cnt = 0
        self.loop_num = 1
        new_stmt = self.Visit(stmt)
        self.task_cnt = self.loop_num * self.task_cnt
        if isinstance(self.task_cnt, int):
            return self.task_cnt, True, None
        opt_pass = transform.GetSubIRFromVars([i[0] for i in self.task_cnt.get_vars()])
        task_func = opt_pass.Visit(new_stmt)
        task_cnt = task_func[ir.Assign("tasks", self.task_cnt)]
        return task_cnt.var, False, task_func

    def VisitWhile(self, stmt: ir.While):
        if isinstance(stmt.judge, ir.Op) and isinstance(stmt.judge.arg0, ir.CSB_Read):
            self.task_cnt += 1
        return stmt

    def VisitFor(self, stmt: ir.For):
        beg_task = self.task_cnt
        new_stmt = super().VisitFor(stmt)
        if beg_task != self.task_cnt: # 说明此循环中包含算子运行，需要进行计数
            self.loop_num = self.loop_num * (new_stmt.extent + new_stmt.init) // new_stmt.stride
        return new_stmt

    """
    def main(self, stmt: ir.Function):
        self.task_cnt = 0
        self.tasks = 0
        self.block = ir.Block()
        self.vars = {}
        new_stmt = self.Visit(stmt)
        if isinstance(self.tasks, int):
            if self.tasks == 0:
                self.tasks += self.task_cnt
            return self.tasks, True, None
        elif isinstance(self.tasks, ne.Expr):
            self.tasks = self.tasks.simplify()
            used_vars = []
            def _wrap(var):
                used_vars.append(var.name)
                if var.name in self.vars.keys():
                    if isinstance(self.vars[var.name], ir.Block):
                        for new_var in self.vars[var.name].body:
                            if isinstance(new_var, (ir.Assign, ir.AssignVar)):
                                for _var in new_var.value.get_vars(True):
                                    _wrap(_var)
                    else:
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
            if self.task_cnt != 1:
                raise RuntimeError("*AUX-BUILD ERROR* : unsupport code state, please wrapper all loops")
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
        else:
            self.vars[new_stmt.var.name] = ne.Numb(new_stmt)
        return new_stmt

    def VisitAssignVar(self, stmt: ir.AssignVar):
        new_stmt = super().VisitAssignVar(stmt)
        if isinstance(new_stmt.value, ne.Expr):
            print(self.vars.keys())
            with ir.Block() as b:
                b += self.vars[new_stmt.var.name]
                b += new_stmt
            self.vars[new_stmt.var.name] = b
        return new_stmt

    def VisitFor(self, stmt: ir.For):
        beg_task = self.task_cnt
        new_stmt = super().VisitFor(stmt)
        for_task = self.task_cnt - beg_task
        self.tasks = self.tasks + for_task * (new_stmt.extent + new_stmt.init) // new_stmt.stride
        return new_stmt

    def VisitIf(self, stmt: ir.If):
        self.block = ir.If(self.Visit(stmt.judge))
        beg_task = [self.tasks, self.task_cnt]
        [self.Visit(b) for b in stmt.then_block.body]
        then_task = [self.tasks, self.task_cnt]
        self.tasks, self.task_cnt = beg_task
        [self.Visit(b) for b in stmt.else_block.body]
        else_task = [self.tasks, self.task_cnt]
        if then_task[1] != else_task[1]:
            raise RuntimeError("*AUX-BUILD ERROR* : if in this code should has same min-unit")
        self.tasks, self.task_cnt = then_task
        return stmt
    """


class AuxBuild(RegsBuild, GraphBuild):

    def __init__(self, aux_max_numb=32, **kwargs):
        kwargs["min_loop"] = 1
        super().__init__(**kwargs)
        self.aux_max_numb = aux_max_numb

    def aux_task_append(self, func, task_id):
        if self.aux_task_numb == 0:
            self.aux_task_ids = np.zeros([self.aux_dat_wth], dtype="uint8")
            self.aux_storage = ne.Var(self.storage.malloc("insts", self.aux_dat_wth), -1)
        task_numb, static, aux_stmt = DynamicTasks().main(func)
        if static:
            storage_id = self.storage.malloc("insts", self.aux_dat_wth*task_id[1]*task_numb)
        else:
            self.aux_task_finish()
            self.aux_task_ids = np.zeros([self.aux_dat_wth], dtype="uint8")
            self.aux_storage = ne.Var(self.storage.malloc("insts", self.aux_dat_wth), -1)
            storage_id = self.storage.malloc("insts", self.aux_dat_wth*task_id[1]*self.aux_max_numb)
        upt_func, insts = CsbUpdate().main(func, storage_id, self.aux_dat_wth*task_id[1]//4, self.aux_kvcahce_var)
        upt_call = ir.Call(upt_func)
        self.model_upt += upt_call
        self.model_upt.update_args(upt_func.args)
        self.model_run.update_args(func.args)
        if static:
            self.insts_block.append(insts)
            for _ in range(task_numb):
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
            aux_func = Tasks.Get(f"atom.{self.device.name}.aux", self.device)
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
            aux_func = Tasks.Get(f"atom.{self.device.name}.aux", self.device)
            aux_func(f, self.aux_storage, self.aux_axis_numb, self.aux_task_numb, self.aux_upt_cache)
        self.model_run += ir.Call(self.aux_opt_pass(f))
        self.insts.append([self.aux_task_ids] + self.insts_block)
        self.insts_block = []
        self.aux_axis_numb, self.aux_task_numb = 0, 0
        self.aux_upt_cache = []
        self.aux_block_ids += 1

    def build(self, expr, init_addr, mod_name):
        self.device = expr.get_device()
        if not hasattr(self.device, "aux_dat_width"):
            msg = f"*AUX Build Error* : device {self.device.name}-{self.device.version} does NOT support aux module, please add \"aux_dat_width\" factor to support"
            raise RuntimeError(msg)
        self.aux_dat_wth = self.device.aux_dat_width
        self.aux_storage = None
        self.aux_task_numb = 0
        self.aux_axis_numb = 0
        self.aux_block_ids = 1
        self.aux_upt_cache = []
        self.aux_kvcahce_var = None
        self.aux_opt_pass = transform.Sequence([
            transform.FoldConstant(),
            transform.DeadCodeEliminate(),
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
        lib.body = [self.storage.export(self.addr_dtype), self.inputs, self.outputs, self.load_params, self.model_upt, self.model_run]
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
            func = expr.op.get_attr("compute", expr.get_device())(args, expr.checked_type.tensors, expr.attrs)
        elif isinstance(expr.checked_type, Tensor):
            func = expr.op.get_attr("compute", expr.get_device())(args, [expr.checked_type], expr.attrs)
        else:
            raise RuntimeError("GraphModule: infer_type first!")
        func.name = expr.ir_name
        func_ir = self.opt_pass(func)
        if "cfg_id" in expr.op.attrs.keys():
            cfg_id = expr.op.attrs["cfg_id"]
        elif expr.op.get_attr("aux-cfg", expr.get_device()) is not None:
            cfg_id = expr.op.get_attr("aux-cfg", expr.get_device())(args, expr.attrs)
        else:
            msg = f"{expr.op.name} has no cfg_id attribute, which could not finish aux build, please check!"
            raise RuntimeError(msg)
        self.aux_task_append(func_ir, cfg_id)
        return expr
