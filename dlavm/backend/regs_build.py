from ..adr import Functor, VM, Tensor, Tuple, DataEnum
from .graph_build import GraphBuild 
from dlavm.driver import Tasks, ir, transform
from dlavm import ne


class RegsBuild(GraphBuild):

    def __init__(self, wt2hbm, ddr_base, hbm_base, lite=False, namespace=False, **kwargs):
        super().__init__(**kwargs)
        self.wt2hbm = wt2hbm
        self.ddr_base = ddr_base
        self.hbm_base = hbm_base
        self.namespace = namespace
        self.opt_pass = transform.Sequence([
            transform.FoldConstant(),
            transform.LoopSimplify(eliminate=lite),
            transform.DeadCodeEliminate(),
        ])

    def _base_addr(self, tensor):
        if tensor.dtype.mapped == DataEnum.hbm:
            return self.hbm_base
        elif tensor.dtype.mapped == DataEnum.ddr:
            return self.ddr_base
        else:
            raise RuntimeError("get unknown type of tensor mapped device")

    def build(self, expr, init_addr, mod_name):
        self.inputs = ir.Block()
        self.outputs = ir.Block()
        self.load_params = ir.Function([], name=mod_name+"_load_params")
        self.model_run = ir.Function([], name=mod_name)
        lib = ir.BlockSplit()
        if self.namespace:
            lib.namespace = mod_name
            self.load_params.name = "load_params"
            self.model_run.name = "model_run"

        graphs = super().build(expr)

        self.storage.set_address(init_addr)
        lib.body = [self.storage.export(), self.inputs, self.outputs, self.load_params, self.model_run]
        return lib, graphs, self.storage, None, None

    def wrap_output(self, tensor):
        super().wrap_output(tensor)
        if isinstance(tensor, Tensor):
            if hasattr(tensor, "csb_read"):
                with ir.Function([], ret="int", name="wrap_" + self.output_name) as f:
                    f += ir.Return(ir.CSB_Read(tensor.csb_read))
                self.outputs += f
            else:
                address = ne.Var(tensor.storage_id, -1, "uint64_t")+tensor.offset + self._base_addr(tensor)
                self.outputs += ir.Assign("output", address, "uint64_t")
        elif isinstance(tensor, Tuple):
            for t, num in enumerate(tensor.tensors):
                if hasattr(t, "csb_read"):
                    with ir.Function([], ret="int", name="wrap_" + self.output_name) as f:
                        f += ir.Return(ir.CSB_Read(t.csb_read))
                    self.outputs += f
                else:
                    address = ne.Var(t.storage_id, -1, "uint64_t")+t.offset + self._base_addr(tensor)
                    self.outputs += ir.Assign(f"{self.output_name}_{num}", address, "uint64_t")

    def visit_var(self, expr):
        expr = super().visit_var(expr)
        tensor = expr.checked_type
        address = ne.Var(tensor.storage_id, -1, "uint64_t")+tensor.offset + self._base_addr(tensor)
        self.inputs += ir.Assign(expr.name, address, "uint64_t")
        return expr

    def visit_constant(self, expr):
        expr = super().visit_constant(expr)
        device, storage_id = expr.checked_type.device, expr.checked_type.storage_id
        total_bytes = expr.checked_type.get_bytesize()
        if expr.dtype.mapped == DataEnum.hbm and expr.dtype.dtype == DataEnum.int4:
            if self.wt2hbm:
                # TODO: temp ddr storage should not be "runtime0" at all times
                func = Tasks.Get("atom.hbm.wt2hbm", device)("runtime0", storage_id, expr.data, total_bytes, self.ddr_base, device)
            else:
                func = Tasks.Get("atom.hbm.pcie2mem", device)(storage_id, expr.data, total_bytes, self.hbm_base, device, True)
        else:
            func = Tasks.Get("atom.hbm.pcie2mem", device)(storage_id, expr.data, total_bytes, self.ddr_base, device, False)
        func.name = expr.ir_name
        func_ir = self.opt_pass(func)
        self.load_params += ir.Call(func_ir)
        self.load_params.update_args(func_ir.args)
        return expr

    def visit_call(self, expr):
        expr = super().visit_call(expr)
        args = [arg.checked_type for arg in expr.args]
        if isinstance(expr.checked_type, Tuple):
            func = expr.op.attrs["compute"](args, expr.checked_type.tensors, **expr.attrs)
        elif isinstance(expr.checked_type, Tensor):
            func = expr.op.attrs["compute"](args, [expr.checked_type], **expr.attrs)
        else:
            raise RuntimeError("GraphModule: infer_type first!")
        func.name = expr.ir_name
        func_ir = self.opt_pass(func)
        self.model_run += ir.Call(func_ir)
        self.model_run.update_args(func_ir.args)
        return expr
