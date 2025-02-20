from ..adr import Functor, VM, Tensor, Tuple, DataEnum
from .plan_memory import Storage
from ..utils import LOG_WITH_PREFIX


class GraphBuild(Functor):

    def __init__(self, debug=True, onchip=False, align=0x10, output_name="output"):
        super().__init__()
        self.output_name = output_name
        self.align = align
        self.debug = debug
        self.onchip = onchip

    def build(self, expr):
        self.graphs = []
        self.vm_step = 0
        self.call_step = 0
        self.storage = Storage()

        # first depth-first traversal to check if result of node could be free after computing 
        info = Functor()
        info.visit(expr)
        self.info_memo = info.memo

        expr = self.visit(expr)
        self.wrap_output(expr.checked_type)

        return self.graphs

    def wrap_output(self, tensor):
        if isinstance(tensor, Tensor):
            self.graphs.append({
                "type": "output",
                "name": self.output_name,
                "tensor": tensor,
            })
        elif isinstance(tensor, Tuple):
            for t, num in enumerate(tensor.tensors):
                self.graphs.append({
                    "type": "output",
                    "name": f"{self.output_name}_{num}",
                    "tensor": t,
                })

    def _step_name(self):
        self.call_step += 1
        return f"step_{self.call_step}"

    def _vm_name(self):
        self.vm_step += 1
        return f"step_{self.vm_step}"

    def _link_storage(self, arg, expr):
        if not hasattr(self, "linked"):
            self.linked = [[arg, expr]]
            return
        for index in range(len(self.linked)):
            if arg in self.linked[index]:
                self.linked[index].append(expr)
                return
        self.linked.append([arg, expr])

    def _check_free(self, arg):
        if hasattr(self, "linked"):
            for index, ll in enumerate(self.linked):
                if arg in ll:
                    for a in ll:
                        if self.info_memo[a][1] != 1:
                            self.info_memo[a][1] -= 1
                            return
                    self.storage.free(arg.checked_type.storage_id)
                    if hasattr(arg.checked_type, "onchip"):
                        self.storage.free(arg.checked_type.onchip)
                    del self.linked[index]
                    return
        if self.info_memo[arg][1] == 1:
            self.storage.free(arg.checked_type.storage_id)
            if hasattr(arg.checked_type, "onchip"):
                self.storage.free(arg.checked_type.onchip)
        else:
            self.info_memo[arg][1] -= 1

    def _malloc(self, tensor, prefix, kvcache=0):
        bytesize = tensor.get_bytesize()
        if bytesize % self.align:
            bytesize = (bytesize // self.align + 1) * self.align
        if kvcache:
            return self.storage.malloc(prefix, tensor.get_bytesize({"token":1}))
        elif tensor.dtype.mapped == DataEnum.ddr:
            return self.storage.malloc(prefix, bytesize)
        elif tensor.dtype.mapped == DataEnum.hbm:
            return self.storage.malloc(prefix, bytesize)
        else:
            print("unknown device, please check!")
            exit(-1)

    def visit_var(self, expr):
        storage_id = self._malloc(expr.checked_type, expr.prefix)
        expr.checked_type.storage_id = storage_id
        self.graphs.append({
            "type": "variable",
            "name": expr.name,
            "ir_name": expr.name,
            "tensor": expr.checked_type,
        })
        setattr(expr, "ir_name", expr.name)
        return expr

    def visit_constant(self, expr):
        storage_id = self._malloc(expr.checked_type, expr.prefix)
        expr.checked_type.storage_id = storage_id
        self.info_memo[expr][1] = 0
        setattr(expr, "ir_name", f"{expr.name}_load_param")
        self.graphs.append({
            "type": "constant",
            "name": expr.name,
            "ir_name": expr.ir_name,
            "tensor": expr.checked_type,
        })
        return expr

    def visit_call(self, expr):
        new_args = [self.visit(arg) for arg in expr.args]
        expr.args = new_args
        not_last_op = True
        onchip = self.onchip and "kvcache" in expr.attrs and not_last_op and "runtime" == expr.prefix and expr.attrs.get("onchip", 1)
        if isinstance(expr.checked_type, Tuple):
            for i in range(len(expr.checked_type.tensors)):
                storage_id = self._malloc(expr.checked_type.tensors[i], expr.prefix)
                expr.checked_type.tensors[i].storage_id = storage_id
            if self.debug:
                tp_args = ", ".join([i.checked_type.storage_id for i in new_args])
                tp_outs = ", ".join([i.storage_id for i in expr.checked_type.tensors])
                log = f"{expr.op.name} [{tp_args}] -> [{tp_outs}]"
                LOG_WITH_PREFIX("graph", log)
        elif isinstance(expr.checked_type, Tensor):
            storage_id = self._malloc(expr.checked_type, expr.prefix)
            expr.checked_type.storage_id = storage_id
            extern_debug = ""
            if onchip:
                onchip_id = self._malloc(expr.checked_type, "onchip", kvcache=1)
                setattr(expr.checked_type, "onchip", onchip_id)
                setattr(expr.checked_type, "onchip_offset", 0)
                extern_debug = f"/{onchip_id}"
            if self.debug:
                tp_args = ", ".join([i.checked_type.storage_id for i in new_args])
                log = f"{expr.op.name} [{tp_args}] -> {expr.checked_type.storage_id}{extern_debug}"
                LOG_WITH_PREFIX("graph", log)
        else:
            raise RuntimeError("GraphModule: infer_type first!")
        for arg in new_args:
            self._check_free(arg)
        setattr(expr, "ir_name", ("." + expr.op.name).split(".")[-1] + f"_{self._step_name()}")
        self.graphs.append({
            "type": "accelop",
            "op_name": expr.op.name,
            "args": [arg.ir_name for arg in expr.args],
            "ir_name": expr.ir_name,
            "tensor": expr.checked_type,
        })
        return expr

    def visit_tupleitem(self, expr):
        arg = self.visit(expr.expr)
        expr.expr = arg
        expr.checked_type = arg.checked_type.tensors[expr.index]
        if isinstance(expr.expr, VM):
            self._link_storage(expr.expr, expr)
        setattr(expr, "ir_name", expr.expr.ir_name + f"_{expr.index}")
        return expr

    def visit_vm(self, expr):
        new_args = [self.visit(arg) for arg in expr.args]
        expr.args = new_args
        storage_id = new_args[0].checked_type.storage_id
        if isinstance(expr.checked_type, Tuple):
            for i in range(len(expr.checked_type.tensors)):
                expr.checked_type.tensors[i].storage_id = storage_id
                if hasattr(new_args[0].checked_type, "onchip"):
                    setattr(expr.checked_type.tensors[i], "onchip", new_args[0].checked_type.onchip)
                    setattr(expr.checked_type.tensors[i], "onchip_offset", 0)
        elif isinstance(expr.checked_type, Tensor):
            expr.checked_type.storage_id = storage_id
            if hasattr(new_args[0].checked_type, "onchip"):
                setattr(expr.checked_type, "onchip", new_args[0].checked_type.onchip)
                setattr(expr.checked_type, "onchip_offset", 0)
        else:
            print("infer_type first!")
            exit(-1)
        self._link_storage(new_args[0], expr)
        tensors = [arg.checked_type for arg in expr.args]
        new_checked_type = expr.op.attrs["driver"](tensors, expr.checked_type, expr.attrs)
        expr.checked_type = new_checked_type
        setattr(expr, "ir_name", ("." + expr.op.name).split(".")[-1] + f"_{self._vm_name()}")
        self.graphs.append({
            "type": "virtualop",
            "op_name": expr.op.name,
            "args": [arg.ir_name for arg in expr.args],
            "ir_name": expr.ir_name,
            "tensor": expr.checked_type,
        })
        return expr

