from .. import driver
from ..adr import Functor, DataEnum, Tensor, Tuple
from .graph_plan_memory import GraphPlanMemory
from .codegen_prototxt import CodeGenPrototxt


class GraphVisualizes(Functor):

    def build(self, expr, init_addr, onchip=False):
        self.step = 0
        self.vm_step = 0
        expr, self.storage = GraphPlanMemory().main(expr, init_addr, onchip)
        self.nodes = []
        outputs = self.visit(expr)
        self._node_output(outputs)
        return expr, self.nodes, self.storage

    def _wrap_storage(self, checked_type):
        if isinstance(checked_type, Tensor):
            return [{"id": checked_type.storage_id, "offset": checked_type.offset}]
        elif isinstance(checked_type, Tuple):
            ret = []
            for tensor in checked_type.tensors:
                ret.append({"id": tensor.storage_id, "offset": tensor.offset})
            return ret
        else:
            print("unknown type of checked_type: " + type(checked_type))
            exit(-1)

    def _make_arguments(self, checked_types):
        wrap_args = []
        for arg in checked_types:
            _tp = [arg, self.storage.get_address(arg.storage_id, arg.offset)]
            if hasattr(arg, "onchip"):
                _tp.append(self.storage.get_address(arg.onchip, arg.offset))
            wrap_args.append(_tp)
        return wrap_args

    def _node_output(self, checked_type):
        if isinstance(checked_type[0], Tensor):
            self.nodes.append({
                "node": "output",
                "name": "data_out",
                "args": [checked_type[1]],
                "storage": self._wrap_storage(checked_type[0]),
                "shape": checked_type[0].shape,
            })
        elif isinstance(checked_type[0], Tuple):
            for num, tensor in enumerate(checked_type[0].tensors):
                self.nodes.append({
                    "node": "output",
                    "args": [checked_type[1] + f"_num"],
                    "name": "data_out" + str(num),
                    "storage": self._wrap_storage(tensor),
                    "shape": tensor.shape,
                })
        else:
            print("unknown type of checked_type: " + str(type(checked_type)))
            exit(-1)

    def _make_output(self, checked_type):
        if isinstance(checked_type, Tensor):
            storage_id, offset = checked_type.storage_id, checked_type.offset
            _tp = [checked_type, self.storage.get_address(storage_id, offset)]
            if hasattr(checked_type, "onchip"):
                _tp.append(self.storage.get_address(checked_type.onchip, offset))
            return _tp
        elif isinstance(checked_type, Tuple):
            outputs = []
            for tensor in checked_type.tensors:
                storage_id, offset = tensor.storage_id, tensor.offset
                _tp = [tensor, self.storage.get_address(storage_id, offset)]
                if hasattr(checked_type, "onchip"):
                    _tp.append(self.storage.get_address(checked_type.onchip, offset))
                outputs.append(_tp)
            return outputs
        else:
            print("unknown type of checked_type: " + type(checked_type))
            exit(-1)

    def visit_var(self, expr):
        self.nodes.append({
            "node": "var",
            "name": expr.name,
            "storage": self._wrap_storage(expr.checked_type),
            "shape": expr.checked_type.shape,
        })
        return [expr.checked_type, expr.name]

    def visit_constant(self, expr):
        if expr.data is not None:
            self.nodes.append({
                "node": "const",
                "name": expr.name,
                "data": expr.data,
                "storage": self._wrap_storage(expr.checked_type),
                "shape": expr.checked_type.shape,
            })
        else:
            self.nodes.append({
                "node": "const",
                "name": expr.name,
                "storage": self._wrap_storage(expr.checked_type),
                "shape": expr.checked_type.shape,
            })
        return [expr.checked_type, expr.name]

    def visit_tupleitem(self, expr):
        [arg, name] = self.visit(expr.expr)
        expr.checked_type = arg.tensors[expr.index]
        return [arg.tensors[expr.index], f"{name}_{expr.index}"]

    def visit_call(self, expr):
        new_args = [self.visit(arg) for arg in expr.args]
        self.step += 1
        name = f"step{self.step}"
        tensors = [args[0] for args in new_args]
        arg_names = [args[1] for args in new_args]
        wrap_args = self._make_arguments(tensors)
        output = self._make_output(expr.checked_type)
        if "driver" in expr.op.attrs.keys():
            self.nodes.append({
                "node": "accel_op",
                "op_name": expr.op.name,
                "csb_regs": expr.op.attrs["driver"](wrap_args, output, expr.attrs),
                "storage": self._wrap_storage(expr.checked_type),
                "output": output,
                "tensors": tensors,
                "args": arg_names,
                "name": name,
                "attrs": expr.attrs,
            })
        elif "source" in expr.op.attrs.keys():
            self.nodes.append({
                "node": "cpu_op",
                "op_name": expr.op.name,
                "source": expr.op.attrs["source"](wrap_args, output, expr.attrs),
                "storage": self._wrap_storage(expr.checked_type),
                "output": output,
                "args": arg_names,
                "tensors": tensors,
                "name": name,
                "attrs": expr.attrs,
            })
        else:
            print(f"Please register the compute function of {expr.op.name}")
            print(expr.op.attrs.keys())
            exit(-1)
        return [expr.checked_type, name]

    def visit_vm(self, expr):
        new_args = [self.visit(arg) for arg in expr.args]
        self.vm_step += 1
        name = f"vm_step{self.vm_step}"
        tensors = [args[0] for args in new_args]
        arg_names = [args[1] for args in new_args]
        new_checked_type = expr.op.attrs["driver"](tensors, expr.checked_type, expr.attrs)
        expr.checked_type = new_checked_type
        output = self._make_output(expr.checked_type)
        self.nodes.append({
            "node": "virtual_op",
            "op_name": expr.op.name,
            "storage": self._wrap_storage(expr.checked_type),
            "args": arg_names,
            "tensors": tensors,
            "output": output,
            "name": name,
            "attrs": expr.attrs,
        })
        return [new_checked_type, name]


def visualize_prototxt(expr, mod_name, init_addr, onchip=False):
    expr, mod, storage = GraphVisualizes().build(expr, init_addr, onchip)
    source = CodeGenPrototxt().build(mod_name, mod, storage, expr.get_device())
    return expr, source, storage, mod