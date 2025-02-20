from .graph_csb_head import GraphCSBHead
from .graph_cfg_memory import GraphCFGMemory
from .codegen_cfg_head import CodeGenCFGHead
from .codegen_cfg_wt2hbm import CodeGenCFGWt2HBM


class GraphCFGHead(GraphCSBHead):

    def build(self, expr, init_addr, onchip):
        expr, self.storage, cfg_group = GraphCFGMemory().main(expr, init_addr, onchip)
        self.nodes = []
        outputs = self.visit(expr)
        self._node_output(outputs)
        return expr, self.nodes, self.storage, cfg_group

    def visit_call(self, expr):
        tensors = [self.visit(arg) for arg in expr.args]
        wrap_args = self._make_arguments(tensors)
        output = self._make_output(expr.checked_type)
        if "driver" in expr.op.attrs.keys():
            self.nodes.append({
                "node": "accel_op",
                "op_name": expr.op.name,
                "csb_regs": expr.op.attrs["driver"](wrap_args, output, expr.attrs),
                "storage": self._wrap_storage(expr.checked_type),
                "cfg_storage": expr.cfg_storage_id,
                "output": output,
            })
        elif "source" in expr.op.attrs.keys():
            self.nodes.append({
                "node": "cpu_op",
                "op_name": expr.op.name,
                "source": expr.op.attrs["source"](wrap_args, output, expr.attrs),
                "storage": self._wrap_storage(expr.checked_type),
                "output": output,
            })
        else:
            print(f"Please register the compute function of {expr.op.name}")
            print(expr.op.attrs.keys())
            exit(-1)
        return expr.checked_type


def cfg_head(expr, mod_name, init_addr, onchip):
    expr, mod, storage, cfg_group = GraphCFGHead().build(expr, init_addr, onchip)
    source, params, debug_params = CodeGenCFGHead().build(mod_name, mod, storage, cfg_group, 16, expr.get_device())
    return expr, source, storage, mod, params, debug_params


def cfg_wt2hbm(expr, mod_name, init_addr, onchip):
    expr, mod, storage, cfg_group = GraphCFGHead().build(expr, init_addr, onchip)
    source, params, debug_params = CodeGenCFGWt2HBM().build(mod_name, mod, storage, cfg_group, 16, expr.get_device())
    return expr, source, storage, mod, params, debug_params
