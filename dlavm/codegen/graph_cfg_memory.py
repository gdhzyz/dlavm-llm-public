from .graph_plan_memory import GraphPlanMemory


class GraphCFGMemory(GraphPlanMemory):

    def __init__(self):
        super().__init__()
        self.cfg_num = 0
        self.aux_num = 0
        self.cfg_group = []

    def main(self, expr, init_addr, onchip, debug=1):
        new_expr, storage = super().main(expr, init_addr, onchip, debug)
        if self.cfg_num != 0:
            self.cfg_group[-1][1] = [self.cfg_num, self.aux_num]
        return new_expr, storage, self.cfg_group

    def visit_call(self, expr):
        expr = super().visit_call(expr)
        if not expr.attrs.get("arg_max", 0):
            device = expr.checked_type.device
            if self.cfg_num == 0:
                self.aux_num = 1
                self.cfg_group.append([self.storage.malloc("cfg", device.AXI_DAT_WIDTH // 8), None, []])
            expr_cfg = self.storage.malloc("cfg", expr.op.attrs["cfg_id"][1] * device.AXI_DAT_WIDTH // 8)
            setattr(expr, "cfg_storage_id", expr_cfg)
            self.cfg_group[-1][2].append([expr.op.attrs["cfg_id"][0], self.aux_num])
            self.cfg_num += 1
            self.aux_num += expr.op.attrs["cfg_id"][1]
            if (self.cfg_num >= device.MAX_CFG_NUM and len(self.cfg_group) < 28) or (self.cfg_num >= device.MAX_CFG_NUM + 1 and len(self.cfg_group) >= 28):
                self.cfg_group[-1][1] = [self.cfg_num, self.aux_num]
                self.cfg_num = 0
            return expr
        setattr(expr, "cfg_storage_id", None)
        return expr