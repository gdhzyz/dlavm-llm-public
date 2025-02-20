from time import strftime, localtime
from .. import ne
from ..runtime import hbm
from ..clib import runtime as rt


class ExecuteModule:

    ddr_base_addr = 0x200000000

    def __init__(self, module, storage):
        self.module = module
        self.storage = storage
        self.CSB_Read, self.CSB_Write, self.DDR_Read, self.DDR_Write = rt.init()

    def execute(self, **args):
        for node in self.module:
            if node["node"] == "accel_op":
                self._exe_accel(node, **args)
            elif node["node"] == "cpu_op":
                self._exe_cpu(node, **args)
            elif node["node"] == "virtual_op":
                self._exe_virtual(node, **args)

    def set_inputs(self, name, inputs):
        for node in self.module:
            if node["node"] == "var" and node["name"] == name:
                self._set_var(node, inputs)
                return 1
        return 0

    def set_consts(self, prefix):
        for node in self.module:
            if node["node"] == "const":
                self._set_const(node, prefix)

    def get_outputs(self):
        outputs = []
        for node in self.module:
            if node["node"] == "output":
                outputs += self._get_output(node)
        return outputs

    def _set_var(self, node, inputs):
        id, offset = node["storage"][0]["id"], node["storage"][0]["offset"]
        address = self.storage.get_address(id, offset, self.ddr_base_addr)
        self.DDR_Write(inputs, address)

    def _set_const(self, node, prefix):
        pass

    def _get_output(node):
        pass

    def _exe_accel(self, node, **args):
        pass

    def _exe_cpu(self, node, **args):
        pass

    def _exe_virtual(self, node, **args):
        pass