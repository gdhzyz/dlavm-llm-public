import numpy as np
from .codegen_csb_head import CodeGenCSBHead
from ..adr import Op
from ..device.hbm_accel import HBM
from .. import ne


class CodeGenCSBModel(CodeGenCSBHead):

    def ext_define(self):
        self.params = b""
        self.offset = []
        self.func_init = []


    def gen_const(self, node):
        enum_name = node["name"]
        id, offset = node["storage"][0]["id"], node["storage"][0]["offset"]
        if "data" in node:
            if id[:3] == "ddr":
                self.params += node["data"].tobytes()
                self.func_init.append(f"")
            elif id[:3] == "hbm":
                self.func_const_hbm.append(f"{self.tab}case {self.mod_name}Node::{enum_name}:")
                self.func_const_hbm.append(f"{self.tab}{self.tab}return (void*)(&{id}[{offset}]);")
                self.enum_nodes[3].append(enum_name)
        else:
            super().gen_const(node)