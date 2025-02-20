import numpy as np
from .codegen_test_head_ops import CodeGenTestHeadOps
from time import strftime, localtime
from ..adr import Op
from .. import ne


class CodeGenPython(CodeGenTestHeadOps):

    def ext_define(self):
        super().ext_define()
        self.func_inits = []
        self.step_id = 0

    def gen_var(self, node):
        enum_name = node["name"]
        id, offset = node["storage"][0]["id"], node["storage"][0]["offset"]
        address = self.storage.get_address(id, offset)
        self.func_input.append("%s = 0x%09x # %d" % (enum_name, address, address & 0xffffffff))
        if enum_name in self.enum_nodes[0] + self.enum_nodes[2] + self.enum_nodes[3]:
            print("*WARNING* : Var或Const节点中存在同名元素，请检查")
            exit(-1)
        else:
            self.enum_nodes[0].append(enum_name)
        for n in node["shape"]:
            if isinstance(n, ne.Expr):
                vars = n.get_vars()
                for var in vars:
                    source = f"{var[0]}"
                    if source not in self.dynamic_var:
                        self.dynamic_var.append(source)

    def gen_output(self, node):
        enum_name = node["name"]
        id, offset = node["storage"][0]["id"], node["storage"][0]["offset"]
        address = self.storage.get_address(id, offset)
        self.func_output.append("%s = 0x%09x # %d" % (enum_name, address, address & 0xffffffff))
        self.enum_nodes[1].append(enum_name)

    def gen_const(self, node):
        enum_name, data = node["name"], node.get("data", None)
        id, offset = node["storage"][0]["id"], node["storage"][0]["offset"]
        address = self.storage.get_address(id, offset)
        if enum_name in self.enum_nodes[0] + self.enum_nodes[2] + self.enum_nodes[3]:
            print("*WARNING* : Var或Const节点中存在同名元素，请检查")
            exit(-1)
        if id[:3] == "ddr":
            self.func_const_ddr.append("%s = 0x%09x # %d" % (enum_name, address, address & 0xffffffff))
            self.enum_nodes[2].append(enum_name)
        elif id[:3] == "hbm":
            self.func_const_hbm.append("%s = 0x%09x # %d" % (enum_name, address, address & 0xffffffff))
            self.enum_nodes[3].append(enum_name)
            if isinstance(data, str):
                for i in range(self.device.HBM_Port):
                    fpath = data % i
                    real_address = address + i*(1 << self.device.log2_Bank_Step)
                    self.init_weight.append(f"{self.tab}DDR_Write_bin(\"{fpath}\", {real_address})")
        else:
            self.func_const_ddr.append("%s = 0x%09x # %d" % (enum_name, address, address & 0xffffffff))
            self.enum_nodes[2].append(enum_name)
            if isinstance(data, str):
                self.init_weight.append(f"{self.tab}DDR_Write_bin(\"{data}\", {address})")
        for n in node["shape"]:
            if isinstance(n, ne.Expr):
                vars = n.get_vars()
                for var in vars:
                    source = f"{var[0]}"
                    if source not in self.dynamic_var:
                        self.dynamic_var.append(source)

    def gen_accel(self, node):
        self.step_id += 1
        op_name, ddr_id, offset = node["op_name"], node["storage"][0]["id"], node["storage"][0]["offset"]
        func_op_name = f"step{self.step_id}"
        tp_func_inits = []
        tp_func_inits.append(f"def {func_op_name} ():")
        tp_func_inits.append(f"{self.tab}# {op_name}")
        tp_dynamic_var, local_dynamic_var = [], []
        tab_num = 1
        for reg in node["csb_regs"]:
            if reg[0] == 1:
                data = reg[2]
                if isinstance(reg[2], ne.Expr):
                    for var in reg[2].get_vars():
                        source = f"{var[0]}"
                        if var[0] not in local_dynamic_var:
                            if source not in self.dynamic_var:
                                self.dynamic_var.append(source)
                            if var[0] not in tp_dynamic_var:
                                tp_dynamic_var.append(var[0])
                    data = reg[2].export("py")
                tp_func_inits.append(f"{self.tab}"*tab_num + f"CSB_Write({reg[1]}, {data})")
            elif reg[0] == 0:
                tp_func_inits.append(f"{self.tab}"*tab_num + f"while(CSB_Read({reg[1]}) != {reg[2]}):\n"+f"{self.tab}"*(tab_num+1)+"pass")
            elif reg[0] == -2:
                tp_func_inits.append(f"{self.tab}"*tab_num + f"for {reg[1][0]} in range({reg[1][1]}, {reg[1][2]}):")
                local_dynamic_var += [i[0] for i in reg[1][0].get_vars()]
                tab_num += 1
            elif reg[0] == -1:
                tab_num -= 1
            else:
                raise RuntimeWarning("No realized this codegen")
        tp_dynamic_str_0 = ", ".join(tp_dynamic_var)
        tp_dynamic_str_1 = ", ".join(tp_dynamic_var)
        tp_func_inits.append("\n")
        tp_func_inits[0] = f"def {func_op_name} ({tp_dynamic_str_0}):"
        self.func_inits += tp_func_inits
        self.func_body.append(f"{self.tab}{func_op_name}({tp_dynamic_str_1})")

    def to_string(self):
        super().to_string()
        self.mod_args = ", ".join(self.dynamic_var)
        self.storages_str = self.storage.gen_source("py")
        self.func_init_str = f"def {self.mod_name}_load_params()" + " :\n" + ("\n".join(self.init_weight) if len(self.init_weight) else f"{self.tab}pass") + "\n\n"
        self.func_init_str += "\n" + "\n".join(self.func_inits)

    def gen_source(self):
        local_time = strftime('%Y-%m-%d %H:%M:%S', localtime())
        self.to_string()
        source_map = {
            "local_time": local_time, "enum_define": self.enum_define_str, "func_input": self.func_input_str,
            "func_const_ddr": self.func_wtddr_str, "func_const_hbm": self.func_wthbm_str, "tab": self.tab,
            "func_output": self.func_output_str, "func_body": self.func_body_str, "mod_name": self.mod_name,
            "storages": self.storages_str, "mod_args": self.mod_args, "func_init": self.func_init_str,
        }
        source = """# generated by codegen c++ test ops head at %(local_time)s
from ffi import CSB_Write, CSB_Read, DDR_Write_bin
%(storages)s

# get input ptr
%(func_input)s

# get output ptr
%(func_output)s

# get weight ddr ptr
%(func_const_ddr)s

# get weight hbm ptr
%(func_const_hbm)s

# mod init
%(func_init)s

def %(mod_name)s(%(mod_args)s):
%(func_body)s
""" % source_map
        return source

