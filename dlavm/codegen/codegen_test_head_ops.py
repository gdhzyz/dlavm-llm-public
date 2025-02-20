import numpy as np
from .codegen_test_head import CodeGenTestHead
from time import strftime, localtime
from ..adr import Op
from .. import ne


class CodeGenTestHeadOps(CodeGenTestHead):

    def ext_define(self):
        super().ext_define()
        self.func_inits = []
        self.step_id = 0

    def gen_var(self, node):
        enum_name = node["name"]
        id, offset = node["storage"][0]["id"], node["storage"][0]["offset"]
        address = self.storage.get_address(id, offset, self.ddr_base_addr)
        self.func_input.append("uint64_t %s = 0x%09x; // %d" % (enum_name, address, address & 0xffffffff))
        if enum_name in self.enum_nodes[0] + self.enum_nodes[2] + self.enum_nodes[3]:
            print("*WARNING* : Var或Const节点中存在同名元素，请检查")
            exit(-1)
        else:
            self.enum_nodes[0].append(enum_name)
        for n in node["shape"]:
            if isinstance(n, ne.Expr):
                vars = n.get_vars()
                for var in vars:
                    source = f"int {var[0]}"
                    if source not in self.dynamic_var:
                        self.dynamic_var.append(source)

    def gen_output(self, node):
        enum_name = node["name"]
        id, offset = node["storage"][0]["id"], node["storage"][0]["offset"]
        address = self.storage.get_address(id, offset, self.ddr_base_addr)
        self.func_output.append("uint64_t %s = 0x%09x; // %d" % (enum_name, address, address & 0xffffffff))
        self.enum_nodes[1].append(enum_name)

    def gen_accel(self, node):
        self.step_id += 1
        op_name, ddr_id, offset = node["op_name"], node["storage"][0]["id"], node["storage"][0]["offset"]
        func_op_name = f"step{self.step_id}"
        tp_func_inits = []
        tp_func_inits.append(f"void {func_op_name} (HANDLE device) " + "{")
        tp_func_inits.append(f"// {op_name} accel operator node, storage data in {ddr_id} with {offset} offset")
        tp_func_inits.append("""#ifdef REGS_DEBUG
LARGE_INTEGER start_run;
LARGE_INTEGER stop_run;
LARGE_INTEGER freq;
double time_sec0;
QueryPerformanceFrequency(&freq);
QueryPerformanceCounter(&start_run);
for (int i = 0; i < 1000; i=i+1) {
#endif""")
        tp_dynamic_var, local_dynamic_var = ["device"], []
        tab_num = 1
        for reg in node["csb_regs"]:
            if reg[0] == 1:
                data = reg[2]
                if isinstance(reg[2], ne.Expr):
                    for var in reg[2].get_vars():
                        source = f"int {var[0]}"
                        if var[0] not in local_dynamic_var:
                            if source not in self.dynamic_var:
                                self.dynamic_var.append(source)
                            if var[0] not in tp_dynamic_var:
                                tp_dynamic_var.append(var[0])
                    data = reg[2].export("cpp")
                tp_func_inits.append(f"{self.tab}"*tab_num + f"CSB_Write(device, {reg[1]}, {data});")
            elif reg[0] == 0:
                tp_func_inits.append(f"#ifdef PRINT_STEP\nprintf(\"start: {func_op_name}!\\n\");\n#endif")
                tp_func_inits.append(f"{self.tab}"*tab_num + f"while(CSB_Read(device, {reg[1]}) != {reg[2]}) " + "{}")
            elif reg[0] == -2:
                args = [_reg.export("cpp") if isinstance(_reg, ne.Expr) else str(_reg) for _reg in reg[1]]
                tp_func_inits.append(f"{self.tab}"*tab_num + f"for (int {args[0]} = {args[1]}; {args[0]} < {args[2]}; {args[0]}++)" + " {")
                local_dynamic_var += [f"{i[0]}" for i in reg[1][0].get_vars()]
                tab_num += 1
            elif reg[0] == -1:
                tab_num -= 1
                tp_func_inits.append(f"{self.tab}"*tab_num + "}")
            else:
                raise RuntimeWarning("No realized this codegen")
        tp_func_inits.append("""#ifdef REGS_DEBUG
}
QueryPerformanceCounter(&stop_run);
time_sec0 = (unsigned long long)(stop_run.QuadPart - start_run.QuadPart) / (double)freq.QuadPart;
printf("%(op_name)s run time     = %%fs(1000 times), %%fs(1 times) \\n",time_sec0, time_sec0/1000);
#endif""" % {"op_name": op_name})
        tp_dynamic_str_0 = ", ".join(["HANDLE " + tp_dynamic_var[i] if i == 0 else "int " + tp_dynamic_var[i] for i in range(len(tp_dynamic_var))])
        tp_dynamic_str_1 = ", ".join(tp_dynamic_var)
        tp_func_inits.append("}\n")
        tp_func_inits[0] = f"void {func_op_name} ({tp_dynamic_str_0}) " + "{"
        self.func_inits += tp_func_inits
        self.func_body.append(f"{self.tab}{func_op_name}({tp_dynamic_str_1});")

    def to_string(self):
        super().to_string()
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
        source = """// generated by codegen c++ test ops head at %(local_time)s
%(storages)s

// get input ptr
%(func_input)s

// get output ptr
%(func_output)s

// get weight ddr ptr
%(func_const_ddr)s

// get weight hbm ptr
%(func_const_hbm)s

// mod init
%(func_init)s

void %(mod_name)s(%(mod_args)s) {
%(func_body)s
}
""" % source_map
        return source
