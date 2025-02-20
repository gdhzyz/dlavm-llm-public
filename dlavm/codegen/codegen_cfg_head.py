from time import strftime, localtime
import numpy as np
from .codegen_test_head import CodeGenTestHead
from ..adr import Op
from .. import ne


class CodeGenCFGHead(CodeGenTestHead):

    def build(self, mod_name: str, module, storage, cfg_group, AXI_DAT_NUM, device=None):
        self.AXI_DAT_NUM = AXI_DAT_NUM
        self.cfg_group = cfg_group
        return super().build(mod_name, module, storage, device)

    def ext_define(self):
        super().ext_define()
        self.step_id = 0
        self.cfg_numb = [0, 0]
        self.task_cfg = []
        self.func_cfg = []
        self.func_init = ""
        self.func_inits = []
        self.func_update = []
        self.task_latency = []
        self.var_mod = []
        self.var_upt = []

    def to_string(self):
        super().to_string()
        self.func_init_str = "\n".join(self.func_inits + [self.func_init])
        self.func_init_str += f"\n\nvoid {self.mod_name}_load_params(HANDLE device, HANDLE h2cx)" + " {\n" + "\n".join(self.init_weight) + "\n}"
        self.mod_args = ", ".join(["HANDLE device", "HANDLE h2cx"] + self.var_mod + self.var_upt)
        self.update_args = ", ".join(["HANDLE h2cx"] + self.var_upt)
        self.func_update_str = "\n".join(self.func_update)

    def gen_cfg(self):
        addr = self.storage.get_address(self.cfg_group[self.cfg_numb[0]][0], 0)
        max_tasks_num, aux_group_num = self.cfg_group[self.cfg_numb[0]][1]
        task_latency_str = "\n".join(self.task_latency)
        self.func_body.append("""#ifdef REGS_DEBUG
QueryPerformanceCounter(&start_run);
#endif
%(tab)sCSB_Write(device, 64+1, %(addr)s);
%(tab)sCSB_Write(device, 64+2, %(aug_group_num)d);
%(tab)sCSB_Write(device, 64+3, %(max_tasks_num)d);
%(tab)sCSB_Write(device, 64+9, 1);//start
#ifdef LATENCY_HIDING
%(upt_task)s
#endif
%(tab)swhile(CSB_Read(device, 64) != 1) {}
#ifdef REGS_DEBUG
QueryPerformanceCounter(&stop_run);
time_sec1 = (unsigned long long)(stop_run.QuadPart - start_run.QuadPart) / (double)freq.QuadPart;
printf("cfg run time     = %%fs \\n",time_sec1);
#endif""" % {"tab": self.tab, "addr": addr, "aug_group_num": aux_group_num, "max_tasks_num": max_tasks_num, "upt_task": task_latency_str})
        self.task_latency = []

    def init_cfg_bin(self):
        if not hasattr(self, "cfg_bin"):
            total_aux_num = 0
            for cfg in self.cfg_group:
                total_aux_num += cfg[1][1]
            self.cfg_bin = np.zeros(shape=(total_aux_num, self.AXI_DAT_NUM), dtype="uint32")
            byte_size = total_aux_num * self.AXI_DAT_NUM * 4
            self.func_init = """void %(mod_name)s_init(HANDLE h2cx, char* fpath) {
%(tab)sDDR_Write_bin(h2cx, fpath, %(id)s+%(base_addr)#x, %(byte_size)d);
}""" % {"mod_name": self.mod_name, "tab": self.tab, "id": self.cfg_group[0][0], "byte_size": byte_size, "base_addr": self.ddr_base_addr}

    def update_cfg(self):
        if self.cfg_numb[1] == len(self.cfg_group[self.cfg_numb[0]][2])-1:
            self.gen_cfg()
            index_aux_num = 0
            for i in range(self.cfg_numb[0]):
                index_aux_num += self.cfg_group[i][1][1]
            for i_cfg in range(len(self.cfg_group[self.cfg_numb[0]][2])):
                tp_i_cfg = i_cfg % 4
                self.cfg_bin[index_aux_num, i_cfg // 4] += (self.cfg_group[self.cfg_numb[0]][2][i_cfg][0]) << (8*(tp_i_cfg))

    def write_cfg_bin(self, index, data):
        index_aux_num = 0
        for i in range(self.cfg_numb[0]):
            index_aux_num += self.cfg_group[i][1][1]
        index_aux_num += self.cfg_group[self.cfg_numb[0]][2][self.cfg_numb[1]][1]
        l_offset = index % self.AXI_DAT_NUM
        index_aux_num += index // self.AXI_DAT_NUM
        self.cfg_bin[index_aux_num, l_offset] = data

    def gen_accel(self, node):
        self.step_id += 1
        op_name, ddr_id, offset, cfg_storage = node["op_name"], node["storage"][0]["id"], node["storage"][0]["offset"], node["cfg_storage"]
        cfg_id = Op.Get(op_name).attrs["cfg_id"][0]
        if cfg_storage is None:
            func_op_name = f"step{self.step_id}"
        else:
            if cfg_id != self.cfg_group[self.cfg_numb[0]][2][self.cfg_numb[1]][0]:
                raise RuntimeError("CFG Group not match real CFG Operator")
            self.init_cfg_bin()
            func_op_name = f"update_step{self.step_id}"
        tp_func_inits = []
        tp_func_inits.append(f"void {func_op_name} (HANDLE device) " + "{")
        tp_func_inits.append(f"// {op_name} accel operator node, storage data in {ddr_id} with {offset} offset")
        if cfg_storage is None:
            tp_func_inits.append("""#ifdef REGS_DEBUG
LARGE_INTEGER start_run;
LARGE_INTEGER stop_run;
LARGE_INTEGER freq;
double time_sec0;
QueryPerformanceFrequency(&freq);
QueryPerformanceCounter(&start_run);
for (int i = 0; i < 1000; i=i+1) {
#endif""")
        if cfg_storage is None:
            tp_dynamic_var = ["device"]
        else:
            tp_dynamic_var = ["h2cx"]
        for num, reg in enumerate(node["csb_regs"]):
            if reg[0] == 1:
                data = reg[2]
                if isinstance(reg[2], ne.Expr):
                    dyn_vars = reg[2].get_vars()
                    for var in dyn_vars:
                        if cfg_storage is not None:
                            source = f"int upt_{var[0]}"
                            if source not in self.var_upt:
                                self.var_upt.append(source)
                        else:
                            source = f"int {var[0]}"
                            if source not in self.var_mod:
                                self.var_mod.append(source)
                        if var[0] not in tp_dynamic_var:
                            tp_dynamic_var.append(var[0])
                    new_expr = ne.expr_var_from_dict(reg[2], {"kvcache": 1})
                    data = reg[2].export("cpp")
                    if cfg_storage is not None and len(dyn_vars):
                        if isinstance(new_expr.simplify(), ne.Numb):
                            tp_func_inits.append(f"{self.tab}if (full_update) DDR_Update(h2cx, {cfg_storage}+{self.ddr_base_addr:#x}+{num<<2}, {data});")
                        else:
                            tp_func_inits.append(f"{self.tab}DDR_Update(h2cx, {cfg_storage}+{self.ddr_base_addr:#x}+{num<<2}, {data});")
                        continue
                if cfg_storage is not None:
                    if node["csb_regs"][num+1][0]:
                        self.write_cfg_bin(num, data)
                else:
                    tp_func_inits.append(f"{self.tab}CSB_Write(device, {reg[1]}, {data});")
            elif reg[0] == 0:
                if cfg_storage is None:
                    tp_func_inits.append(f"#ifdef PRINT_STEP\nprintf(\"{func_op_name}!\\n\");\n#endif")
                    tp_func_inits.append(f"{self.tab}while(CSB_Read(device, {reg[1]}) != {reg[2]}) " + "{}")
        if cfg_storage is None:
            tp_func_inits.append("""#ifdef REGS_DEBUG
}
QueryPerformanceCounter(&stop_run);
time_sec0 = (unsigned long long)(stop_run.QuadPart - start_run.QuadPart) / (double)freq.QuadPart;
printf("%(op_name)s run time     = %%fs(1000 times), %%fs(1 times) \\n",time_sec0, time_sec0/1000);
#endif""" % {"op_name": op_name})
        if cfg_storage is not None:
            tp_dynamic_var.append("full_update")
        tp_dynamic_str_0 = ", ".join(["HANDLE " + tp_dynamic_var[i] if i == 0 else "int " + tp_dynamic_var[i] for i in range(len(tp_dynamic_var))])
        tp_func_inits.append("}\n")
        tp_func_inits[0] = f"void {func_op_name} ({tp_dynamic_str_0}) " + "{"
        self.func_inits += tp_func_inits
        if cfg_storage is not None:
            if "int upt_full_update" not in self.var_upt:
                self.var_upt.append("int upt_full_update")
            tp_dynamic_str_1 = ", ".join([tp_dynamic_var[0]] + ["upt_" + i for i in tp_dynamic_var[1:]])
            self.func_update.append(f"{self.tab}{func_op_name}({tp_dynamic_str_1});")
            self.task_latency.append(f"{self.tab}{func_op_name}({tp_dynamic_str_1});")
            self.update_cfg()
            if self.cfg_numb[1] == len(self.cfg_group[self.cfg_numb[0]][2]) - 1:
                self.cfg_numb = [self.cfg_numb[0]+1, 0]
            else:
                self.cfg_numb[1] += 1
        else:
            tp_dynamic_str_1 = ", ".join(tp_dynamic_var)
            self.func_body.append(f"{self.tab}{func_op_name}({tp_dynamic_str_1});")

    def gen_source(self):
        cfg_params = self.cfg_bin.tobytes()
        local_time = strftime('%Y-%m-%d %H:%M:%S', localtime())
        self.to_string()
        source_map = {
            "local_time": local_time, "enum_define": self.enum_define_str, "func_input": self.func_input_str,
            "func_const_ddr": self.func_wtddr_str, "func_const_hbm": self.func_wthbm_str, "tab": self.tab,
            "func_output": self.func_output_str, "func_body": self.func_body_str, "mod_name": self.mod_name,
            "storages": self.storages_str, "mod_args": self.mod_args, "func_init": self.func_init_str,
            "update_args": self.update_args, "func_update": self.func_update_str,
        }
        source = """// generated by codegen c++ test ops head at %(local_time)s
#define LATENCY_HIDING
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

// mod reg update
void %(mod_name)s_update(%(update_args)s) {
%(func_update)s
}

void %(mod_name)s(%(mod_args)s) {
%(func_body)s
}
""" % source_map
        return source, cfg_params, self.cfg_bin
