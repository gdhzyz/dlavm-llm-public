from time import strftime, localtime
import numpy as np
from .codegen_test_head_ops import CodeGenTestHeadOps
from ..adr import Op, DataEnum
from .. import ne


class CodeGenV80Head(CodeGenTestHeadOps):

    hbm_base_addr = 0x4000000000
    ddr_base_addr = 0x60000000000

    def ext_define(self):
        super().ext_define()
        self.wt2hbm = []

    def to_string(self):
        super().to_string()
        self.func_init_str = "\n".join(self.wt2hbm) + "\n" + self.func_init_str
    
    def gen_wt2hbm(self, enum_name, hbm_addr, total_bytes, log2_Bank_Step=28):
        total_bits = total_bytes * 8
        split_times=(total_bits+self.device.AUX_WT_BUF_DEPTH*self.device.AXI_DAT_WIDTH-1)//(self.device.AUX_WT_BUF_DEPTH*self.device.AXI_DAT_WIDTH)
        if split_times == 1:
            last_split_bits = total_bits
        else:
            last_split_bits = total_bits % (self.device.AUX_WT_BUF_DEPTH*self.device.AXI_DAT_WIDTH)
        if split_times==1:
            total_bits_in_each_slice=total_bits
        else:
            total_bits_in_each_slice=self.device.AUX_WT_BUF_DEPTH*self.device.AXI_DAT_WIDTH
        rt_address = self.storage.get_address("runtime0", 0, self.ddr_base_addr)
        ddr_address = self.storage.get_address("runtime0", 0)
        tp_source = [f"void wt2hbm_{enum_name} (char* path)" + " {"]
        tp_source.append(f"{self.tab}char real_path[100];")
        tp_source.append(f"{self.tab}for (uint64_t port = 0; port < 32; port++)" + " {")
        tp_source.append(f"{self.tab}{self.tab}sprintf(real_path, path, int(port));")
        tp_source.append(f"{self.tab}{self.tab}start_host2fpga(real_path, {rt_address}, {total_bytes}, 1);")
        tp_source.append(f"{self.tab}{self.tab}uint64_t hbm_out_addr = {hbm_addr} + {1<<log2_Bank_Step} * port;")
        if split_times == 1:
            tp_source.append(f"{self.tab}{self.tab}CSB_Write(64+4, {ddr_address});")
            tp_source.append(f"{self.tab}{self.tab}CSB_Write(64+5, ((uint32_t*)(&hbm_out_addr))[0]);")
            tp_source.append(f"{self.tab}{self.tab}CSB_Write(64+6, ((uint32_t*)(&hbm_out_addr))[1]);")
            tp_source.append(f"{self.tab}{self.tab}CSB_Write(64+7, {last_split_bits});")
            tp_source.append(f"{self.tab}{self.tab}CSB_Write(64+8, port);")
            tp_source.append(f"{self.tab}{self.tab}CSB_Write(64+9, 2);")
            tp_source.append(f"{self.tab}{self.tab}while(CSB_Read(64) != 1) " + "{}")
        else:
            addr_step = self.device.AUX_WT_BUF_DEPTH * self.device.AXI_DAT_WIDTH // 8
            tp_source.append(f"{self.tab}{self.tab}for (int i = 0; i < {split_times}; i++)" + " {")
            tp_source.append(f"{self.tab}{self.tab}{self.tab}CSB_Write(64+4, {ddr_address}+i*{addr_step});")
            tp_source.append(f"{self.tab}{self.tab}{self.tab}CSB_Write(64+5, ((uint32_t*)(&hbm_out_addr))[0]);")
            tp_source.append(f"{self.tab}{self.tab}{self.tab}CSB_Write(64+6, ((uint32_t*)(&hbm_out_addr))[1]);")
            tp_source.append(f"{self.tab}{self.tab}{self.tab}CSB_Write(64+7, (i < {split_times-1}) ? {total_bits_in_each_slice} : {last_split_bits});")
            tp_source.append(f"{self.tab}{self.tab}{self.tab}CSB_Write(64+8, port);")
            tp_source.append(f"{self.tab}{self.tab}{self.tab}CSB_Write(64+9, 2);")
            tp_source.append(f"{self.tab}{self.tab}{self.tab}hbm_out_addr += {addr_step};")
            tp_source.append(f"{self.tab}{self.tab}{self.tab}while(CSB_Read(64) != 1) " + "{}")
            tp_source.append(self.tab + self.tab + "}")
        tp_source.append(self.tab + "}")
        tp_source.append(self.tab + f"// init({rt_address}, {total_bytes}); #TODO")
        tp_source.append("}\n")
        self.wt2hbm.append("\n".join(tp_source))

    def gen_const(self, node):
        enum_name, data, dtype = node["name"], node.get("data", None), node["dtype"]
        id, offset = node["storage"][0]["id"], node["storage"][0]["offset"]
        byte_size = self.storage.get_byte_size(id)
        if enum_name in self.enum_nodes[0] + self.enum_nodes[2] + self.enum_nodes[3]:
            print("*WARNING* : Var或Const节点中存在同名元素，请检查")
            exit(-1)
        if dtype.mapped == DataEnum.hbm:
            address = self.storage.get_address(id, offset, self.hbm_base_addr)
            self.func_const_hbm.append("uint64_t %s = 0x%09x; // %d" % (enum_name, address, address & 0xffffffff))
            self.enum_nodes[3].append(enum_name)
            self.gen_wt2hbm(enum_name, address, self.storage.get_byte_size(id))
            if isinstance(data, str):
                self.init_weight.append(f"{self.tab}wt2hbm_{enum_name}(\"{data}\");")
        elif dtype.mapped == DataEnum.ddr:
            address = self.storage.get_address(id, offset, self.ddr_base_addr)
            self.func_const_ddr.append("uint64_t %s = 0x%09x; // %d" % (enum_name, address, address & 0xffffffff))
            self.enum_nodes[2].append(enum_name)
            if isinstance(data, str):
                self.init_weight.append(f"{self.tab}start_host2fpga(\"{data}\", {address}, {byte_size}, 1);")
        else:
            raise RuntimeError(f"Unknown dtype for constant: {dtype.mapped}")
        for n in node["shape"]:
            if isinstance(n, ne.Expr):
                vars = n.get_vars()
                for var in vars:
                    source = f"int {var[0]}"
                    if source not in self.dynamic_var:
                        self.dynamic_var.append(source)


    def gen_accel(self, node):
        self.step_id += 1
        op_name, ddr_id, offset = node["op_name"], node["storage"][0]["id"], node["storage"][0]["offset"]
        func_op_name = f"step{self.step_id}"
        tp_func_inits = []
        tp_func_inits.append(f"void {func_op_name} () " + "{")
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
        tp_dynamic_var, local_dynamic_var = [], []
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
                tp_func_inits.append(f"{self.tab}"*tab_num + f"CSB_Write({reg[1]}, {data});")
            elif reg[0] == 0:
                tp_func_inits.append(f"#ifdef PRINT_STEP\nprintf(\"start: {func_op_name}!\\n\");\n#endif")
                tp_func_inits.append(f"{self.tab}"*tab_num + f"while(CSB_Read({reg[1]}) != {reg[2]}) " + "{}")
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
        tp_dynamic_str_0 = ", ".join(["int " + tp_dynamic_var[i] for i in range(len(tp_dynamic_var))])
        tp_dynamic_str_1 = ", ".join(tp_dynamic_var)
        tp_func_inits.append("}\n")
        tp_func_inits[0] = f"void {func_op_name} ({tp_dynamic_str_0}) " + "{"
        self.func_inits += tp_func_inits
        self.func_body.append(f"{self.tab}{func_op_name}({tp_dynamic_str_1});")

