from time import strftime, localtime
import numpy as np
from .codegen_test_head_ops import CodeGenTestHeadOps
from ..adr import Op, DataEnum
from .. import ne


class CodeGenWt2HbmHead(CodeGenTestHeadOps):

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
        tp_source = [f"void wt2hbm_{enum_name} (HANDLE device, HANDLE h2cx, char* path)" + " {"]
        tp_source.append(f"{self.tab}char real_path[100];")
        tp_source.append(f"{self.tab}for (uint64_t port = 0; port < 32; port++)" + " {")
        tp_source.append(f"{self.tab}{self.tab}sprintf(real_path, path, int(port));")
        tp_source.append(f"{self.tab}{self.tab}DDR_Write_bin(h2cx, real_path, {rt_address}, {total_bytes});")
        tp_source.append(f"{self.tab}{self.tab}uint64_t hbm_out_addr = {hbm_addr} + {1<<log2_Bank_Step} * port;")
        if split_times == 1:
            tp_source.append(f"{self.tab}{self.tab}CSB_Write(device, 64+4, {ddr_address});")
            tp_source.append(f"{self.tab}{self.tab}CSB_Write(device, 64+5, ((uint32_t*)(&hbm_out_addr))[0]);")
            tp_source.append(f"{self.tab}{self.tab}CSB_Write(device, 64+6, ((uint32_t*)(&hbm_out_addr))[1]);")
            tp_source.append(f"{self.tab}{self.tab}CSB_Write(device, 64+7, {last_split_bits});")
            tp_source.append(f"{self.tab}{self.tab}CSB_Write(device, 64+8, port);")
            tp_source.append(f"{self.tab}{self.tab}CSB_Write(device, 64+9, 2);")
            tp_source.append(f"{self.tab}{self.tab}while(CSB_Read(device, 64) != 1) " + "{}")
        else:
            addr_step = self.device.AUX_WT_BUF_DEPTH * self.device.AXI_DAT_WIDTH // 8
            tp_source.append(f"{self.tab}{self.tab}for (int i = 0; i < {split_times}; i++)" + " {")
            tp_source.append(f"{self.tab}{self.tab}{self.tab}CSB_Write(device, 64+4, {ddr_address}+i*{addr_step});")
            tp_source.append(f"{self.tab}{self.tab}{self.tab}CSB_Write(device, 64+5, ((uint32_t*)(&hbm_out_addr))[0]);")
            tp_source.append(f"{self.tab}{self.tab}{self.tab}CSB_Write(device, 64+6, ((uint32_t*)(&hbm_out_addr))[1]);")
            tp_source.append(f"{self.tab}{self.tab}{self.tab}CSB_Write(device, 64+7, (i < {split_times-1}) ? {total_bits_in_each_slice} : {last_split_bits});")
            tp_source.append(f"{self.tab}{self.tab}{self.tab}CSB_Write(device, 64+8, port);")
            tp_source.append(f"{self.tab}{self.tab}{self.tab}CSB_Write(device, 64+9, 2);")
            tp_source.append(f"{self.tab}{self.tab}{self.tab}hbm_out_addr += {addr_step};")
            tp_source.append(f"{self.tab}{self.tab}{self.tab}while(CSB_Read(device, 64) != 1) " + "{}")
            tp_source.append(self.tab + self.tab + "}")
        tp_source.append(self.tab + "}")
        tp_source.append(self.tab + f"init(h2cx, {rt_address}, {total_bytes});")
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
                self.init_weight.append(f"{self.tab}wt2hbm_{enum_name}(device, h2cx, \"{data}\");")
        elif dtype.mapped == DataEnum.ddr:
            address = self.storage.get_address(id, offset, self.ddr_base_addr)
            self.func_const_ddr.append("uint64_t %s = 0x%09x; // %d" % (enum_name, address, address & 0xffffffff))
            self.enum_nodes[2].append(enum_name)
            if isinstance(data, str):
                self.init_weight.append(f"{self.tab}DDR_Write_bin(h2cx, \"{data}\", {address}, {byte_size});")
        else:
            raise RuntimeError(f"Unknown dtype for constant: {dtype.mapped}")
        for n in node["shape"]:
            if isinstance(n, ne.Expr):
                vars = n.get_vars()
                for var in vars:
                    source = f"int {var[0]}"
                    if source not in self.dynamic_var:
                        self.dynamic_var.append(source)