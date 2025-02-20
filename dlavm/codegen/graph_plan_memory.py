from ..adr import Functor, VM, Tensor, Tuple, DataEnum
from .. import ne
from ..utils import LOG_WITH_PREFIX


class StorageNode:

    def __init__(self, byte_size):
        if isinstance(byte_size, ne.Expr):
            byte_size = byte_size.simplify(1).data
        self.byte_size = byte_size

    def get_address(self, offset, base_address):
        if offset == -1:
            return self.address + self.byte_size
        return self.address + offset + base_address

    def set_address(self, address=None):
        if address is None:
            return self.address + self.byte_size
        if isinstance(address, ne.Expr):
            address = address.simplify(1).data
        if hasattr(self, "address"):
            if self.address != address:
                print("*WARNING* : StorageNode new address dosen't match: ", self.address, address)
        self.address = address
        return self.address + self.byte_size

    def __str__(self):
        addr_hex = "0x%09x" % (self.address)
        return "|{0:^15}|{1:^12}*".format(addr_hex, str(self.byte_size)+" B")


class Storage:

    pingpong = ["runtime", "onchip"]

    def __init__(self):
        self.memo_ = {}
        self.free_ = []

    def malloc(self, prefix, data_byte):
        if prefix not in self.memo_.keys():
            self.memo_[prefix] = {}
        if prefix in self.pingpong:
            min_id, min_diff, found = None, None, 0
            for id in self.free_:
                if id not in self.memo_[prefix].keys():
                    continue
                storage = self.memo_[prefix][id]
                new_diff = storage.byte_size - data_byte
                if min_id is None:
                    min_id, min_diff = id, new_diff
                else:
                    if found:
                        if new_diff >= 0:
                            min_id = min_id if new_diff > min_diff else id
                            min_diff = min_diff if new_diff > min_diff else new_diff
                    else:
                        if new_diff >= 0:
                            min_id, min_diff = id, new_diff
                        else:
                            min_id = min_id if new_diff < min_diff else id
                            min_diff = min_diff if new_diff < min_diff else new_diff
                found = min_diff >= 0
            if min_id:
                if found:
                    self.free_.remove(min_id)
                    return min_id
                else:
                    self.memo_[prefix][min_id].byte_size -= min_diff
                    self.free_.remove(min_id)
                    return min_id
        id = f"{prefix}{len(self.memo_[prefix])}"
        self.memo_[prefix][id] = StorageNode(data_byte)
        return id

    def free(self, id):
        for prefix in self.pingpong:
            if prefix in id:
                self.free_.append(id)

    def set_address(self, base_addr_map=None):
        if base_addr_map is None:
            for prefix in self.memo_.keys():
                addr = None
                for id in self.memo_[prefix].keys():
                    addr = self.memo_[prefix][id].set_address(addr)
        else:
            saved_prefix_addr = {}
            self.sort_keys = []
            for prefix, addr in base_addr_map.items():
                if prefix not in self.memo_.keys():
                    continue
                self.sort_keys.append(prefix)
                tp_saved = []
                while True:
                    if isinstance(addr, str):
                        tp_saved.append(addr)
                        if addr in saved_prefix_addr:
                            addr = saved_prefix_addr[addr]
                            break
                        addr = base_addr_map[addr]
                        if addr in tp_saved:
                            raise RuntimeError("AssignAddress has loop design, please check")
                    else:
                        break
                for id in self.memo_[prefix].keys():
                    addr = self.memo_[prefix][id].set_address(addr)
                saved_prefix_addr[prefix] = addr
            if len(self.sort_keys) < len(self.memo_.keys()):
                unassign = ", ".join(list(self.memo_.keys() - self.sort_keys))
                raise RuntimeError(f"AssignAddress has no [{unassign}] address, please check")

    def get_address(self, id, offset, base_address=0):
        for memo_ in self.memo_.values():
            if id in memo_.keys():
                storage = memo_[id]
                return storage.get_address(offset, base_address)
        print("could not find " + id + " storage!")
        exit(-1)

    def get_byte_size(self, id):
        for memo_ in self.memo_.values():
            if id in memo_.keys():
                storage = memo_[id]
                return storage.byte_size
        print("could not find " + id + " storage!")
        exit(-1)

    def gen_source(self, code="cpp"):
        source = ""
        for prefix in self.sort_keys:
            memo = self.memo_[prefix]
            if code == "cpp":
                source += f"// {prefix} storage define\n"
                for id, storage in memo.items():
                    addr_hex = "0x%09x" % (storage.address)
                    source += f"uint64_t {id} = {addr_hex}; "
                    source += f"// storage size: {storage.byte_size} B\n"
            elif code == "py":
                source += f"# {prefix} storage define\n"
                for id, storage in memo.items():
                    addr_hex = "0x%09x" % (storage.address)
                    source += f"{id} = {addr_hex} "
                    source += f"# storage size: {storage.byte_size} B\n"
        return source[:-1]

    def __str__(self):
        ret = "********************************************\n"
        ret += "*{0:^13}|{1:^15}|{2:^12}*\n".format("Storage ID", "Address", "Size")
        if hasattr(self, "sort_keys"):
            for prefix in self.sort_keys:
                memo = self.memo_[prefix]
                ret += "*------------------------------------------*\n"
                for id, storage in memo.items():
                    ret += "*{0:^13}".format(id) + str(storage) + "\n"
            ret += "********************************************"
            return ret
        else:
            for prefix, memo in self.memo_.items():
                ret += "*------------------------------------------*\n"
                for id, storage in memo.items():
                    ret += "*{0:^13}".format(id) + str(storage) + "\n"
            ret += "********************************************"
            return ret


class GraphPlanMemory(Functor):

    def main(self, expr, init_addr, onchip=0, debug=1):
        self.debug = debug
        info = Functor()
        info.visit(expr)
        self.info_memo = info.memo
        self.onchip = onchip
        self.storage = Storage()
        expr = self.visit(expr)
        self.storage.set_address(init_addr)
        return expr, self.storage

    def _link_storage(self, arg, expr):
        if not hasattr(self, "linked"):
            self.linked = [[arg, expr]]
            return
        for index in range(len(self.linked)):
            if arg in self.linked[index]:
                self.linked[index].append(expr)
                return
        self.linked.append([arg, expr])

    def _check_free(self, arg):
        if hasattr(self, "linked"):
            for index, ll in enumerate(self.linked):
                if arg in ll:
                    for a in ll:
                        if self.info_memo[a][1] != 1:
                            self.info_memo[a][1] -= 1
                            return
                    self.storage.free(arg.checked_type.storage_id)
                    if hasattr(arg.checked_type, "onchip"):
                        self.storage.free(arg.checked_type.onchip)
                    del self.linked[index]
                    return
        if self.info_memo[arg][1] == 1:
            self.storage.free(arg.checked_type.storage_id)
            if hasattr(arg.checked_type, "onchip"):
                self.storage.free(arg.checked_type.onchip)
        else:
            self.info_memo[arg][1] -= 1

    def _malloc(self, tensor, prefix, kvcache=0, padding=False):
        bytesize = tensor.get_bytesize()
        if bytesize % 0x4000 and padding:
            bytesize = (bytesize // 0x4000 + 1) * 0x4000
        if kvcache:
            return self.storage.malloc(prefix, tensor.get_bytesize({"token":1}))
        elif tensor.dtype.mapped == DataEnum.ddr:
            return self.storage.malloc(prefix, bytesize)
        elif tensor.dtype.mapped == DataEnum.hbm:
            return self.storage.malloc(prefix, bytesize)
        else:
            print("unknown device, please check!")
            exit(-1)

    def visit_var(self, expr):
        storage_id = self._malloc(expr.checked_type, expr.prefix)
        expr.checked_type.storage_id = storage_id
        return expr

    def visit_constant(self, expr):
        storage_id = self._malloc(expr.checked_type, expr.prefix)
        expr.checked_type.storage_id = storage_id
        self.info_memo[expr][1] = 0
        return expr

    def visit_call(self, expr):
        new_args = []
        for arg in expr.args:
            new_args.append(self.visit(arg))
        not_last_op = True
        onchip = self.onchip == 1 and "kvcache" in expr.attrs and not_last_op and "runtime" == expr.prefix and expr.attrs.get("onchip", 1)
        if isinstance(expr.checked_type, Tuple):
            for i in range(len(expr.checked_type.tensors)):
                storage_id = self._malloc(expr.checked_type.tensors[i], expr.prefix)
                expr.checked_type.tensors[i].storage_id = storage_id
            if self.debug:
                tp_args = ", ".join([i.checked_type.storage_id for i in new_args])
                tp_outs = ", ".join([i.storage_id for i in expr.checked_type.tensors])
                log = f"{expr.op.name} [{tp_args}] -> [{tp_outs}]"
                LOG_WITH_PREFIX("graph", log)
        elif isinstance(expr.checked_type, Tensor):
            storage_id = self._malloc(expr.checked_type, expr.prefix)
            expr.checked_type.storage_id = storage_id
            extern_debug = ""
            if onchip:
                onchip_id = self._malloc(expr.checked_type, "onchip", kvcache=1)
                setattr(expr.checked_type, "onchip", onchip_id)
                setattr(expr.checked_type, "onchip_offset", 0)
                extern_debug = f"/{onchip_id}"
            if self.debug:
                tp_args = ", ".join([i.checked_type.storage_id for i in new_args])
                log = f"{expr.op.name} [{tp_args}] -> {expr.checked_type.storage_id}{extern_debug}"
                LOG_WITH_PREFIX("graph", log)
        else:
            print("infer_type first!")
            exit(-1)
        expr.args = new_args
        for arg in new_args:
            self._check_free(arg)
        return expr

    def visit_tupleitem(self, expr):
        arg = self.visit(expr.expr)
        expr.expr = arg
        expr.checked_type = arg.checked_type.tensors[expr.index]
        if isinstance(expr.expr, VM):
            self._link_storage(expr.expr, expr)
        return expr

    def visit_vm(self, expr):
        new_args = [self.visit(arg) for arg in expr.args]
        storage_id = new_args[0].checked_type.storage_id
        if isinstance(expr.checked_type, Tuple):
            for i in range(len(expr.checked_type.tensors)):
                expr.checked_type.tensors[i].storage_id = storage_id
                if hasattr(new_args[0].checked_type, "onchip"):
                    setattr(expr.checked_type.tensors[i], "onchip", new_args[0].checked_type.onchip)
                    setattr(expr.checked_type.tensors[i], "onchip_offset", 0)
        elif isinstance(expr.checked_type, Tensor):
            expr.checked_type.storage_id = storage_id
            if hasattr(new_args[0].checked_type, "onchip"):
                setattr(expr.checked_type, "onchip", new_args[0].checked_type.onchip)
                setattr(expr.checked_type, "onchip_offset", 0)
        else:
            print("infer_type first!")
            exit(-1)
        self._link_storage(new_args[0], expr)
        expr.args = new_args
        return expr


def graph_plan_memory(expr, init_addr):
    return GraphPlanMemory().main(expr, init_addr)
