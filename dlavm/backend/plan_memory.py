from dlavm import ne
from dlavm.driver import ir


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

    def export(self):
        with ir.Block() as source:
            for prefix in self.sort_keys:
                memo = self.memo_[prefix]
                source += ir.Annotation(f"{prefix} storage define")
                for id, storage in memo.items():
                    addr_hex = "0x%09x" % (storage.address)
                    source += ir.Assign(str(id), addr_hex, "uint64_t")
        return source

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