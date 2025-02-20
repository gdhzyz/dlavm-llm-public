from time import strftime, localtime
from .. import ne
from ..utils.prototxt import Module, Value, List


class CodeGenPrototxt:

    tab = "  "

    def ext_define(self):
        self.step_id = 0

    def build(self, mod_name: str, module, storage, device=None):
        self.func_input = []
        self.func_output = []
        self.func_const_hbm = []
        self.func_const_ddr = []
        self.func_body = []
        self.enum_nodes = [[], [], [], []]
        self.dynamic_var = []
        self.ext_define()
        self.mod_name = mod_name
        self.storage = storage
        self.device = device
        self.prototxt = Module()
        self.prototxt.append(Value("name", mod_name))
        for node in module:
            if node["node"] == "var":
                self.gen_var(node)
            elif node["node"] == "output":
                self.gen_output(node)
            elif node["node"] == "const":
                self.gen_const(node)
            elif node["node"] == "accel_op":
                self.gen_accel(node)
            elif node["node"] == "cpu_op":
                self.gen_cpu(node)
            elif node["node"] == "virtual_op":
                self.gen_virtual(node)
            else:
                print("unkonwn node of module: ", node["node"])
                exit(-1)
        return self.gen_source()

    def gen_source(self):
        source = self.prototxt.export(self.tab)
        return source

    def gen_var(self, node):
        enum_name = node["name"]
        id, offset = node["storage"][0]["id"], node["storage"][0]["offset"]
        address = self.storage.get_address(id, offset)
        layer = List("layer")
        layer.append(Value("type", "input"))
        layer.append(Value("top", enum_name))
        layer.append(Value("name", "input::"+enum_name))
        layer_param = List("layer_param")
        layer_param.append(Value("shape", node["shape"]))
        layer_param.append(Value("address", address))
        layer.append(layer_param)
        self.prototxt.append(layer)

    def gen_output(self, node):
        enum_name, args = node["name"], node["args"]
        id, offset = node["storage"][0]["id"], node["storage"][0]["offset"]
        address = self.storage.get_address(id, offset)
        layer = List("layer")
        layer.append(Value("type", "output"))
        layer.append(Value("top", enum_name))
        for arg in args:
            layer.append(Value("bottom", arg))
        layer.append(Value("name", "output::"+enum_name))
        layer_param = List("layer_param")
        layer_param.append(Value("shape", node["shape"]))
        layer_param.append(Value("address", address))
        layer.append(layer_param)
        self.prototxt.append(layer)

    def gen_const(self, node):
        enum_name, data = node["name"], node.get("data", None)
        id, offset = node["storage"][0]["id"], node["storage"][0]["offset"]
        address = self.storage.get_address(id, offset)
        byte_size = self.storage.get_byte_size(id)
        device = "weight_ddr"
        if enum_name in self.enum_nodes[0] + self.enum_nodes[2] + self.enum_nodes[3]:
            print("*WARNING* : Var或Const节点中存在同名元素，请检查")
            exit(-1)
        if id[:3] == "ddr":
            pass
        elif id[:3] == "hbm":
            device = "weight_hbm"
            self.enum_nodes[3].append(enum_name)
        else:
            self.enum_nodes[2].append(enum_name)
        layer = List("layer")
        layer.append(Value("type", device))
        layer.append(Value("top", enum_name))
        layer.append(Value("name", device+"::"+enum_name))
        layer_param = List("layer_param")
        layer_param.append(Value("shape", node["shape"]))
        layer_param.append(Value("bytes", byte_size))
        layer_param.append(Value("address", address))
        if data:
            layer_param.append(Value("data", data))
        layer.append(layer_param)
        self.prototxt.append(layer)

    def gen_accel(self, node):
        enum_name, args, tensors = node["name"], node["args"], node["tensors"]
        op_name = node["op_name"]
        func_body = []
        for reg in node["csb_regs"]:
            if reg[0] == 1:
                if isinstance(reg[2], ne.Expr):
                    data = reg[2].export("cpp")
                else:
                    data = str(reg[2])
                func_body.append(f"CSB_Write({reg[1]}, {data})")
            elif reg[0] == 0:
                func_body.append(f"while(CSB_Read(device, {reg[1]}) != {reg[2]}) " + "{}")
        layer = List("layer")
        layer.append(Value("type", op_name))
        if len(node["storage"]) > 1:
            for n in range(len(node["storage"])):
                layer.append(Value("top", enum_name+f"_{n}"))
        else:
            layer.append(Value("top", enum_name))
        for arg in args:
            layer.append(Value("bottom", arg))
        layer.append(Value("name", "accel_op::"+enum_name))
        layer_param = List("layer_param")
        if len(node["storage"]) == 1:
            id, offset = node["storage"][0]["id"], node["storage"][0]["offset"]
            layer_param.append(Value("shape", node["output"][0].shape))
            layer_param.append(Value("address", self.storage.get_address(id, offset)))
            if hasattr(node["output"][0], "onchip"):
                onchip, onchip_offset = node["output"][0].onchip, 0
                layer_param.append(Value("onchip", self.storage.get_address(onchip, onchip_offset)))
        else:
            for n in range(len(node["storage"])):
                id, offset = node["output"][n][0].storage_id, node["output"][n][0].offset
                tensor_param = List(f"{enum_name}_{n}")
                tensor_param.append(Value("shape", node["output"][n][0].shape))
                tensor_param.append(Value("address", self.storage.get_address(id, offset)))
            layer_param.append(tensor_param)
        for k, v in node["attrs"].items():
            layer_param.append(Value(k, v))
        layer_param.append(Value("registers", func_body))
        layer.append(layer_param)
        self.prototxt.append(layer)

    def gen_cpu(self, node):
        pass

    def gen_virtual(self, node):
        enum_name, args, tensors = node["name"], node["args"], node["tensors"]
        op_name = node["op_name"]
        layer = List("layer")
        layer.append(Value("type", op_name))
        if len(node["storage"]) > 1:
            for n in range(len(node["storage"])):
                layer.append(Value("top", enum_name+f"_{n}"))
        else:
            layer.append(Value("top", enum_name))
        for arg in args:
            layer.append(Value("bottom", arg))
        layer.append(Value("name", "virtual_op::"+enum_name))
        layer_param = List("layer_param")
        if len(node["storage"]) == 1:
            id, offset = node["storage"][0]["id"], node["storage"][0]["offset"]
            layer_param.append(Value("shape", node["output"][0].shape))
            layer_param.append(Value("address", self.storage.get_address(id, offset)))
        else:
            for n in range(len(node["storage"])):
                id, offset = node["storage"][n]["id"], node["storage"][n]["offset"]
                tensor_param = List(f"{enum_name}_{n}")
                tensor_param.append(Value("shape", node["output"][n][0].shape))
                tensor_param.append(Value("address", self.storage.get_address(id, offset)))
            layer_param.append(tensor_param)
        for k, v in node["attrs"].items():
            layer_param.append(Value(k, v))
        layer.append(layer_param)
        self.prototxt.append(layer)