from ..adr import Functor, VM, Tensor, Tuple, DataEnum
from .graph_build import GraphBuild 
from dlavm.driver import Tasks, ir, transform
from dlavm import ne
from ..utils.prototxt import Module, Value, List

class VisualizePrototxt:

    tab = "  "

    def build(self, graphs, mod_name):
        self.enum_nodes = [[], [], [], []]
        self.prototxt = Module()
        self.prototxt.append(Value("name", mod_name))
        for node in graphs:
            if node["type"] == "variable":
                self.gen_var(node)
            elif node["type"] == "output":
                self.gen_output(node)
            elif node["type"] == "constant":
                self.gen_const(node)
            elif node["type"] == "accelop":
                self.gen_accel(node)
            elif node["type"] == "cpuop":
                self.gen_cpu(node)
            elif node["type"] == "virtualop":
                self.gen_virtual(node)
            else:
                print("unkonwn node of module: ", node["type"])
                exit(-1)
        return self.gen_source()

    def gen_source(self):
        source = self.prototxt.export(self.tab)
        return source

    def gen_var(self, node):
        enum_name = node["name"]
        id, offset = node["tensor"].storage_id, node["tensor"].offset
        shape = node["tensor"].shape
        layer = List("layer")
        layer.append(Value("type", "input"))
        layer.append(Value("top", enum_name))
        layer.append(Value("name", "input::"+enum_name))
        layer_param = List("layer_param")
        layer_param.append(Value("shape", shape))
        layer_param.append(Value("address", id))
        layer_param.append(Value("offset", offset))
        layer.append(layer_param)
        self.prototxt.append(layer)

    def gen_output(self, node):
        enum_name, args = node["name"], node["args"]
        id, offset = node["tensor"].storage_id, node["tensor"].offset
        shape = node["tensor"].shape
        layer = List("layer")
        layer.append(Value("type", "output"))
        layer.append(Value("top", enum_name))
        layer.append(Value("bottom", args))
        layer.append(Value("name", "output::"+enum_name))
        layer_param = List("layer_param")
        layer_param.append(Value("shape", shape))
        layer_param.append(Value("address", id))
        layer_param.append(Value("offset", offset))
        layer.append(layer_param)
        self.prototxt.append(layer)

    def gen_const(self, node):
        enum_name, data = node["ir_name"], node.get("data", None)
        id, offset = node["tensor"].storage_id, node["tensor"].offset
        shape = node["tensor"].shape
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
        layer_param.append(Value("shape", shape))
        layer_param.append(Value("address", id))
        layer_param.append(Value("offset", offset))
        if data:
            layer_param.append(Value("data", data))
        layer.append(layer_param)
        self.prototxt.append(layer)

    def gen_accel(self, node):
        enum_name, args, checked_type = node["ir_name"], node["args"], node["tensor"]
        op_name = node["op_name"]

        layer = List("layer")
        layer.append(Value("type", op_name))
        if isinstance(checked_type, Tuple):
            for n in range(len(checked_type.tensors)):
                layer.append(Value("top", enum_name+f"_{n}"))
        else:
            layer.append(Value("top", enum_name))
        for arg in args:
            layer.append(Value("bottom", arg))
        layer.append(Value("name", "accel_op::"+enum_name))
        layer_param = List("layer_param")
        if isinstance(checked_type, Tuple):
            for n, tensor in enumerate(checked_type.tensors):
                id, offset = tensor.storage_id, tensor.offset
                tensor_param = List(f"{enum_name}_{n}")
                tensor_param.append(Value("shape", tensor.shape))
                tensor_param.append(Value("address", id))
                tensor_param.append(Value("offset", offset))
            layer_param.append(tensor_param)
        elif isinstance(checked_type, Tensor):
            id, offset = checked_type.storage_id, checked_type.offset
            layer_param.append(Value("shape", checked_type.shape))
            layer_param.append(Value("address", id))
            layer_param.append(Value("offset", offset))
        for k, v in node["attrs"].items():
            layer_param.append(Value(k, v))
        layer.append(layer_param)
        self.prototxt.append(layer)

    def gen_cpu(self, node):
        pass

    def gen_virtual(self, node):
        enum_name, args, checked_type = node["ir_name"], node["args"], node["tensor"]
        op_name = node["op_name"]
        layer = List("layer")
        layer.append(Value("type", op_name))
        if isinstance(checked_type, Tuple):
            for n in range(len(checked_type.tensors)):
                layer.append(Value("top", enum_name+f"_{n}"))
        else:
            layer.append(Value("top", enum_name))
        for arg in args:
            layer.append(Value("bottom", arg))
        layer.append(Value("name", "virtual_op::"+enum_name))
        layer_param = List("layer_param")
        if isinstance(checked_type, Tuple):
            for n, tensor in enumerate(checked_type.tensors):
                id, offset = tensor.storage_id, tensor.offset
                tensor_param = List(f"{enum_name}_{n}")
                tensor_param.append(Value("shape", tensor.shape))
                tensor_param.append(Value("address", id))
                tensor_param.append(Value("offset", offset))
            layer_param.append(tensor_param)
        elif isinstance(checked_type, Tensor):
            id, offset = checked_type.storage_id, checked_type.offset
            layer_param.append(Value("shape", checked_type.shape))
            layer_param.append(Value("address", id))
            layer_param.append(Value("offset", offset))
        for k, v in node["attrs"].items():
            layer_param.append(Value(k, v))
        layer.append(layer_param)
        self.prototxt.append(layer)
