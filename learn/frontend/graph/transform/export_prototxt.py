from .. import Graph
from ..operation import *
from ...utils import prototxt as pt


def export_prototxt(graph: Graph):
    """
    Fuse the maxpool op and getitem op to simpllify graph.

    Args:
        graph (Graph): The Graph to be simplified.
    """
    module = pt.Module()
    module.append(pt.Value("name", graph.func_name))
    for i, node in enumerate(graph.body):
        layer = pt.List("layer")
        layer.append(pt.Value("type", type(node).__name__))
        layer.append(pt.Value("top", node.name))
        attrs = pt.List("layer_param")
        for arg in node.parents:
            layer.append(pt.Value("bottom", arg))
        for index in range(len(node.args) - len(node.parents)):
            arg = node.args[index + len(node.parents)]
            attrs.append(pt.Value(f"attr_{index}", arg))
        if len(node.tensor_meta):
            shape = node.tensor_meta["shape"]
            dtype = str(node.tensor_meta["dtype"])
            if len(shape) and isinstance(shape[0], list):
                l_shape = pt.List("shape")
                for i, v in enumerate(shape):
                    l_shape.append(pt.Value(f"output_{i}", v))
                attrs.append(l_shape)
            else:
                attrs.append(pt.Value("shape", shape))
                attrs.append(pt.Value("dtype", dtype))
        if len(attrs.value):
            layer.append(attrs)
        module.append(layer)
    return module.export()


