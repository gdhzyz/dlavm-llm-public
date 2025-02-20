from .. import Graph
from ..ops import CanonicalizeManager as cm


def canonicalize(graph: Graph):
    for i in range(2):
        todo_ops = []
        for node in graph.body:
            if type(node) in cm.opt_keys:
                for [level, pat] in cm.opt_map_pats[type(node)]:
                    if level == i:
                        obj = pat()
                        if obj.match(graph, node):
                            todo_ops.append([obj, node])
                            pass
        for obj, node in todo_ops:
            obj.rewrite(graph, node)

    
