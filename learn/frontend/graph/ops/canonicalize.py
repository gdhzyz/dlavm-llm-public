from ..graph import Graph
from ..operation import *

class CanonicalizeBase:

    def match(self, g:Graph, op: Op):
        pass

    def rewrite(self, g:Graph, op: Op):
        pass

class CanonicalizeManager:

    opt_map_pats = {}
    opt_keys = []

    @classmethod
    def register(cls, op: Op, level: int):
        def _register(obj: CanonicalizeBase):
            if op in cls.opt_keys:
                cls.opt_map_pats[op].append([level, obj])
            else:
                cls.opt_keys.append(op)
                cls.opt_map_pats[op] = [[level, obj]]
        return _register
