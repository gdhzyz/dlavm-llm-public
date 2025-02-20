from ..graph import Graph
from ..operation import LayerNormOp, MulOp, PlaceholderOp, RsqrtOp, AddOp, MeanOp, PowOp
from .canonicalize import CanonicalizeBase, CanonicalizeManager
from .utils import check_list_types


@CanonicalizeManager.register(MulOp, 1)
class MulToLayerNormPat(CanonicalizeBase):

    def qsprt_add_mean_pow(self, g:Graph, op):
        if not isinstance(op, RsqrtOp):
            return False
        parents = [g.node_table[i] for i in op.parents]
        op = parents[0]
        if not isinstance(op, AddOp):
            return False
        parents = [g.node_table[i] for i in op.parents]
        self.eps = op._arguments[len(op.parents)]
        op = parents[0]
        if not isinstance(op, MeanOp):
            return False
        parents = [g.node_table[i] for i in op.parents]
        op = parents[0]
        if not isinstance(op, PowOp):
            return False
        if op._arguments[len(op.parents)] != 2:
            return False
        parents = [g.node_table[i] for i in op.parents]
        self.data = parents[0]
        return True

    def match(self, g:Graph, op: MulOp):
        parents = [g.node_table[i] for i in op.parents]
        _args = check_list_types(parents, [PlaceholderOp, MulOp])
        if len(_args) == 0:
            return False
        self.weight, mul1 = _args
        parents = [g.node_table[i] for i in mul1.parents]
        if self.qsprt_add_mean_pow(g, parents[0]) or self.qsprt_add_mean_pow(g, parents[1]):
            if self.data.name in mul1.parents:
                return True
        return False

    def rewrite(self, g:Graph, op: MulOp):
        parents = [g.node_table[i] for i in op.parents]
        new_op = LayerNormOp()
        new_op._name = "fuse_ln_" + op.name
        new_op._parents = [i.name for i in [self.data, self.weight]]
        new_op._arguments = new_op._parents + [self.eps]
        g.replace_users_with(op, new_op)
        g.delete_node(op, parents)
        for i in [self.data, self.weight]:
            i._children.append(new_op.name)
        g.add_node(new_op)
