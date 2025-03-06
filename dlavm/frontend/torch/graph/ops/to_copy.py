from ..graph import Graph
from ..operation import ToCopyOp
from .canonicalize import CanonicalizeBase, CanonicalizeManager


@CanonicalizeManager.register(ToCopyOp, 0)
class RemoveToCopyPat(CanonicalizeBase):

    def match(self, g:Graph, op: ToCopyOp):
        return True

    def rewrite(self, g:Graph, op: ToCopyOp):
        parents = [g.node_table[i] for i in op.parents]
        g.replace_users_with(op, parents[0])
        g.delete_node(op, parents)
