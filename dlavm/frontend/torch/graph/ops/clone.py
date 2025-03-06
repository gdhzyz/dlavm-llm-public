from ..graph import Graph
from ..operation import CloneOp
from .canonicalize import CanonicalizeBase, CanonicalizeManager


@CanonicalizeManager.register(CloneOp, 0)
class RemoveClonePat(CanonicalizeBase):

    def match(self, g:Graph, op: CloneOp):
        return True

    def rewrite(self, g:Graph, op: CloneOp):
        parents = [g.node_table[i] for i in op.parents]
        g.replace_users_with(op, parents[0])
        g.delete_node(op, parents)
