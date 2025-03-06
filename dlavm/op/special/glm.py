from ...adr.base import Call, Op
from . import (
    _glm
)
from .attrs import *


def pos_emb(*args, **kwargs):
    return Call(Op.Get("glm.pos_emb"), args, Attrs(kwargs))
