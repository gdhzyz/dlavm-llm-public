from ...base import Call, Op
from . import (
    _nn,
    _ohbm
)
from .attrs import *


def mvm(*args, **kwargs):
    return Call(Op.Get("nn.mvm"), args, MVMAttrs(kwargs))

