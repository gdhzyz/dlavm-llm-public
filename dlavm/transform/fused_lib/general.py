import numpy as np
from .register import ConvertCall, FusedStrategy
from ...adr import Constant, Call, Op, hbm, general


def fused_reshape(root, leaf):
    new_shape = root.checked_type.shape
    new_expr = general.reshape(leaf[0][0], new_shape)
    new_expr.checked_type = root.checked_type
    return new_expr


FusedStrategy.Register("accel.reshape", 
    ConvertCall(
        "accel.reshape",
        [
            ConvertCall(
                "accel.reshape",
                [
                    lambda x, y: True,
                ],
                lambda x: [x.args, x.attrs]
            ),
        ],
        fused_reshape
    ),
    opt_level=2
)

