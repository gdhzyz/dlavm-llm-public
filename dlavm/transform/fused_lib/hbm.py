import numpy as np
from .register import ConvertCall, FusedStrategy
from ...adr import Constant, Call, Op, hbm


def bias2bn(const: Constant):
    bias_shape = const.shape[0]
    new_shape = [bias_shape*2]
    new_data = [np.zeros((bias_shape), dtype="float16"), const.data]
    return Constant(const.name, new_data, new_shape, const.dtype, const.device)


def hbm_mvm_bn(add, mvm, bn):
    data, weight = mvm[0]
    new_expr = Call(Op.Get("accel.hbm.mvm_bn"), [data, weight, bn], {"padding": 0, **mvm[1]})
    new_expr.checked_type = add.checked_type
    return new_expr


FusedStrategy.Register("accel.hbm.add", 
    ConvertCall(
        "accel.hbm.add",
        [
            ConvertCall(
                "accel.hbm.mvm",
                [
                    lambda x, y: True,
                    lambda x, y: True
                ],
                lambda x: [x.args, x.attrs]
            ),
            [
                lambda x, y: x.shape[0] == y.shape[-1] and len(x.shape) == 1, 
                bias2bn
            ]
        ],
        hbm_mvm_bn
    )
)


def hbm_mvm_bn_res(add, mvm_bn, res):
    data, weight, bn = mvm_bn[0]
    new_expr = Call(Op.Get("accel.hbm.mvm_bn_res"), [data, weight, bn, res], mvm_bn[1])
    new_expr.checked_type = add.checked_type
    return new_expr


FusedStrategy.Register("accel.hbm.add", 
    ConvertCall(
        "accel.hbm.add",
        [
            ConvertCall(
                "accel.hbm.mvm_bn",
                [
                    lambda x, y: True,
                    lambda x, y: True,
                    lambda x, y: True,
                ],
                lambda x: [x.args, x.attrs]
            ),
            [
                lambda x, y: x.shape == y.shape, 
                lambda x: x
            ]
        ],
        hbm_mvm_bn_res
    )
)


def hbm_mvm_res(add, mvm, res):
    data, weight = mvm[0]
    attrs = mvm[1]
    attrs["arg_max"] = 0
    shape = add.checked_type.shape
    bn = hbm.const_ddr(f"global::bnzeros::{shape[-1]}", [np.ones((shape[-1]), dtype="float16"), np.zeros((shape[-1]), dtype="float16")], [shape[-1]*2])
    new_expr = Call(Op.Get("accel.hbm.mvm_bn_res"), [data, weight, bn, res], attrs)
    new_expr.checked_type = add.checked_type
    return new_expr


FusedStrategy.Register("accel.hbm.add", 
    ConvertCall(
        "accel.hbm.add",
        [
            ConvertCall(
                "accel.hbm.mvm",
                [
                    lambda x, y: True,
                    lambda x, y: True,
                ],
                lambda x: [x.args, x.attrs]
            ),
            [
                lambda x, y: x.shape == y.shape, 
                lambda x: x
            ]
        ],
        hbm_mvm_res
    ),
    opt_level=2,
    reverse=True
)
