import array, copy
from typing import Dict, List, Tuple, Union
import numpy
import sys

from ..graph import TensorDType
from ..graph import *

from dlavm import op as dlop


class AdrConvert:

    @classmethod
    def get_converter(cls, opset, target=""):
        """Get converter matches given opset.

        Parameters
        ----------
        opset: int
            opset from model.

        Returns
        -------
        converter, which should be `_impl_vx`. Number x is the biggest
            number smaller than or equal to opset belongs to all support versions.
        """
        func_name = f"_impl_{target}_v" if len(target) else "_impl_v"
        versions = [int(d.replace(func_name, "")) for d in dir(cls) if func_name in d]
        versions = sorted(versions + [opset])
        version = versions[max([i for i, v in enumerate(versions) if v == opset]) - 1]
        if hasattr(cls, f"{func_name}{version}"):
            return getattr(cls, f"{func_name}{version}")
        raise NotImplementedError(
            "opset version {} of {} on {} not implemented".format(version, cls.__name__, target)
        )


class Embedding(AdrConvert):

    @classmethod
    def _impl_v(cls, inputs, op):
        return dlop.var_ddr("embedding", op.tensor_meta["shape"])


class Transpose(AdrConvert):

    @classmethod
    def _impl_v(cls, inputs, op):
        return dlop.var_ddr("embedding", op.tensor_meta["shape"])


