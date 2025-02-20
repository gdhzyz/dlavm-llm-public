# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, import-self, len-as-condition, unused-argument, too-many-lines
# pylint: disable=import-outside-toplevel
"""ONNX: Open Neural Network Exchange frontend for Relay."""
import copy
import warnings
import numpy as np
from .. import adr
from ..transform import infer_type
from .. import ne

__all__ = ["from_onnx"]


class onnx_input:
    """ Dual purpose list or dictionary access object."""

    def __init__(self):
        self.input_keys = []
        self.input_dict = {}

    def __getitem__(self, item):
        if isinstance(item, int):
            if item > (len(self.input_keys) - 1):
                return None
            return self.input_dict[self.input_keys[item]]
        if isinstance(item, str):
            if item not in self.input_keys:
                return None
            return self.input_dict[item]
        if isinstance(item, slice):
            keys = self.input_keys[item]
            return [self.input_dict[key] for key in keys]

        raise ValueError("Only integer, string, and slice accesses allowed.")

    def __setitem__(self, item, value):
        if isinstance(item, int):
            self.input_dict[self.input_keys[item]] = value
        elif isinstance(item, str):
            self.input_keys.append(item)
            self.input_dict[item] = value
        else:
            raise ValueError("Only integer and string indexed writes allowed.")

    def keys(self):
        return self.input_keys

    def __len__(self):
        return len(self.input_keys)

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self.input_keys):
            output = self.input_dict[self.input_keys[self.n]]
            self.n += 1
            return output

        raise StopIteration


def get_numpy(tensor_proto):
    """Grab data in TensorProto and convert to numpy array."""
    try:
        from onnx.numpy_helper import to_array
    except ImportError as e:
        raise ImportError("Unable to import onnx which is required {}".format(e))
    return to_array(tensor_proto)


def get_type(elem_type):
    """Converts onnx integer datatype to numpy datatype"""
    try:
        from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
    except ImportError as e:
        raise ImportError("Unable to import onnx which is required {}".format(e))

    return str(TENSOR_TYPE_TO_NP_TYPE[elem_type])


def get_info(info_proto):
    """Extract the shape from a ValueInfoProto."""
    shape = []
    shape_name = []
    for dim in info_proto.type.tensor_type.shape.dim:
        name = dim.dim_param
        value = dim.dim_value
        if value is None or value == 0:
            value = ne.Var(name)
            shape_name.append(name)
        else:
            shape_name.append(value)
        shape.append(value)

    name = info_proto.name
    dtype = get_type(info_proto.type.tensor_type.elem_type)
    return name, shape, dtype, shape_name


def dimension_picker(prefix, suffix=""):
    """Check that dimensions are supported."""

    def _impl(attr):
        kernel = attr["kernel_shape"]
        if len(kernel) == 1:
            return prefix + "1d" + suffix
        if len(kernel) == 2:
            return prefix + "2d" + suffix
        if len(kernel) == 3:
            return prefix + "3d" + suffix
        msg = "Only 1D, 2D, and 3D kernels are supported for operator {}."
        op_name = prefix + "1d/2d/3d"
        raise RuntimeError(msg.format(op_name))

    return _impl


def revert_caffe2_pad(pads):
    """Caffe2 requires two times the normal padding."""
    if len(pads) == 4:
        pads = pads[:2]
    elif len(pads) == 2:
        pass
    else:
        raise RuntimeError("Number of pads must be either 2 or 4.")
    return pads


def get_pad_pair(input1d, kernel1d, stride1d, mode):
    """infer pad size"""
    if input1d % stride1d == 0:
        pad = max(kernel1d - stride1d, 0)
    else:
        pad = max(kernel1d - (input1d % stride1d), 0)
    pad_before = pad // 2
    pad_after = pad - pad_before
    if "LOWER" in mode:
        return [pad_after, pad_before]
    return [pad_before, pad_after]


def onnx_default_layout(dims, op_name):
    if dims == 1:
        return "NCW"
    if dims == 2:
        return "NCHW"
    if dims == 3:
        return "NCDHW"

    msg = "Only 1D, 2D and 3D layouts are currently supported for operator {}."
    raise RuntimeError(msg.format(op_name))


def onnx_storage_order2layout(storage_order, dims, op_name):
    """converter of onnx storage order parameter to tvm storage order format"""
    if storage_order not in (0, 1):
        raise RuntimeError("Mode of storage_order must be either 0 or 1")

    if dims == 1:
        return "NCW" if storage_order == 0 else "NWC"
    if dims == 2:
        return "NCHW" if storage_order == 0 else "NHWC"
    if dims == 3:
        return "NCDHW" if storage_order == 0 else "NDHWC"

    msg = "Only 1D, 2D and 3D layouts are currently supported for operator {}."
    raise RuntimeError(msg.format(op_name))


def dimension_constraint():
    def _dim_check(attrs):
        if len(attrs["kernel_shape"]) in [1, 2, 3]:
            return True
        return False

    return _dim_check, "Only 1d, 2d and 3d kernel supported."


class OnnxOpConverter(object):
    """A helper class for holding onnx op converters."""

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


class MatMul(OnnxOpConverter):
    """Operator converter for MatMul."""

    @classmethod
    def _impl_hbm_v1(cls, inputs, attr, params):
        assert len(inputs) == 2, "MatMul op take 2 inputs, {} given".format(len(inputs))
        # When performing a batch matmul, we need to properly handle N-dim shapes.
        return adr.hbm.mvm(inputs[0], inputs[1])


class QuantLinear(OnnxOpConverter):
    """Operator converter for MatMul."""

    @classmethod
    def _impl_hbm_v1(cls, inputs, attr, params):
        assert len(inputs) == 3, "QuantLinear op take 3 inputs, {} given".format(len(inputs))
        # When performing a batch matmul, we need to properly handle N-dim shapes.
        weight_data = [inputs[1].data.transpose(), inputs[2].data.transpose()]
        weight_shape = list(inputs[1].data.shape)
        weight_shape = [weight_shape[0]*8, weight_shape[1]]
        weight = adr.hbm.const_hbm(inputs[1].name, weight_data, shape=weight_shape)
        return adr.hbm.mvm(inputs[0], weight)


class Constant(OnnxOpConverter):
    """Operator converter for Constant."""

    @classmethod
    def _impl_hbm_v1(cls, inputs, attr, params):
        if "value" not in attr:
            raise RuntimeError("no value in Constant")
        value = attr.pop("value")
        # Constants may rarely have string types. These are likely exported
        # from other frameworks and not actually used in TVM. We'll just use
        # a zero valued constant for compatibility.
        if isinstance(value, bytes):
            np_value = np.asarray([0]).astype("int")
        else:
            np_value = get_numpy(value)
        if isinstance(np_value, int):
            return np_value
        elif np_value.size == 1:
            if np_value.dtype == "int":
                return int(np_value)
        return list(np_value)

    @classmethod
    def _impl_v9(cls, inputs, attr, params):
        if "value" not in attr:
            raise RuntimeError("no value in Constant")
        value = attr.pop("value")
        # Constants may rarely have string types. These are likely exported
        # from other frameworks and not actually used in TVM. We'll just use
        # a zero valued constant for compatibility.
        if isinstance(value, bytes):
            np_value = np.asarray([0]).astype("int64")
        else:
            np_value = get_numpy(value)
        dtype = np_value.dtype.name
        # value = _expr.const(np_value, dtype)
        # return value


class ArgMax(OnnxOpConverter):
    """Operator converter for ArgMax."""

    @classmethod
    def _impl_hbm_v13(cls, inputs, attr, params):
        return inputs[0]


class Reshape(OnnxOpConverter):
    """Operator converter for Reshape."""

    @classmethod
    def _impl_hbm_v1(cls, inputs, attr, params):
        new_shape = [int(i.astype(int)) if isinstance(i, np.int64) else i for i in inputs[1]]
        return adr.reshape(inputs[0], new_shape)


class LayerNorm(OnnxOpConverter):
    """Operator converter for LayerNorm."""

    @classmethod
    def _impl_hbm_v1(cls, inputs, attr, params):
        weight_shape = list(inputs[1].data.shape)
        weight_data = [inputs[1].data, np.zeros(weight_shape)]
        weight_shape[-1] = weight_shape[-1]*2
        weight = adr.hbm.const_ddr(inputs[1].name, weight_data, shape=weight_shape)
        return adr.hbm.layer_norm(inputs[0], weight)


class RotaryPosEmb(OnnxOpConverter):
    """Operator converter for LayerNorm."""

    @classmethod
    def _impl_hbm_v1(cls, inputs, attr, params):
        if isinstance(inputs[1], adr.Var):
            weight = adr.hbm.var_ddr(inputs[1].name, [inputs[1].shape[1], inputs[1].shape[0]])
        else:
            weight_shape = list(inputs[1].data.shape)
            weight_data = inputs[1].data
            weight = adr.hbm.const_ddr(inputs[1].name, weight_data, shape=weight_shape)
        return adr.hbm.pos_emb(inputs[0], weight)

    @classmethod
    def _impl_hbm_v2(cls, inputs, attr, params):
        if isinstance(inputs[1], adr.Var):
            weight = adr.onnx_transpose(inputs[1], [1, 0])
        else:
            weight_shape = list(inputs[1].data.shape)
            weight_data = inputs[1].data
            weight = adr.hbm.const_ddr(inputs[1].name, weight_data, shape=weight_shape)
        data = adr.onnx_transpose(inputs[0], [1, 2, 0, 3])
        output = adr.hbm.pos_emb(data, weight)
        return adr.onnx_transpose(output, [2, 0, 1, 3])


class Shape(OnnxOpConverter):
    """Operator converter for Shape."""

    @classmethod
    def _impl_hbm_v1(cls, inputs, attr, params):
        output = infer_type(inputs[0])
        shape = output.checked_type.shape
        return shape


class ConstantOfShape(OnnxOpConverter):
    """Operator converter for Shape."""

    @classmethod
    def _impl_hbm_v1(cls, inputs, attr, params):
        value = get_numpy(attr["value"])[0]
        data = np.zeros(inputs[0])
        return data + value


class Mul(OnnxOpConverter):
    """Operator converter for Shape."""

    @classmethod
    def _impl_hbm_v1(cls, inputs, attr, params):
        if isinstance(inputs[0], (int, np.ndarray)) and isinstance(inputs[1], (int, np.ndarray)):
            data0, data1 = inputs[0], inputs[1]
            return data0 * data1
        else:
            return adr.hbm.mul(inputs[0], inputs[1])


class Add(OnnxOpConverter):
    """Operator converter for Shape."""

    @classmethod
    def _impl_hbm_v1(cls, inputs, attr, params):
        if isinstance(inputs[0], (int, np.ndarray)) and isinstance(inputs[1], (int, np.ndarray)):
            data0, data1 = inputs[0], inputs[1]
            return data0 + data1
        else:
            data0, data1 = inputs[0], inputs[1]
            if isinstance(inputs[1], adr.Constant):
                data1 = adr.hbm.const_ddr(inputs[1].name, inputs[1].data, inputs[1].data.shape)
            return adr.hbm.add(data0, data1)


class Div(OnnxOpConverter):
    """Operator converter for Shape."""

    @classmethod
    def _impl_hbm_v1(cls, inputs, attr, params):
        if isinstance(inputs[0], (int, np.ndarray)) and isinstance(inputs[1], (int, np.ndarray)):
            data0, data1 = inputs[0], inputs[1]
            return data0 // data1
        else:
            print("Div Expr Error")
            exit(-1)


class Equal(OnnxOpConverter):
    """Operator converter for Shape."""

    @classmethod
    def _impl_hbm_v1(cls, inputs, attr, params):
        data1, data2 = inputs[0], inputs[1]
        data1 = data1 if isinstance(data1, np.ndarray) else np.array(data1)
        data2 = data2 if isinstance(data2, np.ndarray) else np.array(data2)
        return data1 == data2


class Where(OnnxOpConverter):
    """Operator converter for Shape."""

    @classmethod
    def _impl_hbm_v1(cls, inputs, attr, params):
        condition, data1, data2 = inputs[0], inputs[1], inputs[2]
        condition = condition if isinstance(condition, np.ndarray) else np.array(condition)
        data1 = data1 if isinstance(data1, np.ndarray) else np.array(data1)
        data2 = data2 if isinstance(data2, np.ndarray) else np.array(data2)
        return np.where(condition, data1, data2)


class Expand(OnnxOpConverter):
    """Operator converter for Shape."""

    @classmethod
    def _impl_hbm_v1(cls, inputs, attr, params):
        data, shape = inputs[0], inputs[1]
        return adr.onnx_expand(data, shape)


class Transpose(OnnxOpConverter):
    """Operator converter for Shape."""

    @classmethod
    def _impl_hbm_v1(cls, inputs, attr, params):
        data = inputs[0]
        perm = attr["perm"]
        return adr.onnx_transpose(data, perm)


class Attention(OnnxOpConverter):
    """Operator converter for Shape."""

    @classmethod
    def _impl_hbm_v1(cls, inputs, attr, params):
        return adr.onnx_attention(inputs[0], inputs[1], inputs[2])


class Gather(OnnxOpConverter):
    """Operator converter for Shape."""

    @classmethod
    def _impl_hbm_v1(cls, inputs, attr, params):
        axis = attr["axis"]
        data = np.array(inputs[0])
        indices = np.array([inputs[1]])
        return np.take_along_axis(data, indices, axis)


class Slice(OnnxOpConverter):
    """Operator converter for Shape."""

    @classmethod
    def _impl_hbm_v1(cls, inputs, attr, params):
        return adr.onnx_slice(inputs[0], inputs[1], inputs[2], inputs[3])


class Sigmoid(OnnxOpConverter):
    """Operator converter for Shape."""

    @classmethod
    def _impl_hbm_v1(cls, inputs, attr, params):
        return adr.onnx_sigmoid(inputs[0])


class Silu(OnnxOpConverter):
    """Operator converter for Shape."""

    @classmethod
    def _impl_hbm_v1(cls, inputs, attr, params):
        return adr.hbm.silu(inputs[0])


class Unsqueeze(OnnxOpConverter):
    """Operator converter for Shape."""

    @classmethod
    def _impl_hbm_v1(cls, inputs, attr, params):
        if isinstance(inputs[0], (list, np.ndarray)):
            data = np.array(inputs[0])
            axes = [inputs[1]]
            return np.expand_dims(data, axes)[0]
        elif isinstance(inputs[0], adr.Base):
            output = infer_type(inputs[0])
            shape = [i for i in output.checked_type.shape]
            if isinstance(inputs[1], int):
                shape.insert(inputs[1], 1)
            else:
                for i in inputs[1]:
                    shape.insert(i, 1)
            return adr.reshape(inputs[0], shape)
        else:
            print(type(inputs[0]))

    @classmethod
    def _impl_sparse_v1(cls, inputs, attr, params):
        if isinstance(inputs[0], (list, np.ndarray)):
            data = np.array(inputs[0])
            axes = [inputs[1]]
            return np.expand_dims(data, axes)[0]
        elif isinstance(inputs[0], adr.Base):
            output = infer_type(inputs[0])
            shape = [i for i in output.checked_type.shape]
            if isinstance(inputs[1], int):
                shape.insert(inputs[1], 1)
            else:
                for i in inputs[1]:
                    shape.insert(i, 1)
            return adr.reshape(inputs[0], shape)
        else:
            print(type(inputs[0]))




class Concat(OnnxOpConverter):
    """Operator converter for Shape."""

    @classmethod
    def _impl_hbm_v1(cls, inputs, attr, params):
        datas = [np.array([i]) if isinstance(i, int) else np.array(i) for i in inputs]
        axis = attr["axis"]
        return np.concatenate(datas, axis)


class Split(OnnxOpConverter):
    """Operator converter for Shape."""

    @classmethod
    def _impl_hbm_v1(cls, inputs, attr, params):
        axis = attr["axis"]
        return adr.split(inputs[0], axis, inputs[1])


# compatible operators that do NOT require any conversion.
_identity_list = []


# _convert_map defines maps of name to converter functor(callable)
# for 1 to 1 mapping, use Renamer if nothing but name is different
# use AttrCvt if attributes need to be converted
# for 1 to N mapping(composed), use custom callable functions
# for N to 1 mapping, currently not supported(?)
def _get_convert_map(opset, target):
    return {
        "MatMul": MatMul.get_converter(opset, target),
        "LayerNorm": LayerNorm.get_converter(opset, target),
        "Shape": Shape.get_converter(opset, target),
        "ConstantOfShape": ConstantOfShape.get_converter(opset, target),
        "Mul": Mul.get_converter(opset, target),
        "Add": Add.get_converter(opset, target),
        "Div": Div.get_converter(opset, target),
        "Equal": Equal.get_converter(opset, target),
        "Where": Where.get_converter(opset, target),
        "Expand": Expand.get_converter(opset, target),
        "Transpose": Transpose.get_converter(opset, target),
        "Attention": Attention.get_converter(opset, target),
        "Gather": Gather.get_converter(opset, target),
        "Slice": Slice.get_converter(opset, target),
        "Sigmoid": Sigmoid.get_converter(opset, target),
        "Silu": Silu.get_converter(opset, target),
        "Unsqueeze": Unsqueeze.get_converter(opset, target),
        "Concat": Concat.get_converter(opset, target),
        "ArgMax": ArgMax.get_converter(opset, target),
        "Constant": Constant.get_converter(opset, target),
        "Split": Split.get_converter(opset, target),
        "QuantLinear": QuantLinear.get_converter(opset, target),
        "RotaryPosEmb": RotaryPosEmb.get_converter(opset, target),
        "Reshape": Reshape.get_converter(opset, target),
    }


class GraphProto:
    """A helper class for handling Relay expression copying from pb2.GraphProto.
    Definition: https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

        Parameters
    ----------
    shape : dict of str to tuple, optional
        The input shape to the graph

    dtype : str or dict of str to str
        The input types to the graph

    freeze_params: bool
        If this parameter is true, the importer will take any provided
        onnx input values (weights, shapes, etc) and embed them into the relay model
        as Constants instead of variables. This allows more aggressive optimizations
        at compile time and helps in making models static if certain inputs represent
        attributes relay would traditionally consider compile-time constants.

    """

    current = None

    def __init__(self, shape, dtype, freeze_params=False):
        self._nodes = {}
        self._params = {}
        self._inputs = {}
        self._renames = {}
        self._num_input = 0
        self._num_param = 0
        self._shape = shape if shape else {}
        self._dtype = dtype
        self.opset = None
        self._freeze_params = freeze_params

    def __enter__(self):
        self._old_manager = GraphProto.current
        GraphProto.current = self
        return self

    def __exit__(self, ptype, value, trace):
        GraphProto.current = self._old_manager

    def freeze(self, func, params):
        bind_map = {}
        for name in params.keys():
            if name in self._nodes.keys():
                bind_map[self._nodes[name]] = adr.Constant(name, params[name])
        return bind_map, {}

    def from_onnx(self, graph, opset, target="hbm"):
        """Construct Relay expression from ONNX graph.

        Onnx graph is a python protobuf object.
        The companion parameters will be handled automatically.
        However, the input names from onnx graph is vague, mixing inputs and
        network weights/bias such as "1", "2"...
        For convenience, we rename the `real` input names to "input_0",
        "input_1"... And renaming parameters to "param_0", "param_1"...

        Parameters
        ----------
        graph : onnx protobuf object
            The loaded onnx graph

        opset : opset version

        get_output_expr: bool
            If set to true, this conversion will return each output expression rather
            than a packaged module. This can be useful when converting subgraphs to
            relay.

        Returns
        -------
        mod : tvm.IRModule
            The returned relay module

        params : dict
            A dict of name: tvm.nd.array pairs, used as pretrained weights
        """
        var_ddr =  {
            "hbm": adr.hbm.var_ddr,
        }
        self.opset = opset
        # parse network inputs to relay, aka parameters
        for init_tensor in graph.initializer:
            if not init_tensor.name.strip():
                raise ValueError("Tensor's name is required.")
            array = self._parse_array(init_tensor)
            if self._freeze_params:
                self._nodes[init_tensor.name] = adr.Constant(init_tensor.name, array)
            else:
                self._params[init_tensor.name] = array
                self._nodes[init_tensor.name] = var_ddr[target](
                    init_tensor.name,
                    shape=self._params[init_tensor.name].shape,
                )
        for i in graph.input:
            # from onnx v0.2, GraphProto.input has type ValueInfoProto,
            #  and the name is 'i.name'
            i_name, i_shape, d_type, i_shape_name = get_info(i)
            if i_name in self._params:
                # i is a param instead of input
                self._num_param += 1
                self._params[i_name] = self._params.pop(i_name)
                self._nodes[i_name] = var_ddr[target](
                    i_name, shape=self._params[i_name].shape
                )
            elif i_name in self._nodes:
                continue
            else:
                self._num_input += 1
                if i_name in self._shape:
                    i_shape = self._shape.pop(i_name)
                else:
                    if "?" in str(i_shape):
                        warning_msg = (
                            "Input %s has unknown dimension shapes: %s. "
                            "Specifying static values may improve performance"
                            % (i_name, str(i_shape_name))
                        )
                        warnings.warn(warning_msg)
                if isinstance(self._dtype, dict):
                    dtype = self._dtype[i_name] if i_name in self._dtype else d_type
                else:
                    dtype = d_type
                self._nodes[i_name] = var_ddr[target](i_name, shape=i_shape)
            self._inputs[i_name] = self._nodes[i_name]
        assert (
            len(self._shape) == 0
        ), "User specified the shape for inputs that weren't found in the graph: " + str(
            self._shape
        )
        # get list of unsupported ops
        convert_map = _get_convert_map(opset, target)
        unsupported_ops = set()
        for node in graph.node:
            op_name = node.op_type
            if (
                op_name not in convert_map
                and op_name != "Constant"
                and op_name not in _identity_list
            ):
                unsupported_ops.add(op_name)
        # if unsupported_ops:
        #     msg = "The following operators are not supported for frontend ONNX: "
        #     msg += ", ".join(unsupported_ops)
        #     raise RuntimeError(msg)
        # construct nodes, nodes are stored as directed acyclic graph
        for node in graph.node:
            op_name = node.op_type
            attr = self._parse_attr(node.attribute)
            # Create and populate onnx input object.
            inputs = onnx_input()
            for i in node.input:
                if i != "":
                    inputs[i] = self._nodes[self._renames.get(i, i)]
                else:
                    inputs[i] = None
            i_name = self._parse_value_proto(node)
            node_output = self._fix_outputs(op_name, node.output)
            attr["tvm_custom"] = {}
            attr["tvm_custom"]["name"] = i_name
            attr["tvm_custom"]["num_outputs"] = len(node_output)

            op = self._convert_operator(op_name, inputs, attr, opset, target)
            outputs_num = len(node_output)
            assert (
                len(node_output) == outputs_num
            ), "Number of output mismatch {} vs {} in {}.".format(
                len(node_output), outputs_num, op_name
            )
            if outputs_num == 1:
                self._nodes[node_output[0]] = op
            else:
                for k, i in zip(list(node_output), range(len(node_output))):
                    self._nodes[k] = op[i]

        # now return the outputs
        outputs = [self._nodes[self._parse_value_proto(i)] for i in graph.output]
        outputs = outputs[0] if len(outputs) == 1 else adr.Tuple(outputs)
        # If requested, directly return the converted expressions.
        return outputs

    def _parse_value_proto(self, value_proto):
        """Parse ValueProto or raw str."""
        try:
            name = value_proto.name
        except AttributeError:
            name = value_proto
        return name

    def _parse_array(self, tensor_proto):
        np_array = get_numpy(tensor_proto).reshape(tuple(tensor_proto.dims))
        return np_array

    def _parse_attr(self, attr_proto):
        """Convert a list of AttributeProto to a dict, with names as keys."""
        attrs = {}
        for a in attr_proto:
            for f in ["f", "i", "s", "g"]:
                if a.HasField(f):
                    attrs[a.name] = getattr(a, f)
            for f in ["floats", "ints", "strings"]:
                if list(getattr(a, f)):
                    assert a.name not in attrs, "Only one type of attr is allowed"
                    attrs[a.name] = tuple(getattr(a, f))
            for f in ["t"]:
                if a.HasField(f):
                    attrs[a.name] = getattr(a, f)
            for f in ["tensors"]:
                if list(getattr(a, f)):
                    assert a.name not in attrs, "Only one type of attr is allowed"
                    attrs[a.name] = tuple(getattr(a, f))
            for f in ["graphs"]:
                if list(getattr(a, f)):
                    raise NotImplementedError("Field {} is not supported in relay.".format(f))
            if a.name not in attrs:
                raise ValueError("Cannot parse attribute: \n{}\n.".format(a))
        return attrs

    def _convert_operator(self, op_name, inputs, attrs, opset, target):
        """Convert ONNX operator into a Relay operator.
        The converter must specify conversions explicitly for incompatible name, and
        apply handlers to operator attributes.

        Parameters
        ----------
        op_name : str
            Operator name, such as Convolution, FullyConnected
        inputs : list of tvm.relay.function.Function
            List of inputs.
        attrs : dict
            Dict of operator attributes
        opset : int
            Opset version

        Returns
        -------
        sym : tvm.relay.function.Function
            Converted relay function
        """
        convert_map = _get_convert_map(opset, target)
        if op_name in convert_map:
            sym = convert_map[op_name](inputs, attrs, self._params)
        else:
            raise NotImplementedError("Operator {} not implemented.".format(op_name))
        return sym

    def _fix_outputs(self, op_name, outputs):
        """A hack to handle dropout or similar operator that have more than one out
        in ONNX.
        """
        if op_name == "Dropout":
            if len(outputs) == 1:
                return outputs
            # TODO(zhreshold): support dropout mask?
            outputs = outputs[:-1]
        return outputs


def from_onnx(model, shape=None, dtype="float16", opset=None, freeze_params=True, target="hbm"):
    """Convert a ONNX model into an equivalent Relay Function.

    ONNX graphs are represented as Python Protobuf objects.
    The companion parameters will be handled automatically.
    However, the input names from onnx graph is vague, mixing inputs and
    network weights/bias such as "1", "2"...
    For convenience, we rename the `real` input names to "input_0",
    "input_1"... And renaming parameters to "param_0", "param_1"...

    By default, ONNX defines models in terms of dynamic shapes. The ONNX importer
    retains that dynamism upon import, and the compiler attempts to convert the
    model into a static shapes at compile time. If this fails, there may still
    be dynamic operations in the model. Not all TVM kernels currently support
    dynamic shapes, please file an issue on discuss.tvm.apache.org
    if you hit an error with dynamic kernels.

    Parameters
    ----------
    model : protobuf object
        ONNX ModelProto after ONNX v1.1.0

    shape : dict of str to tuple, optional
        The input shape to the graph

    dtype : str or dict of str to str
        The input types to the graph

    opset : int, optional
        Override to autodetected opset.
        This can be helpful for some testing.

    freeze_params: bool
        If this parameter is true, the importer will take any provided
        onnx input values (weights, shapes, etc) and embed them into the relay model
        as Constants instead of variables. This allows more aggressive optimizations
        at compile time and helps in making models static if certain inputs represent
        attributes relay would traditionally consider compile-time constants.

    Returns
    -------
    mod : tvm.IRModule
        The relay module for compilation

    params : dict of str to tvm.nd.NDArray
        The parameter dict to be used by relay
    """
    try:
        import onnx

        if hasattr(onnx.checker, "check_model"):
            # try use onnx's own model checker before converting any model
            try:
                onnx.checker.check_model(model)
            except Exception as e:  # pylint: disable=c-extension-no-member, broad-except
                # the checker is a bit violent about errors, so simply print warnings here
                warnings.warn(str(e))
    except ImportError:
        pass
    g = GraphProto(shape, dtype, freeze_params)
    graph = model.graph
    if opset is None:
        try:
            opset = model.opset_import[0].version if model.opset_import else 1
        except AttributeError:
            opset = 1
    # Use the graph proto as a scope so that ops can access other nodes if needed.
    with g:
        mod = g.from_onnx(graph, opset, target)
    return mod
