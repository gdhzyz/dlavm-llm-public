# ===- graph.py ----------------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# This is the graph level of the Buddy Compiler frontend.
#
# ===---------------------------------------------------------------------------

from typing import Any, List, Optional
from types import FunctionType
import ctypes
import functools
import numpy as np

from .operation import *
from .type import *


class Graph:
    """
    Graph is a graph-level expression for the Buddy Compiler frontends.
    It acts as a model compute graph, which converts a Graph into an equivalent
    MLIR module.

    Attributes:
    - _body: List[Op]
        The sequence of operation nodes in the graph.
    - _inputs: List[TensorMeta]
        The model inputs represented as TensorMeta objects.
    - _fake_params: List[TensorMeta]
        The fake parameters represented as TensorMeta objects.
    - device: str
        The hardware for graph runtime.
    - _imported_module: Union[None, ImportedModuleType]
        The imported MLIR module after compilation, if set.
    - _ops_registry: dict
        The ops lower strategy for the graph.
    - _func_name: str
        The function name for the MLIR module.
    - _output_memref: Union[None, ctypes.POINTER]
        The memref pointer in the MLIR function output, if set.
    - _output_descriptor: Union[None, OutputDescriptorType]
        The output descriptor for the MLIR function, if set.
    - ee_: Union[None, ExecutionEngineType]
        The execution engine for the graph, if set.
    """

    def __init__(
        self,
        inputs: List[TensorMeta],
        fake_params: List[TensorMeta],
        ops_registry: dict,
        func_name: str,
        device: DeviceType = DeviceType.CPU,
        verbose=False,
    ) -> None:
        """
        Initializes the Graph.

        Args:
            inputs: List[TensorMeta]
                The model inputs represented as TensorMeta objects.
            fake_params: List[TensorMeta]
                The fake parameters represented as TensorMeta objects.
            ops_registry: dict
                The ops lower strategy for the graph.
            func_name: str
                The function name for the MLIR module.
        """
        self._body = []
        self._inputs = inputs
        self.node_table: Dict[str, Op] = {}
        self._fake_params = fake_params
        self.device = device
        self._imported_module = None
        self._verbose = verbose
        self._ops_registry = ops_registry
        self._func_name = func_name
        self._output_memref = None
        self._output_descriptor = None
        self.execution_engine = None
        self.op_groups: Dict[str, List[Op]] = {}
        self.group_map_device: Dict[str, DeviceType] = {}

    @property
    def func_name(self):
        return self._func_name
    @property
    def body(self):
        return self._body

    @body.setter
    def body(self, new_body):
        self._body = new_body

    def add_node(self, node: Op):
        """
        Adds an operation node to the graph's body.

        Parameters:
        - node: Op
            The operation node to be added to the graph.

        Returns:
        None

        Example:
        graph_instance = Graph(inputs, fake_params, ops_registry, func_name)
        op_node = Op()
        graph_instance.add_node(op_node)
        # The op_node is now part of the graph's body
        """
        self._body.append(node)
        self.node_table[node.name] = node

    def check_delete_node(self, node: Op) -> bool:
        """
        Determines if a node exists in the graph and has no child nodes.

        Args:
            node (Op): The operation node to check for deletion eligibility.

        Returns:
            bool: True if the node exists in the graph and has no children.
        """
        if not (node.name in self.node_table):
            raise KeyError("node{0} not in graph".format(node.name))

        if len(node._children) == 0:
            return True
        return False

    def delete_node(self, node: Op, parents: List[Op]):
        """
        Removes a node from the graph and updates its parent nodes accordingly.

        Args:
            node (Op): The operation node to be deleted from the graph.
            parents (List[Op]): A list of parent operation nodes that reference the node to be deleted.

        Returns:
            None
        """
        for i in parents:
            i._children.remove(node.name)
        node.args.clear()
        node.kwargs.clear()
        node._children.clear()
        self._body.remove(node)
        self.node_table.pop(node.name)

    def displace_node(self, node: Op, newnode: Op):
        """
        Replaces an existing node with a new node in the graph.

        Args:
            node (Op): The operation node to be replaced.
            newnode (Op): The new operation node that will replace the existing node.

        Returns:
            None
        """
        newnode._arguments = node.args
        newnode._keyword_arguments = node.kwargs
        newnode._tensor_meta = node.tensor_meta
        newnode._op_type = node._op_type

        for i in node._children:
            newnode.add_children(i)
        users = [self.node_table[i] for i in node._children]
        for user in users:
            if node.name in user._parents:
                user._parents[user._parents.index(node.name)] = newnode.name
            user.args[user.args.index(node.name)] = newnode.name
        node._children.clear()
        # deal with parents+args
        for i in node._parents:
            newnode.add_parent(i)
        parents = [self.node_table[i] for i in node._parents]
        for parent in parents:
            parent._children[parent._children.index(node.name)] = newnode.name
        node._parents.clear()
        # update node table
        self._body[self._body.index(node)] = newnode
        self.node_table.pop(node.name)
        self.node_table[newnode.name] = newnode

    def init_op_group(self):
        """
        Initializes operation groups within the graph.

        Returns:
        - None
        """
        for i, op in enumerate(self._body):
            if isinstance(op, PlaceholderOp) or isinstance(op, OutputOp):
                continue
            group = [op]
            subgraph_name = "subgraph{}".format(i)
            self.group_map_device[subgraph_name] = DeviceType.CPU
            self.op_groups[subgraph_name] = group

    def fuse_ops(self, pattern_list: List[FunctionType]):
        """
        Fuse operations in the graph based on provided fusion patterns.

        Args:
        - pattern_list (List[FunctionType]): A list of functions representing
        fusion patterns.

        Returns:
        - None
        """
        # TODO: discuss two fuse strategy
        # 1. fuse ops adapt for DSA(hardware dependent)
        # 2. common fuse strategy(hardware independent)

        # Apply fusion patterns
        for pattern_func in pattern_list:
            pattern_func(self)

    def perform(self, func_list: List[FunctionType]):
        """
        Perform a series of transformations on the graph using the provided list
        of functions.

        Args:
        - func_list (List[FunctionType]): A list of functions representing
        transformations to be applied to the graph.

        Returns:
        - None
        """
        for transform_func in func_list:
            transform_func(self)

    def replace_users_with(self, node: Op, target: Op):
        childs = [self.node_table[i] for i in node._children]
        target._children += node._children
        node._children.clear()
        for c in childs:
            c._parents = [target.name if i == node.name else i for i in c._parents]

